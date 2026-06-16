/*
 *  CUDA implementation of mars::orthonormalize() for the linear_only forward
 *  pass. See cuda/mars_cuda.h for the host-facing contract and kernels.cc for
 *  the CPU oracle this must match (within the ~1e-5 f32 storage floor).
 *
 *  Precision contract (load-bearing): the T = Boᵀ·Bx contraction accumulates in
 *  f64 (cublasDgemm on f64 copies of Bo/Bx). f32 accumulation corrupts the
 *  near-zero projection entries via cancellation (1e-2…4.6e-1 rel error -- see
 *  CLAUDE.md). The per-element f32 *rounding* points are matched to the CPU
 *  kernel exactly: the B·x product is rounded to f32 before the GEMM widen, the
 *  projection residual is rounded to f32 before its squared norm accumulates,
 *  and the normalize scale is computed in f64 (1/sqrt(s+tol), not rsqrtf).
 */

#include "../kernels.h"   // mars::DGKS_GATE_RATIO_SQ
#include "mars_cuda.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>    // fprintf
#include <cstdlib>   // getenv
#include <stdexcept>
#include <string>
#include <vector>

namespace mars {
namespace cuda {

namespace {

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err_ = (call);                                             \
        if (err_ != cudaSuccess) {                                             \
            throw std::runtime_error(std::string("CUDA error: ") +             \
                                     cudaGetErrorString(err_) + " at " +       \
                                     __FILE__ ":" + std::to_string(__LINE__)); \
        }                                                                      \
    } while (0)

#define CUBLAS_CHECK(call)                                                     \
    do {                                                                       \
        cublasStatus_t st_ = (call);                                           \
        if (st_ != CUBLAS_STATUS_SUCCESS) {                                    \
            throw std::runtime_error(std::string("cuBLAS error ") +            \
                                     std::to_string((int)st_) + " at " +       \
                                     __FILE__ ":" + std::to_string(__LINE__)); \
        }                                                                      \
    } while (0)

constexpr int BLOCK = 256;

// f64 GEMM via cublasGemmEx so the compute type is explicit per call: native
// CUBLAS_COMPUTE_64F (d884 DMMA) or CUBLAS_COMPUTE_64F_EMULATED_FIXEDPOINT
// (integer-tensor-core emulation on Blackwell). All operands stay CUDA_R_64F.
inline void gemm64(cublasHandle_t h, cublasComputeType_t ct,
                   cublasOperation_t ta, cublasOperation_t tb,
                   int m, int n, int k, const double *alpha,
                   const double *A, int lda, const double *B, int ldb,
                   const double *beta, double *C, int ldc)
{
    CUBLAS_CHECK(cublasGemmEx(h, ta, tb, m, n, k,
                              alpha, A, CUDA_R_64F, lda,
                              B, CUDA_R_64F, ldb,
                              beta, C, CUDA_R_64F, ldc,
                              ct, CUBLAS_GEMM_DEFAULT));
}

// --- reductions -------------------------------------------------------------

__inline__ __device__ double warpReduceSum(double v)
{
    for (int off = warpSize / 2; off > 0; off >>= 1) {
        v += __shfl_down_sync(0xffffffffu, v, off);
    }
    return v;
}

__inline__ __device__ double blockReduceSum(double v)
{
    __shared__ double shared[32];  // one slot per warp (max 1024 threads = 32 warps)
    const int lane = threadIdx.x % warpSize;
    const int wid  = threadIdx.x / warpSize;
    v = warpReduceSum(v);
    if (lane == 0) {
        shared[wid] = v;
    }
    __syncthreads();
    const int nwarps = (blockDim.x + warpSize - 1) / warpSize;
    v = (threadIdx.x < nwarps) ? shared[lane] : 0.0;
    if (wid == 0) {
        v = warpReduceSum(v);
    }
    return v;
}

// Reduce two independent accumulators with a single barrier (thread 0 gets both
// block sums). Used by the fused round/norm/dot kernel.
__inline__ __device__ void blockReduceSum2(double &a, double &b)
{
    __shared__ double sa[32];
    __shared__ double sb[32];
    const int lane = threadIdx.x % warpSize;
    const int wid  = threadIdx.x / warpSize;
    a = warpReduceSum(a);
    b = warpReduceSum(b);
    if (lane == 0) {
        sa[wid] = a;
        sb[wid] = b;
    }
    __syncthreads();
    const int nwarps = (blockDim.x + warpSize - 1) / warpSize;
    a = (threadIdx.x < nwarps) ? sa[lane] : 0.0;
    b = (threadIdx.x < nwarps) ? sb[lane] : 0.0;
    if (wid == 0) {
        a = warpReduceSum(a);
        b = warpReduceSum(b);
    }
}

// --- kernels ----------------------------------------------------------------

// Bx[i,j] = B[i, mask[j]] * x[i], stored both as f32 (the kernel output) and as
// the f64 widening of that rounded f32 value (the cublasDgemm input). Matches
// the CPU: an f32 multiply, stored f32, then upcast at the GEMM load.
__global__ void fill_kernel(const float *B, size_t n, const float *x,
                            const int *mask, float *bx32, double *bx64)
{
    const size_t j = blockIdx.y;
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    const int   col = mask[j];
    const float v   = B[(size_t)col * n + i] * x[i];
    const size_t idx = j * n + i;
    bx32[idx] = v;
    bx64[idx] = (double)v;
}

// Fused round + norm + dot, parallelized over both columns and rows.
//
// For each column j (= blockIdx.y), tiles of rows are split across blockIdx.x
// blocks (grid-stride), so the whole GPU is filled instead of one block/column.
// Each block: rounds the f64 residual to f32 and stores it, then block-reduces
// two partials -- Σ(round)² and (if want_dot) Σ(round)·y -- and atomic-adds them
// to ds[j] / ybx_raw[j] (which the caller must zero first). f64 throughout, so
// the f32-store-rounding semantics match the CPU kernel exactly.
__global__ void round_reduce_kernel(const double *in, float *out,
                                    const float *y, size_t n,
                                    double *ds, double *ybx_raw, int want_dot)
{
    const size_t j = blockIdx.y;
    const double *cin  = in  + j * n;
    float        *cout = out + j * n;
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    double s_norm = 0.0, s_dot = 0.0;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < n; i += stride) {
        const float  v  = (float)cin[i];
        cout[i] = v;
        const double vb = (double)v;
        s_norm += vb * vb;
        if (want_dot) {
            s_dot += vb * (double)y[i];
        }
    }
    blockReduceSum2(s_norm, s_dot);
    if (threadIdx.x == 0) {
        atomicAdd(&ds[j], s_norm);
        if (want_dot) {
            atomicAdd(&ybx_raw[j], s_dot);
        }
    }
}

// DGKS retry commit (rare): recompute ds[j]/ybx_raw[j] for flagged columns only,
// one block per column (direct write, no atomics -- a single block does the full
// column reduction). Non-flagged columns are left untouched (uniform return).
__global__ void round_reduce_masked_kernel(const double *in, float *out,
                                           const float *y, size_t n,
                                           double *ds, double *ybx_raw,
                                           const int *flags, int want_dot)
{
    const size_t j = blockIdx.x;
    if (!flags[j]) {
        return;
    }
    const double *cin  = in  + j * n;
    float        *cout = out + j * n;
    double s_norm = 0.0, s_dot = 0.0;
    for (size_t i = threadIdx.x; i < n; i += blockDim.x) {
        const float  v  = (float)cin[i];
        cout[i] = v;
        const double vb = (double)v;
        s_norm += vb * vb;
        if (want_dot) {
            s_dot += vb * (double)y[i];
        }
    }
    blockReduceSum2(s_norm, s_dot);
    if (threadIdx.x == 0) {
        ds[j] = s_norm;
        if (want_dot) {
            ybx_raw[j] = s_dot;
        }
    }
}

// out[j] = ||M[:,j]||^2 over `rows` f64 entries (one block per column).
__global__ void colsumsq_f64_kernel(const double *M, size_t ld, size_t rows,
                                    double *out)
{
    const size_t j = blockIdx.x;
    const double *col = M + j * ld;
    double partial = 0.0;
    for (size_t i = threadIdx.x; i < rows; i += blockDim.x) {
        const double v = col[i];
        partial += v * v;
    }
    partial = blockReduceSum(partial);
    if (threadIdx.x == 0) {
        out[j] = partial;
    }
}

__global__ void widen_kernel(const float *in, double *out, size_t total)
{
    const size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) {
        out[idx] = (double)in[idx];
    }
}

// scale[j] = (ds[j] > tol) ? (float)(1/sqrt(ds[j]+tol)) : 0  -- f64 reciprocal
// sqrt then a single cast, matching the CPU normalize step. When want_ybx, also
// fold the scale into the linear ΔSSE input: ybx[j] = scale[j]·(Σ Bx·y). This
// is the normalized-column dot product without a separate normalize pass:
// ‖Bx‖·scale applied once to the reduced dot instead of per element.
__global__ void scale_fold_kernel(const double *ds, float *scale,
                                  const double *ybx_raw, double *ybx,
                                  size_t p, double tol, int want_ybx)
{
    const size_t j = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= p) {
        return;
    }
    const double s  = ds[j];
    const float  sc = (s > tol) ? (float)(1.0 / sqrt(s + tol)) : 0.0f;
    scale[j] = sc;
    if (want_ybx) {
        ybx[j] = (double)sc * ybx_raw[j];
    }
}

__global__ void normalize_kernel(float *bx32, const float *scale, size_t n)
{
    const size_t j = blockIdx.y;
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    bx32[j * n + i] *= scale[j];
}

inline void check_launch()
{
    CUDA_CHECK(cudaGetLastError());
}

}  // namespace

// --- Context ----------------------------------------------------------------

struct Context {
    cublasHandle_t handle = nullptr;
    cudaStream_t   stream = nullptr;
    cublasComputeType_t compute_type = CUBLAS_COMPUTE_64F;  // or _EMULATED_FIXEDPOINT

    size_t n         = 0;
    size_t max_terms = 0;
    size_t ldT       = 0;   // == max_terms
    double tol       = 0.0;
    size_t m_synced  = 0;   // columns of B/Bo currently resident on device

    // resident basis + target
    float  *dB      = nullptr;  // (n, max_terms) col-major, ld n
    double *dBo_f64 = nullptr;  // (n, max_terms) col-major, ld n
    float  *dy      = nullptr;  // (n) normalized target
    const float *target_ptr = nullptr;  // host y pointer last uploaded to dy

    // per-call scratch (sized for the worst case p == max_terms)
    float  *dBx_f32 = nullptr;  // (n, max_terms) col-major, ld n
    double *dBx_f64 = nullptr;  // (n, max_terms) col-major, ld n
    double *dT      = nullptr;  // (max_terms, max_terms) col-major, ld max_terms
    float  *dx      = nullptr;  // (n)
    int    *dmask   = nullptr;  // (max_terms)
    double *ds       = nullptr;  // (max_terms)
    double *dtnorm2  = nullptr;  // (max_terms)
    double *dybx_raw = nullptr;  // (max_terms) Σ Bx·y on the un-normalized column
    double *dybx     = nullptr;  // (max_terms) scale·ybx_raw -- the ΔSSE input
    float  *dscale   = nullptr;  // (max_terms)
    int    *dflags  = nullptr;  // (max_terms)
    float  *dBo_col = nullptr;  // (n) Bo-column widen staging

    // host staging
    std::vector<float>  h_stage;    // (n) gather one Bo column
    std::vector<double> h_ds;       // (max_terms)
    std::vector<double> h_tnorm2;   // (max_terms)
    std::vector<int>    h_flags;    // (max_terms)
};

Context *context_create(size_t n, size_t max_terms, double tol)
{
    Context *ctx = new Context();
    try {
        ctx->n         = n;
        ctx->max_terms = max_terms;
        ctx->ldT       = max_terms;
        ctx->tol       = tol;

        CUDA_CHECK(cudaStreamCreate(&ctx->stream));
        CUBLAS_CHECK(cublasCreate(&ctx->handle));
        CUBLAS_CHECK(cublasSetStream(ctx->handle, ctx->stream));

        // Optional: run the f64 GEMMs via cuBLAS FP64 emulation on the integer
        // tensor cores (Blackwell). Opt-in with MARS_CUDA_FP64_EMULATE=1. The
        // fixed-point scheme reconstructs the f64 result (not a lossy f32/TF32
        // approximation), so it should clear the cancellation gate -- but the
        // precision-stress + DGKS tests must confirm it. Native f64 (d884 DMMA)
        // remains the default.
        const char *emu = std::getenv("MARS_CUDA_FP64_EMULATE");
        if (emu && emu[0] == '1') {
            // Request emulation per-GEMM via the compute type (cublasGemmEx);
            // EAGER tells cuBLAS to use emulation whenever the compute type asks.
            ctx->compute_type = CUBLAS_COMPUTE_64F_EMULATED_FIXEDPOINT;
            CUBLAS_CHECK(cublasSetEmulationStrategy(ctx->handle,
                                                    CUBLAS_EMULATION_STRATEGY_EAGER));
            std::fprintf(stderr,
                "[mars-cuda] FP64 emulation enabled: compute_type=%d, strategy=EAGER\n",
                (int)CUBLAS_COMPUTE_64F_EMULATED_FIXEDPOINT);
        }

        const size_t nmt = n * max_terms;
        CUDA_CHECK(cudaMalloc(&ctx->dB,      nmt * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&ctx->dBo_f64, nmt * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&ctx->dy,      n * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&ctx->dBx_f32, nmt * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&ctx->dBx_f64, nmt * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&ctx->dT,      max_terms * max_terms * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&ctx->dx,      n * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&ctx->dmask,   max_terms * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&ctx->ds,       max_terms * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&ctx->dtnorm2,  max_terms * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&ctx->dybx_raw, max_terms * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&ctx->dybx,     max_terms * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&ctx->dscale,   max_terms * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&ctx->dflags,  max_terms * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&ctx->dBo_col, n * sizeof(float)));

        ctx->h_stage.resize(n);
        ctx->h_ds.resize(max_terms);
        ctx->h_tnorm2.resize(max_terms);
        ctx->h_flags.resize(max_terms);
    } catch (...) {
        context_destroy(ctx);
        throw;
    }
    return ctx;
}

void context_destroy(Context *ctx)
{
    if (!ctx) {
        return;
    }
    cudaFree(ctx->dB);
    cudaFree(ctx->dBo_f64);
    cudaFree(ctx->dy);
    cudaFree(ctx->dBx_f32);
    cudaFree(ctx->dBx_f64);
    cudaFree(ctx->dT);
    cudaFree(ctx->dx);
    cudaFree(ctx->dmask);
    cudaFree(ctx->ds);
    cudaFree(ctx->dtnorm2);
    cudaFree(ctx->dybx_raw);
    cudaFree(ctx->dybx);
    cudaFree(ctx->dscale);
    cudaFree(ctx->dflags);
    cudaFree(ctx->dBo_col);
    if (ctx->handle) {
        cublasDestroy(ctx->handle);
    }
    if (ctx->stream) {
        cudaStreamDestroy(ctx->stream);
    }
    delete ctx;
}

void context_sync_basis(Context *ctx, size_t m,
                        const float *B,  size_t ldB,
                        const float *Bo, size_t ldBo)
{
    if (m <= ctx->m_synced) {
        return;
    }
    const size_t n     = ctx->n;
    const size_t start = ctx->m_synced;
    const size_t cnt   = m - start;

    // B: columns [start, m) -> dB columns [start, m). Both col-major; cudaMemcpy2D
    // tolerates ldB != n (it is n in practice -> a single contiguous block).
    CUDA_CHECK(cudaMemcpy2DAsync(
        ctx->dB + start * n, n * sizeof(float),
        B + start * ldB,     ldB * sizeof(float),
        n * sizeof(float),   cnt,
        cudaMemcpyHostToDevice, ctx->stream));

    // Bo: host is row-major (stride ldBo). Gather each new column, upload, widen
    // to f64 into the device column-major dBo_f64. One column at a time with a
    // stream sync so the reusable host stage buffer is safe to overwrite. Rare
    // (≈ one column per epoch), so the per-column cost is amortized.
    const dim3 grid_col((n + BLOCK - 1) / BLOCK);
    for (size_t c = start; c < m; ++c) {
        float *stage = ctx->h_stage.data();
        for (size_t i = 0; i < n; ++i) {
            stage[i] = Bo[i * ldBo + c];
        }
        CUDA_CHECK(cudaMemcpyAsync(ctx->dBo_col, stage, n * sizeof(float),
                                   cudaMemcpyHostToDevice, ctx->stream));
        widen_kernel<<<grid_col, BLOCK, 0, ctx->stream>>>(
            ctx->dBo_col, ctx->dBo_f64 + c * n, n);
        check_launch();
        CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
    }

    ctx->m_synced = m;
}

void context_set_target(Context *ctx, const float *y)
{
    if (ctx->target_ptr == y) {
        return;  // y is fixed for a fit -- upload once
    }
    CUDA_CHECK(cudaMemcpyAsync(ctx->dy, y, ctx->n * sizeof(float),
                               cudaMemcpyHostToDevice, ctx->stream));
    CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
    ctx->target_ptr = y;
}

void orthonormalize(Context *ctx, size_t m, size_t p,
                    const float *x, const int *mask,
                    float *Bx, size_t ldBx,
                    double *ybx,
                    std::atomic<long> *dgks_counter)
{
    if (p == 0) {
        return;
    }
    const size_t n   = ctx->n;
    const size_t ldT = ctx->ldT;
    const double tol = ctx->tol;
    cudaStream_t   s = ctx->stream;
    const double one = 1.0, zero = 0.0, neg_one = -1.0;

    // Upload the candidate column and the active-basis mask.
    CUDA_CHECK(cudaMemcpyAsync(ctx->dx, x, n * sizeof(float),
                               cudaMemcpyHostToDevice, s));
    CUDA_CHECK(cudaMemcpyAsync(ctx->dmask, mask, p * sizeof(int),
                               cudaMemcpyHostToDevice, s));

    // Phase 0: Bx = B[:,mask] .* x  (f32 store + f64 widen).
    const dim3 block(BLOCK);
    const dim3 grid_np((n + BLOCK - 1) / BLOCK, p);
    fill_kernel<<<grid_np, block, 0, s>>>(ctx->dB, n, ctx->dx, ctx->dmask,
                                          ctx->dBx_f32, ctx->dBx_f64);
    check_launch();

    // Phase 1: T = Boᵀ · Bx   (m×p, f64).
    gemm64(ctx->handle, ctx->compute_type, CUBLAS_OP_T, CUBLAS_OP_N,
           (int)m, (int)p, (int)n,
           &one, ctx->dBo_f64, (int)n,
           ctx->dBx_f64, (int)n,
           &zero, ctx->dT, (int)ldT);

    // Phase 2a: Bx -= Bo · T (in place, f64), then a single fused pass that
    // rounds to f32, stores it, and reduces both ||Bx[:,j]||² (ds) and Σ Bx·y
    // (ybx_raw). The reduction is split over rows AND columns (atomic f64
    // partials), so the whole GPU is used -- not one block per column.
    gemm64(ctx->handle, ctx->compute_type, CUBLAS_OP_N, CUBLAS_OP_N,
           (int)n, (int)p, (int)m,
           &neg_one, ctx->dBo_f64, (int)n,
           ctx->dT, (int)ldT,
           &one, ctx->dBx_f64, (int)n);
    const int want_dot = (ybx != nullptr) ? 1 : 0;
    unsigned tiles = (unsigned)std::min<size_t>(64, (n + BLOCK - 1) / BLOCK);
    if (tiles == 0) {
        tiles = 1;
    }
    const dim3 grid_reduce(tiles, (unsigned)p);
    CUDA_CHECK(cudaMemsetAsync(ctx->ds, 0, p * sizeof(double), s));
    if (want_dot) {
        CUDA_CHECK(cudaMemsetAsync(ctx->dybx_raw, 0, p * sizeof(double), s));
    }
    round_reduce_kernel<<<grid_reduce, block, 0, s>>>(
        ctx->dBx_f64, ctx->dBx_f32, ctx->dy, n, ctx->ds, ctx->dybx_raw, want_dot);
    check_launch();

    // Phase 2b: DGKS gate. s[j] > tol && s[j]*9 < ||T[:,j]||^2 -> retry.
    colsumsq_f64_kernel<<<(unsigned)p, block, 0, s>>>(ctx->dT, ldT, m, ctx->dtnorm2);
    check_launch();
    CUDA_CHECK(cudaMemcpyAsync(ctx->h_ds.data(), ctx->ds, p * sizeof(double),
                               cudaMemcpyDeviceToHost, s));
    CUDA_CHECK(cudaMemcpyAsync(ctx->h_tnorm2.data(), ctx->dtnorm2,
                               p * sizeof(double), cudaMemcpyDeviceToHost, s));
    CUDA_CHECK(cudaStreamSynchronize(s));

    long fired = 0;
    for (size_t j = 0; j < p; ++j) {
        const bool flag = ctx->h_ds[j] > tol &&
                          ctx->h_ds[j] * mars::DGKS_GATE_RATIO_SQ < ctx->h_tnorm2[j];
        ctx->h_flags[j] = flag ? 1 : 0;
        fired += flag ? 1 : 0;
    }

    if (fired > 0) {
        CUDA_CHECK(cudaMemcpyAsync(ctx->dflags, ctx->h_flags.data(),
                                   p * sizeof(int), cudaMemcpyHostToDevice, s));
        // Re-orthogonalize from the rounded Bx: widen, recompute T, reproject,
        // then recompute ds/ybx_raw for the flagged columns only.
        const size_t total = n * p;
        const dim3 grid_total((unsigned)((total + BLOCK - 1) / BLOCK));
        widen_kernel<<<grid_total, block, 0, s>>>(ctx->dBx_f32, ctx->dBx_f64, total);
        check_launch();
        CUBLAS_CHECK(cublasDgemm(ctx->handle, CUBLAS_OP_T, CUBLAS_OP_N,
                                 (int)m, (int)p, (int)n,
                                 &one, ctx->dBo_f64, (int)n,
                                 ctx->dBx_f64, (int)n,
                                 &zero, ctx->dT, (int)ldT));
        CUBLAS_CHECK(cublasDgemm(ctx->handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                 (int)n, (int)p, (int)m,
                                 &neg_one, ctx->dBo_f64, (int)n,
                                 ctx->dT, (int)ldT,
                                 &one, ctx->dBx_f64, (int)n));
        round_reduce_masked_kernel<<<(unsigned)p, block, 0, s>>>(
            ctx->dBx_f64, ctx->dBx_f32, ctx->dy, n,
            ctx->ds, ctx->dybx_raw, ctx->dflags, want_dot);
        check_launch();
        if (dgks_counter) {
            dgks_counter->fetch_add(fired, std::memory_order_relaxed);
        }
    }

    // Phase 2c: scale, and fold it into ybx (= scale·Σ Bx·y -- the normalized
    // column dot, no separate normalize pass needed for the linear ΔSSE).
    scale_fold_kernel<<<(unsigned)((p + BLOCK - 1) / BLOCK), block, 0, s>>>(
        ctx->ds, ctx->dscale, ctx->dybx_raw, ctx->dybx, p, tol, want_dot);
    check_launch();
    if (ybx) {
        CUDA_CHECK(cudaMemcpyAsync(ybx, ctx->dybx, p * sizeof(double),
                                   cudaMemcpyDeviceToHost, s));
    }

    // Only when the caller needs the matrix itself (hinge sweep) do we physically
    // normalize and copy Bx back. In linear_only Bx is nullptr and never leaves
    // the device, so the normalize pass is skipped entirely.
    if (Bx) {
        normalize_kernel<<<grid_np, block, 0, s>>>(ctx->dBx_f32, ctx->dscale, n);
        check_launch();
        CUDA_CHECK(cudaMemcpy2DAsync(
            Bx, ldBx * sizeof(float),
            ctx->dBx_f32, n * sizeof(float),
            n * sizeof(float), p,
            cudaMemcpyDeviceToHost, s));
    }
    CUDA_CHECK(cudaStreamSynchronize(s));
}

}  // namespace cuda
}  // namespace mars
