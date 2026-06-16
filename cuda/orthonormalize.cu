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

#include <cmath>
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

// Round the f64 projection residual to f32, store it, and accumulate the
// squared norm of the *rounded* value (one block per column). ld is the column
// stride (== n) shared by `in` and `out`.
__global__ void round_colnorm_kernel(const double *in, float *out, size_t n,
                                     size_t ld, double *ds)
{
    const size_t j = blockIdx.x;
    const double *cin  = in  + j * ld;
    float        *cout = out + j * ld;
    double partial = 0.0;
    for (size_t i = threadIdx.x; i < n; i += blockDim.x) {
        const float v = (float)cin[i];
        cout[i] = v;
        const double vb = (double)v;
        partial += vb * vb;
    }
    partial = blockReduceSum(partial);
    if (threadIdx.x == 0) {
        ds[j] = partial;
    }
}

// DGKS retry commit: same as round_colnorm_kernel but only for flagged columns.
// Non-flagged columns are left untouched (uniform early-return per block).
__global__ void round_colnorm_masked_kernel(const double *in, float *out,
                                            size_t n, size_t ld,
                                            double *ds, const int *flags)
{
    const size_t j = blockIdx.x;
    if (!flags[j]) {
        return;
    }
    const double *cin  = in  + j * ld;
    float        *cout = out + j * ld;
    double partial = 0.0;
    for (size_t i = threadIdx.x; i < n; i += blockDim.x) {
        const float v = (float)cin[i];
        cout[i] = v;
        const double vb = (double)v;
        partial += vb * vb;
    }
    partial = blockReduceSum(partial);
    if (threadIdx.x == 0) {
        ds[j] = partial;
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
// sqrt then a single cast, matching the CPU normalize step.
__global__ void scale_kernel(const double *ds, float *scale, size_t p, double tol)
{
    const size_t j = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= p) {
        return;
    }
    const double s = ds[j];
    scale[j] = (s > tol) ? (float)(1.0 / sqrt(s + tol)) : 0.0f;
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

    size_t n         = 0;
    size_t max_terms = 0;
    size_t ldT       = 0;   // == max_terms
    double tol       = 0.0;
    size_t m_synced  = 0;   // columns of B/Bo currently resident on device

    // resident basis
    float  *dB      = nullptr;  // (n, max_terms) col-major, ld n
    double *dBo_f64 = nullptr;  // (n, max_terms) col-major, ld n

    // per-call scratch (sized for the worst case p == max_terms)
    float  *dBx_f32 = nullptr;  // (n, max_terms) col-major, ld n
    double *dBx_f64 = nullptr;  // (n, max_terms) col-major, ld n
    double *dT      = nullptr;  // (max_terms, max_terms) col-major, ld max_terms
    float  *dx      = nullptr;  // (n)
    int    *dmask   = nullptr;  // (max_terms)
    double *ds      = nullptr;  // (max_terms)
    double *dtnorm2 = nullptr;  // (max_terms)
    float  *dscale  = nullptr;  // (max_terms)
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

        const size_t nmt = n * max_terms;
        CUDA_CHECK(cudaMalloc(&ctx->dB,      nmt * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&ctx->dBo_f64, nmt * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&ctx->dBx_f32, nmt * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&ctx->dBx_f64, nmt * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&ctx->dT,      max_terms * max_terms * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&ctx->dx,      n * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&ctx->dmask,   max_terms * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&ctx->ds,      max_terms * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&ctx->dtnorm2, max_terms * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&ctx->dscale,  max_terms * sizeof(float)));
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
    cudaFree(ctx->dBx_f32);
    cudaFree(ctx->dBx_f64);
    cudaFree(ctx->dT);
    cudaFree(ctx->dx);
    cudaFree(ctx->dmask);
    cudaFree(ctx->ds);
    cudaFree(ctx->dtnorm2);
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

void orthonormalize(Context *ctx, size_t m, size_t p,
                    const float *x, const int *mask,
                    float *Bx, size_t ldBx,
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
    CUBLAS_CHECK(cublasDgemm(ctx->handle, CUBLAS_OP_T, CUBLAS_OP_N,
                             (int)m, (int)p, (int)n,
                             &one, ctx->dBo_f64, (int)n,
                             ctx->dBx_f64, (int)n,
                             &zero, ctx->dT, (int)ldT));

    // Phase 2a: Bx -= Bo · T  (in place, f64), then round to f32 + column norms.
    CUBLAS_CHECK(cublasDgemm(ctx->handle, CUBLAS_OP_N, CUBLAS_OP_N,
                             (int)n, (int)p, (int)m,
                             &neg_one, ctx->dBo_f64, (int)n,
                             ctx->dT, (int)ldT,
                             &one, ctx->dBx_f64, (int)n));
    round_colnorm_kernel<<<(unsigned)p, block, 0, s>>>(
        ctx->dBx_f64, ctx->dBx_f32, n, n, ctx->ds);
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
        // Re-orthogonalize from the rounded Bx: widen, recompute T, reproject.
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
        round_colnorm_masked_kernel<<<(unsigned)p, block, 0, s>>>(
            ctx->dBx_f64, ctx->dBx_f32, n, n, ctx->ds, ctx->dflags);
        check_launch();
        if (dgks_counter) {
            dgks_counter->fetch_add(fired, std::memory_order_relaxed);
        }
    }

    // Phase 2c: normalize.
    scale_kernel<<<(unsigned)((p + BLOCK - 1) / BLOCK), block, 0, s>>>(
        ctx->ds, ctx->dscale, p, tol);
    check_launch();
    normalize_kernel<<<grid_np, block, 0, s>>>(ctx->dBx_f32, ctx->dscale, n);
    check_launch();

    // Copy Bx back to the host (device col-major ld n -> host col-major ld ldBx).
    CUDA_CHECK(cudaMemcpy2DAsync(
        Bx, ldBx * sizeof(float),
        ctx->dBx_f32, n * sizeof(float),
        n * sizeof(float), p,
        cudaMemcpyDeviceToHost, s));
    CUDA_CHECK(cudaStreamSynchronize(s));
}

}  // namespace cuda
}  // namespace mars
