/*
 *  CUDA implementation of mars::orthonormalize() for the linear_only forward
 *  pass. See cuda/mars_cuda.h for the host-facing contract and kernels.cc for
 *  the CPU oracle this must match (within the ~1e-5 f32 storage floor).
 *
 *  Precision contract (load-bearing): the near-zero entries of T = Boᵀ·Bx
 *  cancel, so T must be accumulated in f64 -- a single f32 GEMM over K=n gives
 *  1e-2…4.6e-1 relative error (see CLAUDE.md). The default path therefore
 *  splits K into chunks of MARS_CUDA_KCHUNK rows, runs each chunk as an f32
 *  cublasSgemm, and sums the chunk results into the f64 T; only the
 *  within-chunk f32 error (~sqrt(kchunk)·eps_f32) remains. MARS_CUDA_KCHUNK=0
 *  selects the native-f64 cublasDgemm reference, bit-reproducible against the
 *  CPU kernel. The projection Bx -= Bo·T runs as an f32 sgemm whose result is
 *  subtracted from the f64 B·x inside the fused round/reduce pass. The DGKS
 *  retry (cancelling columns) runs entirely in f64 and restarts each flagged
 *  column from its f64 B·x: on such a column the f32 projection leaves
 *  out-of-span noise ~eps_f32·√m·‖T‖, i.e. eps_f32·√m·κ relative to the
 *  residual, which a second projection cannot remove (ΔSSE was off by up to
 *  tens of percent at m=400, κ>30, in a numpy model of this path). The per-element f32
 *  rounding points match the CPU kernel: the B·x product is rounded to f32
 *  before the GEMM, the residual is rounded to f32 before its squared norm
 *  accumulates, and the normalize scale is 1/sqrt(s+tol) in f64.
 */

#include "../kernels.h"   // mars::DGKS_GATE_RATIO_SQ
#include "context.cuh"    // Context, CUDA_CHECK, reductions (shared with hinge_sweep.cu)

#include <algorithm>
#include <chrono>    // host-side phase timing (MARS_CUDA_PROFILE)
#include <cmath>
#include <cstdio>    // fprintf
#include <cstdlib>   // getenv
#include <stdexcept>
#include <string>
#include <vector>

namespace mars {
namespace cuda {

namespace {

// --- kernels ----------------------------------------------------------------

// Bx[i,j] = B[i, mask[j]] * x[i], stored both as f32 (the GEMM operand and the
// kernel output) and as the f64 widening of that rounded value (the minuend of
// the fused projection subtract, and the f64 reference GEMM's operand). Matches
// the CPU: an f32 multiply, stored f32, then upcast at the load.
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

// Scale one resident candidate column in place: col[:] *= s. A distinct f32
// pass (not folded into fill_batch) so the candidate is rounded to f32 before
// multiplying by B -- matching the CPU's x = X·s then B·x ordering. Run once per
// column when it first becomes resident (context_sync_xcols).
__global__ void scale_col_kernel(float *col, size_t n, float s)
{
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        col[i] *= s;
    }
}

// Batched fill: Bx[:,j] = B[:, src_basis[j]] .* Xcand[:, src_xcol[j]], stacking
// candidate columns from many X-columns (which share Bo) into one matrix. Each
// output column j names its parent basis column and its (block-local) candidate
// X column. f32 store + f64 widen, as in fill_kernel.
__global__ void fill_batch_kernel(const float *B, size_t n, const float *Xcand,
                                  const int *src_xcol, const int *src_basis,
                                  float *bx32, double *bx64)
{
    const size_t j = blockIdx.y;
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    const int k  = src_basis[j];
    const int xc = src_xcol[j];
    const float v = B[(size_t)k * n + i] * Xcand[(size_t)xc * n + i];
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
// `proj` (optional, n×P, ld n): when non-null, the f32 projection to subtract
// in f64 first (residual = in - (double)proj) -- fuses the f32-projection
// subtract into this pass so it isn't a separate n×P read-modify-write.
__global__ void round_reduce_kernel(const double *in, float *out,
                                    const float *y, size_t n,
                                    double *ds, double *ybx_raw, int want_dot,
                                    const float *proj)
{
    const size_t j = blockIdx.y;
    const double *cin   = in   + j * n;
    float        *cout  = out  + j * n;
    const float  *cproj = proj ? proj + j * n : nullptr;
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    double s_norm = 0.0, s_dot = 0.0;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
            i < n; i += stride) {
        const double r  = cproj ? (cin[i] - (double)cproj[i]) : cin[i];
        const float  v  = (float)r;
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

// The source index of compact column c in an f64 re-projection chunk starting
// at `base`: idx[base + c], or base + c when idx is null (identity).
__device__ __forceinline__ size_t sel_idx(const int *idx, size_t base, size_t c)
{
    return idx ? (size_t)idx[base + c] : base + c;
}

// DGKS retry, refill half: recompute the f64 B·x of each flagged column into a
// COMPACT buffer -- dst[:,c] = (double)(B[:, basis[j]] * x_j), j = sel_idx(c),
// with x_j = X + xcol[j]*n (batched path) or X itself (per-eval path, xcol null).
// The same f32 multiply as fill_kernel / fill_batch_kernel, so bit-identical to
// the minuend the main pass used. The retry then runs at width `fired`, so its
// cost scales with the cancelling columns, not the batch. blockIdx.y = compact c.
__global__ void refill_widen_kernel(const float *B, size_t n, const float *X,
                                    const int *xcol, const int *basis,
                                    const int *idx, size_t base, double *dst)
{
    const size_t c = blockIdx.y;
    const size_t j = sel_idx(idx, base, c);
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    const float *x = xcol ? X + (size_t)xcol[j] * n : X;
    const float  v = B[(size_t)basis[j] * n + i] * x[i];
    dst[c * n + i] = (double)v;
}

// dst[:,c] = src[:, sel_idx(c)] over `rows` f64 entries (src col stride
// ld_src, dst ld_dst): the flagged columns' first-pass T, compacted for the retry.
__global__ void gather_cols_f64_kernel(const double *src, size_t ld_src, size_t rows,
                                       const int *idx, size_t base,
                                       double *dst, size_t ld_dst)
{
    const size_t c = blockIdx.y;
    const size_t k = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= rows) {
        return;
    }
    dst[c * ld_dst + k] = src[sel_idx(idx, base, c) * ld_src + k];
}

// DGKS retry, scatter half: round each compact retry column c to f32, SCATTER it
// back to its global column sel_idx(c) in `out`, and recompute ds/ybx_raw there.
// One block per compact column (direct write, no atomics). Mirrors the
// f32-rounding semantics of round_reduce_kernel exactly; only flagged globals
// are touched.
__global__ void round_reduce_scatter_kernel(const double *cin, size_t n,
        const int *idx, size_t base, float *out, const float *y,
        double *ds, double *ybx_raw, int want_dot)
{
    const size_t c    = blockIdx.x;
    const size_t j    = sel_idx(idx, base, c);
    const double *col  = cin + c * n;
    float        *ocol = out + (size_t)j * n;
    double s_norm = 0.0, s_dot = 0.0;
    for (size_t i = threadIdx.x; i < n; i += blockDim.x) {
        const float  v  = (float)col[i];
        ocol[i] = v;
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

// dst[i*ld] = src[i]: scatter a contiguous column into a strided (row-major)
// matrix column.
__global__ void scatter_col_kernel(const float *src, float *dst, size_t ld, size_t n)
{
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        dst[i * ld] = src[i];
    }
}

__global__ void widen_kernel(const float *in, double *out, size_t total)
{
    const size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) {
        out[idx] = (double)in[idx];
    }
}

// dst[k,j] += (double)src[k,j] over a (rows x cols) block; src is compact
// (ld = rows), dst has column stride ld_dst. Accumulates each per-K-chunk f32 T
// into the f64 T for the blocked GEMM.
__global__ void accum_f32_to_f64_kernel(const float *src, double *dst,
                                        size_t rows, size_t ld_dst, size_t cols)
{
    const size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows * cols) {
        return;
    }
    const size_t k = idx % rows;
    const size_t j = idx / rows;
    dst[j * ld_dst + k] += (double)src[j * rows + k];
}

// Cast the f64 T (col stride ld_src) to a compact f32 dTc (ld = rows) for the
// f32 projection sgemm.
__global__ void cast_f64_to_f32_kernel(const double *src, size_t ld_src,
                                       float *dst, size_t rows, size_t cols)
{
    const size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows * cols) {
        return;
    }
    const size_t k = idx % rows;
    const size_t j = idx / rows;
    dst[j * rows + k] = (float)src[j * ld_src + k];
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

}  // namespace

// --- profiling helpers ------------------------------------------------------
// Record a stream marker; read the GPU ms between two markers (valid only after
// a stream sync, so callers accumulate after their end-of-call synchronize).
static inline void prof_mark(Context *ctx, int i)
{
    if (ctx->profile) {
        cudaEventRecord(ctx->ev[i], ctx->stream);
    }
}

static inline double prof_span(Context *ctx, int a, int b)
{
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, ctx->ev[a], ctx->ev[b]);
    return (double)ms;
}

// Fold one call's six markers (set in project_reduce + the fill site) into the
// running per-phase totals. Call only when profiling, after a full stream sync.
static void prof_accumulate(Context *ctx, size_t cols)
{
    ctx->ms_fill  += prof_span(ctx, 6, 7);
    ctx->ms_gemmT += prof_span(ctx, 0, 1);
    ctx->ms_proj  += prof_span(ctx, 1, 2);
    ctx->ms_round += prof_span(ctx, 2, 3);
    ctx->ms_dgks  += prof_span(ctx, 3, 4);
    ctx->ms_scale += prof_span(ctx, 4, 5);
    ctx->n_calls++;
    ctx->prof_cols += cols;
}

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

        // Candidate-column capacity for the batched path. Per output column the
        // scratch costs ~ (Bx_f32 4 + Bx_f64 8 + Xcand 4) = 16 bytes/row, so cap
        // the block at MARS_CUDA_BLOCK_GB (default 8) GB of n-row columns. Never
        // below max_terms (the per-eval path needs that many).
        double block_gb = 8.0;
        if (const char *bg = std::getenv("MARS_CUDA_BLOCK_GB")) {
            const double v = std::atof(bg);
            if (v > 0.0) {
                block_gb = v;
            }
        }
        const size_t budget = (size_t)(block_gb * 1e9);
        const size_t bytes_per_col = n * (sizeof(float) + sizeof(double) + sizeof(float));
        size_t p_cap = budget / (bytes_per_col ? bytes_per_col : 1);
        // The batched kernels put the candidate column on grid.y, whose limit
        // is 65535 (reached at the default budget for n below ~7600).
        p_cap = std::min<size_t>(p_cap, 65535);
        if (p_cap < max_terms) {
            p_cap = max_terms;
        }
        ctx->p_cap = p_cap;

        // Blocked-f32 T GEMM: MARS_CUDA_KCHUNK = K-chunk size (rows per cublasSgemm,
        // accumulated in f64 across chunks). ON by default (2048) -- ~2x at m=400
        // with fit quality preserved; set MARS_CUDA_KCHUNK=0 for the native-f64
        // reference path (bit-reproducible vs CPU).
        ctx->kchunk = 2048;
        if (const char *kc = std::getenv("MARS_CUDA_KCHUNK")) {
            const long v = std::atol(kc);
            if (v >= 0) {
                ctx->kchunk = (size_t)v;  // 0 disables (native f64)
            }
        }
        // Hinge sweep: MARS_CUDA_SWEEP_CHUNK pins the sorted-row chunk length
        // of the two-pass recurrence (0 / unset = sized from n and p, see
        // hinge_sweep.cu).
        if (const char *sc = std::getenv("MARS_CUDA_SWEEP_CHUNK")) {
            const long v = std::atol(sc);
            if (v > 0) {
                ctx->sweep_chunk = (size_t)v;
            }
        }
        // MARS_CUDA_SWEEP_BLOCK pins the sweep's block size (a multiple of 32,
        // <= 256); the tests use it to force several lanes per thread.
        if (const char *sb = std::getenv("MARS_CUDA_SWEEP_BLOCK")) {
            const long v = std::atol(sb);
            if (v >= 32 && v <= 256 && v % 32 == 0) {
                ctx->sweep_block = (int)v;
            }
        }
        // MARS_CUDA_SWEEP_TILE pins the rows staged per shared-memory tile
        // (a multiple of 4 in [4, 128]); the tile bounds the per-block shared
        // memory and hence the blocks resident per SM.
        if (const char *st = std::getenv("MARS_CUDA_SWEEP_TILE")) {
            const long v = std::atol(st);
            if (v >= 4 && v <= 128 && v % 4 == 0) {
                ctx->sweep_tile = (int)v;
            }
        }
        // MARS_CUDA_SWEEP_JB: candidate columns per sweep block (4 or 8).
        if (const char *sj = std::getenv("MARS_CUDA_SWEEP_JB")) {
            const long v = std::atol(sj);
            if (v == 4 || v == 8) {
                ctx->sweep_jb = (int)v;
            }
        }
        // MARS_CUDA_SWEEP_BATCH: grid slots per on-grid reduction batch (1..8).
        if (const char *sb2 = std::getenv("MARS_CUDA_SWEEP_BATCH")) {
            const long v = std::atol(sb2);
            if (v >= 1 && v <= 8) {
                ctx->sweep_batch = (int)v;
            }
        }
        if (std::getenv("MARS_CUDA_PROFILE")) {
            ctx->profile = 1;
            for (int i = 0; i < 8; ++i) {
                CUDA_CHECK(cudaEventCreate(&ctx->ev[i]));
            }
        }
        if (std::getenv("MARS_CUDA_VERBOSE")) {
            std::fprintf(stderr,
                         "[mars-cuda] context: n=%zu max_terms=%zu block_gb=%.3g "
                         "-> p_cap=%zu candidate cols/block; kchunk=%zu\n",
                         n, max_terms, block_gb, p_cap, ctx->kchunk);
        }

        const size_t nmt = n * max_terms;   // resident basis
        const size_t npc = n * p_cap;        // batched scratch
        CUDA_CHECK(cudaMalloc(&ctx->dB,        nmt * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&ctx->dBo_f32,   nmt * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&ctx->dBo_f64,   nmt * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&ctx->dBo_rm,    nmt * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&ctx->dy,        n * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&ctx->dBx_f32,   npc * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&ctx->dBx_f64,   npc * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&ctx->dT,        max_terms * p_cap * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&ctx->dTc_f32,   max_terms * p_cap * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&ctx->dx,        n * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&ctx->dmask,     p_cap * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&ctx->ds,        p_cap * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&ctx->dtnorm2,   p_cap * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&ctx->dybx_raw,  p_cap * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&ctx->dybx,      p_cap * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&ctx->dscale,    p_cap * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&ctx->didx,      p_cap * sizeof(int)));
        // DGKS retry: flagged columns are processed in chunks of retry_cols,
        // so the compact first-pass T needs only (max_terms, retry_cols).
        ctx->retry_cols = std::min(p_cap, std::max<size_t>(256, max_terms));
        CUDA_CHECK(cudaMalloc(&ctx->dT_retry, max_terms * ctx->retry_cols * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&ctx->dsrc_xcol, p_cap * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&ctx->dsrc_basis, p_cap * sizeof(int)));
        // dXscaled is allocated lazily in context_sync_xcols (needs p_x).

        ctx->h_stage.resize(n);
        ctx->h_ds.resize(p_cap);
        ctx->h_tnorm2.resize(p_cap);
        ctx->h_idx.resize(p_cap);
    }
    catch (...) {
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
    if (ctx->profile) {
        const double gpu = ctx->ms_fill + ctx->ms_gemmT + ctx->ms_proj +
                           ctx->ms_round + ctx->ms_dgks + ctx->ms_scale;
        std::fprintf(stderr,
                     "[mars-cuda] PROFILE: %ld batch calls, %zu cand-cols total\n"
                     "  fill  B*x  (n*P)    %10.1f ms\n"
                     "  T = Boᵀ*Bx GEMM     %10.1f ms\n"
                     "  projection GEMM     %10.1f ms\n"
                     "  round/norm/dot      %10.1f ms\n"
                     "  DGKS gate + retry   %10.1f ms  (%ld fired)\n"
                     "  scale-fold          %10.1f ms\n"
                     "  --- GPU total       %10.1f ms\n"
                     "  sync_basis (host)   %10.1f ms\n"
                     "  sync_xcols (host)   %10.1f ms\n",
                     ctx->n_calls, ctx->prof_cols,
                     ctx->ms_fill, ctx->ms_gemmT, ctx->ms_proj, ctx->ms_round,
                     ctx->ms_dgks, ctx->n_dgks_fired, ctx->ms_scale, gpu,
                     ctx->ms_sync_basis, ctx->ms_sync_xcols);
        if (ctx->n_sweeps) {
            std::fprintf(stderr,
                         "[mars-cuda] PROFILE hinge sweep: %ld calls (host wall time)\n"
                         "  sort + prep        %10.1f ms\n"
                         "  pass 1 (carries)   %10.1f ms\n"
                         "  pass 2 (scores)    %10.1f ms\n"
                         "  argmax + copies    %10.1f ms\n",
                         ctx->n_sweeps, ctx->ms_sweep_sort, ctx->ms_sweep_pass1,
                         ctx->ms_sweep_pass2, ctx->ms_sweep_misc);
        }
        if (ctx->n_lc_calls) {
            std::fprintf(stderr,
                         "[mars-cuda] PROFILE linear T cache: %ld calls (host wall time)\n"
                         "  row extension      %10.1f ms  (%ld slot-rows)\n"
                         "  fresh slots        %10.1f ms  (%ld slots)\n"
                         "  closed-form score  %10.1f ms\n"
                         "  f64 fallback       %10.1f ms  (%ld candidates, %ld dead)\n",
                         ctx->n_lc_calls, ctx->ms_lc_ext, ctx->n_lc_rows,
                         ctx->ms_lc_fresh, ctx->n_lc_fresh, ctx->ms_lc_score,
                         ctx->ms_lc_fb, ctx->n_lc_fb, ctx->n_lc_dead);
        }
        for (int i = 0; i < 8; ++i) {
            if (ctx->ev[i]) {
                cudaEventDestroy(ctx->ev[i]);
            }
        }
    }
    cudaFree(ctx->dB);
    cudaFree(ctx->dBo_f32);
    cudaFree(ctx->dBo_f64);
    cudaFree(ctx->dBo_rm);
    cudaFree(ctx->dy);
    sweep_free(ctx);
    linear_cache_free(ctx);
    cudaFree(ctx->dBx_f32);
    cudaFree(ctx->dBx_f64);
    cudaFree(ctx->dT);
    cudaFree(ctx->dTc_f32);
    cudaFree(ctx->dx);
    cudaFree(ctx->dmask);
    cudaFree(ctx->ds);
    cudaFree(ctx->dtnorm2);
    cudaFree(ctx->dybx_raw);
    cudaFree(ctx->dybx);
    cudaFree(ctx->dscale);
    cudaFree(ctx->didx);
    cudaFree(ctx->dT_retry);
    cudaFree(ctx->dyres);
    cudaFree(ctx->dXscaled);
    cudaFree(ctx->dsrc_xcol);
    cudaFree(ctx->dsrc_basis);
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
    const auto   t0    = std::chrono::steady_clock::now();
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

    // Bo: host is row-major (stride ldBo). Gather each new column straight into
    // the resident f32 dBo_f32, then widen in place to dBo_f64. One column at a
    // time with a stream sync so the reusable host stage buffer is safe to
    // overwrite. Rare (≈ one column per epoch), so the per-column cost amortizes.
    const dim3 grid_col((n + BLOCK - 1) / BLOCK);
    for (size_t c = start; c < m; ++c) {
        float *stage = ctx->h_stage.data();
        for (size_t i = 0; i < n; ++i) {
            stage[i] = Bo[i * ldBo + c];
        }
        CUDA_CHECK(cudaMemcpyAsync(ctx->dBo_f32 + c * n, stage, n * sizeof(float),
                                   cudaMemcpyHostToDevice, ctx->stream));
        widen_kernel<<<grid_col, BLOCK, 0, ctx->stream>>>(
            ctx->dBo_f32 + c * n, ctx->dBo_f64 + c * n, n);
        check_launch();
        // Row-major copy for the hinge sweep's per-row gather (stride
        // max_terms). A strided 4-byte scatter, once per appended column.
        scatter_col_kernel<<<grid_col, BLOCK, 0, ctx->stream>>>(
            ctx->dBo_f32 + c * n, ctx->dBo_rm + c, ctx->max_terms, n);
        check_launch();
        CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
    }

    ctx->m_synced = m;
    if (ctx->profile) {
        ctx->ms_sync_basis += std::chrono::duration<double, std::milli>(
                                  std::chrono::steady_clock::now() - t0).count();
    }
}

size_t context_batch_capacity(Context *ctx)
{
    return ctx->p_cap;
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

void context_set_residual(Context *ctx, const float *yres, size_t tag)
{
    if (ctx->yres_ptr == yres && ctx->yres_tag == tag) {
        return;  // unchanged since the last append
    }
    if (!ctx->dyres) {
        CUDA_CHECK(cudaMalloc(&ctx->dyres, ctx->n * sizeof(float)));
    }
    CUDA_CHECK(cudaMemcpyAsync(ctx->dyres, yres, ctx->n * sizeof(float),
                               cudaMemcpyHostToDevice, ctx->stream));
    CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
    ctx->yres_ptr = yres;
    ctx->yres_tag = tag;
}

void context_sync_xcols(Context *ctx, size_t p_x,
                        const float *X, size_t ldX, const float *s,
                        const int *cols, size_t ncols)
{
    const auto   t0 = std::chrono::steady_clock::now();
    const size_t n = ctx->n;
    if (!ctx->dXscaled) {
        CUDA_CHECK(cudaMalloc(&ctx->dXscaled, n * p_x * sizeof(float)));
        ctx->p_x = p_x;
        ctx->x_resident.assign(p_x, 0);
    }
    const dim3 grid_col((n + BLOCK - 1) / BLOCK);
    for (size_t t = 0; t < ncols; ++t) {
        const int c = cols[t];
        if (ctx->x_resident[c]) {
            continue;  // x_c = X[:,c]·s[c] is fixed for the fit -- upload once
        }
        CUDA_CHECK(cudaMemcpyAsync(ctx->dXscaled + (size_t)c * n,
                                   X + (size_t)c * ldX, n * sizeof(float),
                                   cudaMemcpyHostToDevice, ctx->stream));
        scale_col_kernel<<<grid_col, BLOCK, 0, ctx->stream>>>(
            ctx->dXscaled + (size_t)c * n, n, s[c]);
        check_launch();
        ctx->x_resident[c] = 1;
    }
    CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
    if (ctx->profile) {
        ctx->ms_sync_xcols += std::chrono::duration<double, std::milli>(
                                  std::chrono::steady_clock::now() - t0).count();
    }
}

void f64_reproject(Context *ctx, size_t m, size_t w_total,
                   const double *Tsrc, size_t ldTsrc, const int *tidx,
                   const FillSrc &src, const int *sidx, const int *oidx,
                   const float *ydot, int want_dot)
{
    const size_t n = ctx->n;
    cudaStream_t s = ctx->stream;
    const dim3 block(BLOCK);
    const double one = 1.0, zero = 0.0, neg_one = -1.0;
    const size_t R = ctx->retry_cols;
    const int ldR = (int)ctx->max_terms;
    for (size_t c0 = 0; c0 < w_total; c0 += R) {
        const int w = (int)std::min(R, w_total - c0);
        gather_cols_f64_kernel<<<dim3((unsigned)((m + BLOCK - 1) / BLOCK), (unsigned)w),
                                 block, 0, s>>>(Tsrc, ldTsrc, m, tidx, c0,
                                                ctx->dT_retry, ctx->max_terms);
        check_launch();
        refill_widen_kernel<<<dim3((unsigned)((n + BLOCK - 1) / BLOCK), (unsigned)w),
                              block, 0, s>>>(ctx->dB, n, src.X, src.xcol, src.basis,
                                             sidx, c0, ctx->dBx_f64);
        check_launch();
        CUBLAS_CHECK(cublasDgemm(ctx->handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                 (int)n, w, (int)m, &neg_one, ctx->dBo_f64, (int)n,
                                 ctx->dT_retry, ldR, &one, ctx->dBx_f64, (int)n));
        CUBLAS_CHECK(cublasDgemm(ctx->handle, CUBLAS_OP_T, CUBLAS_OP_N,
                                 (int)m, w, (int)n, &one, ctx->dBo_f64, (int)n,
                                 ctx->dBx_f64, (int)n, &zero, ctx->dT_retry, ldR));
        CUBLAS_CHECK(cublasDgemm(ctx->handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                 (int)n, w, (int)m, &neg_one, ctx->dBo_f64, (int)n,
                                 ctx->dT_retry, ldR, &one, ctx->dBx_f64, (int)n));
        round_reduce_scatter_kernel<<<(unsigned)w, block, 0, s>>>(
            ctx->dBx_f64, n, oidx, c0, ctx->dBx_f32, ydot,
            ctx->ds, ctx->dybx_raw, want_dot);
        check_launch();
    }
}

// Shared post-fill pipeline for P candidate columns already laid out in
// dBx_f32 (rounded) / dBx_f64 (widened): T = Boᵀ·Bx, project out Bo, fused
// round + norm + (optional) Σ Bx·t with t the residual target (else y), DGKS
// retry on cancelling columns, and scale-fold. On return ctx->ds /
// ctx->dscale / ctx->dybx (if want_dot) hold the per-column results; the caller
// owns any downloads and the final sync.
static void project_reduce(Context *ctx, size_t m, size_t P, int want_dot,
                           const FillSrc &src, std::atomic<long> *dgks_counter)
{
    const float *ydot = ctx->dyres ? ctx->dyres : ctx->dy;
    const size_t n   = ctx->n;
    const size_t ldT = ctx->ldT;
    const double tol = ctx->tol;
    cudaStream_t   s = ctx->stream;
    const double one = 1.0, zero = 0.0;
    const dim3 block(BLOCK);

    // Phase 1: T = Boᵀ · Bx  (m×P).
    prof_mark(ctx, 0);
    if (ctx->kchunk == 0) {
        // Native f64 reference (exact dot of the f32-stored inputs).
        CUBLAS_CHECK(cublasDgemm(ctx->handle, CUBLAS_OP_T, CUBLAS_OP_N,
                                 (int)m, (int)P, (int)n, &one, ctx->dBo_f64, (int)n,
                                 ctx->dBx_f64, (int)n, &zero, ctx->dT, (int)ldT));
    }
    else {
        // Blocked f32: split K=n into chunks, cublasSgemm each in f32 (fast on
        // weak-f64 GPUs), accumulate the chunk results into the f64 dT. The
        // cross-chunk f64 sum handles the global cancellation exactly; only the
        // within-chunk f32 error (~sqrt(kchunk)*eps_f32) remains.
        const float one_f = 1.0f, zero_f = 0.0f;
        CUDA_CHECK(cudaMemsetAsync(ctx->dT, 0, ldT * P * sizeof(double), s));
        const size_t nblk = (m * P + BLOCK - 1) / BLOCK;
        for (size_t k0 = 0; k0 < n; k0 += ctx->kchunk) {
            const int cc = (int)std::min(ctx->kchunk, n - k0);
            CUBLAS_CHECK(cublasSgemm(ctx->handle, CUBLAS_OP_T, CUBLAS_OP_N,
                                     (int)m, (int)P, cc,
                                     &one_f, ctx->dBo_f32 + k0, (int)n,
                                     ctx->dBx_f32 + k0, (int)n,
                                     &zero_f, ctx->dTc_f32, (int)m));
            accum_f32_to_f64_kernel<<<(unsigned)nblk, block, 0, s>>>(
                ctx->dTc_f32, ctx->dT, m, ldT, P);
            check_launch();
        }
    }

    // Phase 2a: project out Bo. proj = Bo·T as an f32 sgemm (K=m) into dBx_f32,
    // whose phase-0 contents (the f32 B·x) are no longer needed; the subtract
    // residual = (f64 B·x) - proj is fused into the round/reduce pass below, so
    // there is no separate n×P read-modify-write. That pass rounds the residual
    // to f32, stores it, and reduces ‖Bx[:,j]‖² (ds) and Σ Bx·y (ybx_raw), split
    // over rows AND columns (atomic f64 partials) so the whole GPU is used.
    prof_mark(ctx, 1);
    const float one_f = 1.0f, zero_f = 0.0f;
    const size_t nmp = m * P;
    cast_f64_to_f32_kernel<<<(unsigned)((nmp + BLOCK - 1) / BLOCK), block, 0, s>>>(
        ctx->dT, ldT, ctx->dTc_f32, m, P);
    check_launch();
    CUBLAS_CHECK(cublasSgemm(ctx->handle, CUBLAS_OP_N, CUBLAS_OP_N,
                             (int)n, (int)P, (int)m,
                             &one_f, ctx->dBo_f32, (int)n,
                             ctx->dTc_f32, (int)m,
                             &zero_f, ctx->dBx_f32, (int)n));

    unsigned tiles = (unsigned)std::min<size_t>(64, (n + BLOCK - 1) / BLOCK);
    if (tiles == 0) {
        tiles = 1;
    }
    const dim3 grid_reduce(tiles, (unsigned)P);
    CUDA_CHECK(cudaMemsetAsync(ctx->ds, 0, P * sizeof(double), s));
    if (want_dot) {
        CUDA_CHECK(cudaMemsetAsync(ctx->dybx_raw, 0, P * sizeof(double), s));
    }
    // `out` and `proj` are the same buffer (dBx_f32): each thread reads then
    // writes its own index, so the aliasing is benign.
    prof_mark(ctx, 2);
    round_reduce_kernel<<<grid_reduce, block, 0, s>>>(
        ctx->dBx_f64, ctx->dBx_f32, ydot, n, ctx->ds, ctx->dybx_raw, want_dot,
        ctx->dBx_f32);
    check_launch();
    prof_mark(ctx, 3);

    // Phase 2b: DGKS gate. s[j] > tol && s[j]*9 < ‖T[:,j]‖² -> retry. Build the
    // compact list of flagged columns on the host so the (rare) retry runs only
    // over those, not the whole P-wide block.
    colsumsq_f64_kernel<<<(unsigned)P, block, 0, s>>>(ctx->dT, ldT, m, ctx->dtnorm2);
    check_launch();
    CUDA_CHECK(cudaMemcpyAsync(ctx->h_ds.data(), ctx->ds, P * sizeof(double),
                               cudaMemcpyDeviceToHost, s));
    CUDA_CHECK(cudaMemcpyAsync(ctx->h_tnorm2.data(), ctx->dtnorm2,
                               P * sizeof(double), cudaMemcpyDeviceToHost, s));
    CUDA_CHECK(cudaStreamSynchronize(s));

    int fired = 0;
    for (size_t j = 0; j < P; ++j) {
        const bool flag = ctx->h_ds[j] > tol &&
                          ctx->h_ds[j] * mars::DGKS_GATE_RATIO_SQ < ctx->h_tnorm2[j];
        if (flag) {
            ctx->h_idx[fired++] = (int)j;
        }
    }

    if (fired > 0) {
        // Surgical retry, entirely in f64 and at width `fired` (the f64 GEMMs
        // are the weak path on this card, so their cost must scale with the
        // cancelling columns, not the batch). Each flagged column restarts from
        // its f64 B·x -- recomputed into dBx_f64, dead after round_reduce --
        // NOT from the rounded residual in dBx_f32: the f32 projection left
        // out-of-span noise ~eps_f32·√m·‖T‖ there that re-projecting cannot
        // remove. Two f64 passes then follow, r1 = Bx - Bo·T with the
        // first-pass (f64-accumulated) T, and r2 = r1 - Bo·(Boᵀ·r1); each
        // refined column is rounded and scattered back to its global slot with
        // fresh ds/ybx_raw. Chunked by retry_cols to bound the compact T.
        CUDA_CHECK(cudaMemcpyAsync(ctx->didx, ctx->h_idx.data(),
                                   (size_t)fired * sizeof(int),
                                   cudaMemcpyHostToDevice, s));
        f64_reproject(ctx, m, (size_t)fired, ctx->dT, ldT, ctx->didx,
                      src, ctx->didx, ctx->didx, ydot, want_dot);
        if (dgks_counter) {
            dgks_counter->fetch_add(fired, std::memory_order_relaxed);
        }
    }
    if (ctx->profile) {
        ctx->n_dgks_fired += fired;
    }
    prof_mark(ctx, 4);

    // Phase 2c: scale, folded into ybx (= scale·Σ Bx·y).
    scale_fold_kernel<<<(unsigned)((P + BLOCK - 1) / BLOCK), block, 0, s>>>(
        ctx->ds, ctx->dscale, ctx->dybx_raw, ctx->dybx, P, tol, want_dot);
    check_launch();
    prof_mark(ctx, 5);
}

void orthonormalize(Context *ctx, size_t m, size_t p,
                    const float *x, const int *mask,
                    float *Bx, size_t ldBx,
                    double *ybx,
                    std::atomic<long> *dgks_counter,
                    bool normalize_resident)
{
    if (p == 0) {
        return;
    }
    const size_t n = ctx->n;
    cudaStream_t s = ctx->stream;
    const dim3 block(BLOCK);
    const dim3 grid_np((n + BLOCK - 1) / BLOCK, (unsigned)p);
    const int want_dot = (ybx != nullptr) ? 1 : 0;

    // Upload the candidate column + active-basis mask, fill Bx = B[:,mask]·x.
    CUDA_CHECK(cudaMemcpyAsync(ctx->dx, x, n * sizeof(float),
                               cudaMemcpyHostToDevice, s));
    CUDA_CHECK(cudaMemcpyAsync(ctx->dmask, mask, p * sizeof(int),
                               cudaMemcpyHostToDevice, s));
    prof_mark(ctx, 6);
    fill_kernel<<<grid_np, block, 0, s>>>(ctx->dB, n, ctx->dx, ctx->dmask,
                                          ctx->dBx_f32, ctx->dBx_f64);
    check_launch();
    prof_mark(ctx, 7);

    project_reduce(ctx, m, p, want_dot, FillSrc{ctx->dx, nullptr, ctx->dmask},
                   dgks_counter);

    if (ybx) {
        CUDA_CHECK(cudaMemcpyAsync(ybx, ctx->dybx, p * sizeof(double),
                                   cudaMemcpyDeviceToHost, s));
    }
    // Only when the caller needs the matrix itself (the CPU hinge sweep, or the
    // GPU sweep reading it in place) do we physically normalize it, and only
    // the CPU sweep copies it back. In linear_only Bx is nullptr and stays
    // resident un-normalized (ybx carries the scale).
    if (Bx || normalize_resident) {
        normalize_kernel<<<grid_np, block, 0, s>>>(ctx->dBx_f32, ctx->dscale, n);
        check_launch();
    }
    if (Bx) {
        CUDA_CHECK(cudaMemcpy2DAsync(
                       Bx, ldBx * sizeof(float),
                       ctx->dBx_f32, n * sizeof(float),
                       n * sizeof(float), p,
                       cudaMemcpyDeviceToHost, s));
    }
    CUDA_CHECK(cudaStreamSynchronize(s));
    if (ctx->profile) {
        prof_accumulate(ctx, p);
    }
}

void orthonormalize_batch(Context *ctx, size_t m, size_t P,
                          const int *src_xcol, const int *src_basis,
                          double *ybx, std::atomic<long> *dgks_counter)
{
    if (P == 0) {
        return;
    }
    if (P > ctx->p_cap) {
        throw std::runtime_error(
            "orthonormalize_batch: P exceeds batch capacity (caller must block)");
    }
    const size_t n = ctx->n;
    cudaStream_t s = ctx->stream;
    const dim3 block(BLOCK);

    // Upload the per-output-column source maps and fill the stacked Bx (n×P) from
    // the RESIDENT scaled candidates (dXscaled, indexed by global X column) and
    // resident B -- no per-block X upload.
    CUDA_CHECK(cudaMemcpyAsync(ctx->dsrc_xcol, src_xcol, P * sizeof(int),
                               cudaMemcpyHostToDevice, s));
    CUDA_CHECK(cudaMemcpyAsync(ctx->dsrc_basis, src_basis, P * sizeof(int),
                               cudaMemcpyHostToDevice, s));
    const dim3 grid_nP((n + BLOCK - 1) / BLOCK, (unsigned)P);
    prof_mark(ctx, 6);
    fill_batch_kernel<<<grid_nP, block, 0, s>>>(
        ctx->dB, n, ctx->dXscaled, ctx->dsrc_xcol, ctx->dsrc_basis,
        ctx->dBx_f32, ctx->dBx_f64);
    check_launch();
    prof_mark(ctx, 7);

    project_reduce(ctx, m, P, /*want_dot=*/1,
                   FillSrc{ctx->dXscaled, ctx->dsrc_xcol, ctx->dsrc_basis},
                   dgks_counter);

    CUDA_CHECK(cudaMemcpyAsync(ybx, ctx->dybx, P * sizeof(double),
                               cudaMemcpyDeviceToHost, s));
    CUDA_CHECK(cudaStreamSynchronize(s));
    if (ctx->profile) {
        prof_accumulate(ctx, P);
    }
}

}  // namespace cuda
}  // namespace mars
