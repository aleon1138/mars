#pragma once

/*
 *  Internal, CUDA-side definitions shared by the .cu translation units of the
 *  mars_cuda static library: the error-check macros, the warp/block reduction
 *  helpers and the full definition of the opaque mars::cuda::Context declared
 *  in mars_cuda.h. Never included by host-compiled code.
 */

#include "mars_cuda.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <stdexcept>
#include <string>
#include <vector>

namespace mars {
namespace cuda {

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

inline void check_launch()
{
    CUDA_CHECK(cudaGetLastError());
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

// --- Context ----------------------------------------------------------------

struct Context {
    cublasHandle_t handle = nullptr;
    cudaStream_t   stream = nullptr;

    size_t n         = 0;
    size_t max_terms = 0;
    size_t ldT       = 0;   // == max_terms
    size_t p_cap     = 0;   // candidate-column capacity of the scratch (>= max_terms)
    double tol       = 0.0;
    size_t m_synced  = 0;   // columns of B/Bo currently resident on device

    // resident basis + target
    float  *dB      = nullptr;  // (n, max_terms) col-major, ld n
    float  *dBo_f32 = nullptr;  // (n, max_terms) col-major, ld n -- f32 Bo (blocked GEMM + widen src)
    double *dBo_f64 = nullptr;  // (n, max_terms) col-major, ld n -- widened from dBo_f32
    float  *dBo_rm  = nullptr;  // (n, max_terms) ROW-major, ld max_terms -- the hinge sweep's row gather
    float  *dy      = nullptr;  // (n) normalized target
    const float *target_ptr = nullptr;  // host y pointer last uploaded to dy
    float  *dyres   = nullptr;  // (n) residual target y - Bo·ybo (ybx dot target once set)
    const float *yres_ptr = nullptr;    // host y_res pointer last uploaded to dyres
    size_t  yres_tag = (size_t)-1;      // caller's basis count at that upload

    // blocked-f32 T = Boᵀ·Bx: split K=n into chunks, cublasSgemm each in f32,
    // accumulate the chunk results into the f64 dT. Default 2048 (on);
    // MARS_CUDA_KCHUNK=0 selects the native-f64 T GEMM.
    size_t  kchunk   = 2048;
    float  *dTc_f32  = nullptr;  // (max_terms, p_cap) per-chunk f32 T, then the f32 cast of dT

    // per-call scratch (sized for the worst case p == p_cap; the per-eval path
    // uses the leading <= max_terms columns, the batched path up to p_cap)
    float  *dBx_f32 = nullptr;  // (n, p_cap) col-major, ld n
    double *dBx_f64 = nullptr;  // (n, p_cap) col-major, ld n
    double *dT      = nullptr;  // (max_terms, p_cap) col-major, ld max_terms
    float  *dx      = nullptr;  // (n) single candidate (per-eval path)
    int    *dmask   = nullptr;  // (p_cap)
    double *ds       = nullptr;  // (p_cap)
    double *dtnorm2  = nullptr;  // (p_cap)
    double *dybx_raw = nullptr;  // (p_cap) Σ Bx·y on the un-normalized column
    double *dybx     = nullptr;  // (p_cap) scale·ybx_raw -- the ΔSSE input
    float  *dscale   = nullptr;  // (p_cap)
    int    *didx    = nullptr;  // (p_cap) compact global indices of DGKS-flagged cols
    double *dT_retry = nullptr; // (max_terms, retry_cols) compact T of the flagged cols
    size_t  retry_cols = 0;     // flagged columns per retry chunk

    // batched path: resident scaled candidates (x_c = X[:,c]·s[c]) and the
    // per-output-column source maps. dXscaled is allocated (n, p_x) on the first
    // context_sync_xcols call; columns are uploaded+scaled lazily, once each.
    float  *dXscaled  = nullptr;  // (n, p_x) col-major -- resident scaled candidates
    size_t  p_x       = 0;        // X-column count (dXscaled width)
    std::vector<char> x_resident; // (p_x) which columns are uploaded+scaled
    int    *dsrc_xcol = nullptr;  // (p_cap) output col -> global X column
    int    *dsrc_basis = nullptr; // (p_cap) output col -> B/Bo column

    // hinge sweep (cuda/hinge_sweep.cu): the sort of the raw X column, the
    // sorted scaled candidate and its deltas, the per-chunk carries of the
    // two-pass recurrence and the per-(chunk, column) best cuts. Allocated on
    // the first sweep (sized by n / max_terms); the chunk-sized buffers grow
    // on demand.
    float  *dxraw      = nullptr;  // (n) raw X column (the cut values)
    float  *dkeys      = nullptr;  // (n) sort keys: raw X with -0.0 folded to +0.0
    float  *dkeys_out  = nullptr;  // (n) sorted keys (unused output of the sort)
    int    *dseq       = nullptr;  // (n) 0..n-1 (sort values in)
    int    *dk         = nullptr;  // (n) sort permutation (descending, stable)
    int    *dpos       = nullptr;  // (n) inverse permutation: sorted position of each row
    float  *dxs        = nullptr;  // (n) x[k[t]] -- the scaled candidate in sorted order
    float  *dd         = nullptr;  // (n) d[t] = xs[t-1] - xs[t] (f32), d[0] = 0
    double *dybo       = nullptr;  // (max_terms) Boᵀ·y
    int    *di_last    = nullptr;  // (p_cap) last sorted position with a nonzero parent entry
    void   *dsort_tmp  = nullptr;  // cub radix-sort workspace
    size_t  sort_tmp_bytes = 0;
    double *dcarry_f   = nullptr;  // (nchunks, p, lanes) carry-in f per lane
    double *dcarry_g   = nullptr;  // (nchunks, p, lanes) carry-in g per lane
    double *dcarry_h2  = nullptr;  // (nchunks, p) carry-in ||h+||^2
    double *ddsum      = nullptr;  // (nchunks) Σ d over each chunk (f64)
    double *dbest_sse  = nullptr;  // (nchunks, p) per-chunk best score
    int    *dbest_idx  = nullptr;  // (nchunks, p) per-chunk best sorted position (-1: none)
    double *dh_sse     = nullptr;  // (p_cap) final best score per column
    int    *dh_idx     = nullptr;  // (p_cap) final best sorted position per column
    double *dh_cut     = nullptr;  // (p_cap) final cut value per column
    size_t  carry_cap  = 0;        // doubles in each of dcarry_f / dcarry_g
    size_t  chunkp_cap = 0;        // entries in dcarry_h2 / dbest_sse / dbest_idx
    size_t  chunk_cap  = 0;        // entries in ddsum
    size_t  sweep_chunk = 0;       // MARS_CUDA_SWEEP_CHUNK: fixed chunk length (0 = auto)
    int     sweep_block = 0;       // MARS_CUDA_SWEEP_BLOCK: fixed block size (0 = auto; tests)
    int     sweep_tile  = 0;       // MARS_CUDA_SWEEP_TILE: rows staged per tile (0 = auto)
    int     sweep_jb    = 4;       // MARS_CUDA_SWEEP_JB: columns per block, 4 or 8
    int     sweep_batch = 0;       // MARS_CUDA_SWEEP_BATCH: grid slots per reduction batch (0 = 1)

    // host staging
    std::vector<float>  h_stage;    // (n) gather one Bo column
    std::vector<double> h_ds;       // (p_cap)
    std::vector<double> h_tnorm2;   // (p_cap)
    std::vector<int>    h_idx;      // (p_cap) compact flagged-column indices

    // --- profiling (MARS_CUDA_PROFILE): per-phase GPU time, accumulated over
    // the whole fit and dumped at context_destroy. Zero overhead when off.
    int         profile = 0;
    cudaEvent_t ev[8] = {};         // per-call phase markers (created when on)
    double ms_fill = 0, ms_gemmT = 0, ms_proj = 0, ms_round = 0,
           ms_dgks = 0, ms_scale = 0;       // GPU phases (cudaEvent deltas)
    double ms_sync_basis = 0, ms_sync_xcols = 0;  // host-side (std::chrono)
    long   n_calls = 0, n_dgks_fired = 0;
    size_t prof_cols = 0;           // total candidate columns processed
    // hinge sweep phases (host-side wall time around stream syncs)
    double ms_sweep_sort = 0, ms_sweep_pass1 = 0, ms_sweep_pass2 = 0, ms_sweep_misc = 0;
    long   n_sweeps = 0;

    // linear_only T cache (cuda/linear_cache.cu). One slot per (X column c,
    // parent k), slot = c*max_terms + k, holding that candidate's T column
    // (ldT rows) and its closed-form scalars. Allocated by linear_cache_init.
    int     lc_on    = 0;        // 0: not tried, 1: allocated, -1: unavailable
    size_t  lc_slots = 0;        // p_x * max_terms
    size_t  lc_rows  = 0;        // Bo rows every live slot's T is extended to
    size_t  lc_mE    = 0;        // E columns resident in dLE
    size_t  lc_terms = 0;        // basis rows whose dead-on-append mark is applied
    int     lc_sms   = 1;        // SM count (sizes the row-extension grid)
    double *dLT    = nullptr;    // (ldT, lc_slots) col-major
    double *dLnb   = nullptr;    // (lc_slots) ‖Bx‖²
    double *dLby   = nullptr;    // (lc_slots) <Bx, y>
    double *dLtt   = nullptr;    // (lc_slots) TᵀT
    double *dLtet  = nullptr;    // (lc_slots) TᵀET
    double *dLty   = nullptr;    // (lc_slots) Tᵀybo
    double *dLE    = nullptr;    // (max_terms, max_terms) col-major E = BoᵀBo - I
    double *dLybo  = nullptr;    // (max_terms)
    double *dLEybo = nullptr;    // (max_terms) E·ybo
    int    *dLlive = nullptr;    // (lc_slots) live slot ids, ascending
    int    *dLreq  = nullptr;    // (lc_slots) slot of each scored candidate
    double *dLout  = nullptr;    // (lc_slots) closed-form ΔSSE per scored candidate
    int    *dLflag = nullptr;    // (lc_slots) 1: near-degenerate, take the f64 fallback
    int    *dLidx  = nullptr;    // (p_cap) slots of a fresh / fallback piece
    std::vector<char>   h_lstate;  // (lc_slots) 0 absent, 1 live, 2 dead (in span)
    std::vector<int>    h_llive;   // live slots, ascending
    bool                lc_live_dirty = false;  // h_llive changed since its upload
    std::vector<int>    h_lreq, h_lflag, h_lpos;
    std::vector<double> h_lout;
    double ms_lc_ext = 0, ms_lc_fresh = 0, ms_lc_score = 0, ms_lc_fb = 0;
    long   n_lc_calls = 0, n_lc_fresh = 0, n_lc_fb = 0, n_lc_dead = 0, n_lc_rows = 0;
    long   n_lc_dead_append = 0;   // slots marked dead on append (the lin_x / lin_b record)
};

// Where output column j's B·x came from, so an f64 re-projection can recompute
// it: x_j = X + xcol[j]*n (xcol non-null, batched path) or X (per-eval path),
// times B[:, basis[j]]. All device pointers.
struct FillSrc {
    const float *X;
    const int   *xcol;
    const int   *basis;
};

/*
 *  Explicit f64 projection of w_total columns (the DGKS retry, and the T
 *  cache's near-degenerate fallback). For compact column c, with each index
 *  array mapping c -> idx[c] (or c itself when null):
 *      T0  = Tsrc[:, tidx[c]]          (the first-pass, f64-accurate T)
 *      r1  = (f64) B·x of source sidx[c] (src) - Bo·T0
 *      r2  = r1 - Bo·(Boᵀ·r1)
 *  then r2 is rounded to f32 into dBx_f32 column oidx[c], with ds = ‖r2_f32‖²
 *  and ybx_raw = <r2_f32, ydot> written at that index. Runs in chunks of
 *  retry_cols through dT_retry / dBx_f64. Asynchronous on ctx->stream.
 */
void f64_reproject(Context *ctx, size_t m, size_t w_total,
                   const double *Tsrc, size_t ldTsrc, const int *tidx,
                   const FillSrc &src, const int *sidx, const int *oidx,
                   const float *ydot, int want_dot);

// Free the T cache buffers (context_destroy, and linear_cache_init on failure).
void linear_cache_free(Context *ctx);

// Free the hinge-sweep buffers (context_destroy, and hinge_sweep.cu on regrow).
void sweep_free(Context *ctx);

} // namespace cuda
} // namespace mars
