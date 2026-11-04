/*
 *  The linear_only T cache on the GPU: the device half of
 *  MarsAlgo::eval_linear_cached (marsalgo.cc), behind eval_batch().
 *
 *  Bo, B and the scaled X columns never change during a fit, so the entries
 *  T[r, (c,k)] = <bo_r, fl(b_k * x_c)> of a candidate's T column never change
 *  once computed. Each slot (X column c, parent k) keeps its T column plus
 *  the scalars of the closed form (mars::linear_dsse):
 *      den = ‖Bx‖² − TᵀT + TᵀET,   u = <Bx,y> − Tᵀybo + TᵀE·ybo,
 *      ΔSSE = u² / (den + tol)
 *  with E = BoᵀBo − I. Per call this file:
 *    1. extends every live slot by the Bo rows appended since the last call
 *       (an f64 kernel forming the same f32 product as the fill), and updates
 *       its TᵀT / TᵀET / Tᵀybo row by row;
 *    2. computes a full T column for each newly requested slot (fill +
 *       cublasDgemm in f64);
 *    3. scores every requested candidate in closed form (one warp each);
 *    4. sends the near-degenerate ones (κ² > LINEAR_KAPPA2_MAX) through the
 *       explicit f64 projection (f64_reproject, shared with the DGKS retry)
 *       and marks those found in span(Bo) dead: the span only grows, so they
 *       score 0 for good and are no longer extended.
 *
 *  T must be f64-accurate: den takes a T error at first order, which is why
 *  the blocked-f32 T of orthonormalize_batch() cannot feed the closed form.
 *  Both GEMM-shaped pieces here are thin (one new row for every live slot, and
 *  m rows for the new slots), so they are affordable in f64 on this card.
 */

#include "../kernels.h"   // mars::LINEAR_KAPPA2_MAX
#include "context.cuh"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>

namespace mars {
namespace cuda {

namespace {

constexpr int LC_RMAX = 4;   // new Bo rows per extension launch

// T[slot, r0 + r] += Σ_{i in chunk} bo_{r0+r}[i] · (double)fl(b_k[i] · x_c[i]),
// r < R, for every live slot. One thread per slot (consecutive slots share c,
// so x is a warp broadcast, as is bo), blockIdx.y = row chunk; the chunk
// partials meet in f64 atomics on rows the fresh store zeroed. The f32 product
// is formed exactly as the fill forms it, so T stays the T of the column that
// nb / by describe (an unrounded product would perturb den at first order).
template <int R>
__global__ void lc_cross_rows_kernel(const float *B, const float *X, const double *Bo64,
                                     size_t n, const int *live, size_t L, size_t mt,
                                     size_t r0, size_t chunk, double *T, size_t ldT)
{
    const size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= L) {
        return;
    }
    const size_t s  = (size_t)live[t];
    const size_t c  = s / mt;
    const size_t k  = s % mt;
    const float  *b  = B + k * n;
    const float  *x  = X + c * n;
    const double *bo = Bo64 + r0 * n;
    const size_t i0 = (size_t)blockIdx.y * chunk;
    const size_t i1 = min(n, i0 + chunk);
    double acc[R];
#pragma unroll
    for (int r = 0; r < R; ++r) {
        acc[r] = 0.0;
    }
#pragma unroll 4
    for (size_t i = i0; i < i1; ++i) {
        const double v = (double)__fmul_rn(b[i], x[i]);
#pragma unroll
        for (int r = 0; r < R; ++r) {
            acc[r] = fma(v, bo[r * n + i], acc[r]);
        }
    }
    double *ts = T + s * ldT + r0;
#pragma unroll
    for (int r = 0; r < R; ++r) {
        atomicAdd(ts + r, acc[r]);
    }
}

// The closed-form scalars of slots[w] from T rows [r0, m), one warp per slot:
//     tt += Σ t_r²,   ty += Σ t_r·ybo_r,
//     tet += Σ_r (2 t_r Σ_{l<r} E[l,r] t_l + E[r,r] t_r²)
// which is TᵀET grown one row at a time (E symmetric; only its upper triangle,
// column r rows l <= r, is read, so only E's new columns need uploading). With
// r0 == 0 the three are set rather than incremented (a fresh slot).
__global__ void lc_scalars_kernel(const int *slots, size_t cnt, const double *T, size_t ldT,
                                  const double *E, size_t ldE, const double *ybo,
                                  size_t r0, size_t m, double *tt, double *tet, double *ty)
{
    const size_t w    = ((size_t)blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    const int    lane = threadIdx.x % warpSize;
    if (w >= cnt) {
        return;                     // whole warps exit together
    }
    const size_t  s = (size_t)slots[w];
    const double *t = T + s * ldT;
    double a_tt = 0.0, a_tet = 0.0, a_ty = 0.0;
    for (size_t r = r0; r < m; ++r) {
        const double *Er = E + r * ldE;
        double el = 0.0;
        for (size_t l = lane; l < r; l += warpSize) {
            el = fma(Er[l], t[l], el);
        }
        el = warpReduceSum(el);
        if (lane == 0) {
            const double tr = t[r];
            a_tet += 2.0 * tr * el + Er[r] * tr * tr;
            a_tt  += tr * tr;
            a_ty  += tr * ybo[r];
        }
    }
    if (lane == 0) {
        if (r0 == 0) {
            tt[s] = a_tt;
            tet[s] = a_tet;
            ty[s] = a_ty;
        }
        else {
            tt[s] += a_tt;
            tet[s] += a_tet;
            ty[s] += a_ty;
        }
    }
}

// Fresh slots: bx64[:,j] = (double)fl(B[:, basis[j]] · X[:, xcol[j]]) for the
// Dgemm, and nb[j] = Σ v², by[j] = Σ v·y (f64; atomics over the row tiles, so
// the caller zeroes nb / by first). blockIdx.y = j.
__global__ void lc_fill_stats_kernel(const float *B, const float *X, const float *y,
                                     size_t n, const int *xcol, const int *basis,
                                     double *bx64, double *nb, double *by)
{
    const size_t j = blockIdx.y;
    const float *b = B + (size_t)basis[j] * n;
    const float *x = X + (size_t)xcol[j] * n;
    double      *o = bx64 + j * n;
    double s_nb = 0.0, s_by = 0.0;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n;
            i += (size_t)gridDim.x * blockDim.x) {
        const double v = (double)__fmul_rn(b[i], x[i]);
        o[i] = v;
        s_nb = fma(v, v, s_nb);
        s_by = fma(v, (double)y[i], s_by);
    }
    blockReduceSum2(s_nb, s_by);
    if (threadIdx.x == 0) {
        atomicAdd(&nb[j], s_nb);
        atomicAdd(&by[j], s_by);
    }
}

// Store fresh column j into its slot: T rows [0, m) from the Dgemm output, rows
// [m, ldT) zeroed (the row extension accumulates into them), nb / by copied.
__global__ void lc_store_kernel(const int *slots, const double *Tsrc, size_t ldTs,
                                const double *nb_src, const double *by_src,
                                size_t m, double *T, size_t ldT, double *nb, double *by)
{
    const size_t j = blockIdx.y;
    const size_t r = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t s = (size_t)slots[j];
    if (r < ldT) {
        T[s * ldT + r] = r < m ? Tsrc[j * ldTs + r] : 0.0;
    }
    if (r == 0) {
        nb[s] = nb_src[j];
        by[s] = by_src[j];
    }
}

// Closed-form score of candidate q (slot req[q]), one warp each. Writes
// out[q] = ΔSSE and flag[q] = 1 when κ² = nb/den exceeds kappa2_max (the
// caller re-scores those through the explicit path).
__global__ void lc_score_kernel(const int *req, size_t P, const double *T, size_t ldT,
                                const double *Eybo, size_t m,
                                const double *nb, const double *by, const double *tt,
                                const double *tet, const double *ty,
                                double tol, double kappa2_max, double *out, int *flag)
{
    const size_t q    = ((size_t)blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    const int    lane = threadIdx.x % warpSize;
    if (q >= P) {
        return;
    }
    const size_t  s = (size_t)req[q];
    const double *t = T + s * ldT;
    double tey = 0.0;
    for (size_t r = lane; r < m; r += warpSize) {
        tey = fma(t[r], Eybo[r], tey);
    }
    tey = warpReduceSum(tey);
    if (lane == 0) {
        const double den = nb[s] - tt[s] + tet[s];
        if (den * kappa2_max < nb[s]) {
            out[q]  = 0.0;
            flag[q] = 1;
        }
        else {
            const double u = by[s] - ty[s] + tey;
            out[q]  = den > tol ? u * u / (den + tol) : 0.0;
            flag[q] = 0;
        }
    }
}

using clock_t_ = std::chrono::steady_clock;

// Profiling: when on, close the current phase (stream sync) and add its wall
// time to `acc`.
inline void lc_phase(Context *ctx, clock_t_::time_point &t0, double &acc)
{
    if (!ctx->profile) {
        return;
    }
    CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
    const auto t1 = clock_t_::now();
    acc += std::chrono::duration<double, std::milli>(t1 - t0).count();
    t0 = t1;
}

// Mark slot (xcol, basis) dead: scored 0 from now on, never filled or extended.
void mark_dead(Context *ctx, size_t sl)
{
    if (sl >= ctx->lc_slots || ctx->h_lstate[sl] == 2) {
        return;
    }
    if (ctx->h_lstate[sl] == 1) {
        auto &lv = ctx->h_llive;
        lv.erase(std::lower_bound(lv.begin(), lv.end(), (int)sl));
        ctx->lc_live_dirty = true;
    }
    ctx->h_lstate[sl] = 2;
}

}  // namespace

void linear_cache_free(Context *ctx)
{
    double **dbl[] = {&ctx->dLT, &ctx->dLnb, &ctx->dLby, &ctx->dLtt, &ctx->dLtet,
                      &ctx->dLty, &ctx->dLE, &ctx->dLybo, &ctx->dLEybo, &ctx->dLout
                     };
    for (double **p : dbl) {
        cudaFree(*p);
        *p = nullptr;
    }
    int **ints[] = {&ctx->dLlive, &ctx->dLreq, &ctx->dLflag, &ctx->dLidx};
    for (int **p : ints) {
        cudaFree(*p);
        *p = nullptr;
    }
}

bool linear_cache_init(Context *ctx, size_t p_x)
{
    if (ctx->lc_on != 0) {
        return ctx->lc_on > 0;
    }
    const size_t mt    = ctx->max_terms;
    const size_t slots = p_x * mt;
    try {
        CUDA_CHECK(cudaMalloc(&ctx->dLT,    ctx->ldT * slots * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&ctx->dLnb,   slots * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&ctx->dLby,   slots * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&ctx->dLtt,   slots * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&ctx->dLtet,  slots * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&ctx->dLty,   slots * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&ctx->dLE,    mt * mt * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&ctx->dLybo,  mt * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&ctx->dLEybo, mt * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&ctx->dLout,  slots * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&ctx->dLlive, slots * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&ctx->dLreq,  slots * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&ctx->dLflag, slots * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&ctx->dLidx,  ctx->p_cap * sizeof(int)));
        int dev = 0, sms = 1;
        CUDA_CHECK(cudaGetDevice(&dev));
        CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));
        ctx->lc_sms = std::max(sms, 1);
    }
    catch (const std::exception &e) {
        // Out of device memory (or similar): run without the cache.
        linear_cache_free(ctx);
        cudaGetLastError();
        ctx->lc_on = -1;
        if (std::getenv("MARS_CUDA_VERBOSE")) {
            std::fprintf(stderr, "[mars-cuda] linear T cache unavailable (%s)\n", e.what());
        }
        return false;
    }
    ctx->lc_slots = slots;
    ctx->lc_rows  = 0;
    ctx->lc_mE    = 0;
    ctx->lc_terms = 0;
    ctx->h_lstate.assign(slots, 0);
    ctx->h_llive.clear();
    ctx->lc_live_dirty = false;
    ctx->lc_on = 1;
    if (std::getenv("MARS_CUDA_VERBOSE")) {
        std::fprintf(stderr, "[mars-cuda] linear T cache: %zu slots, %.3g GB of T\n",
                     slots, (double)(ctx->ldT * slots * sizeof(double)) / 1e9);
    }
    return true;
}

void linear_cache_eval(Context *ctx, size_t m, size_t P,
                       const int *src_xcol, const int *src_basis,
                       const double *E, size_t ldE, const double *ybo, const double *Eybo,
                       const int *lin_x, const int *lin_b,
                       double *dsse, std::atomic<long> *fallback_counter)
{
    if (ctx->lc_on <= 0) {
        throw std::runtime_error("linear_cache_eval: the T cache is not initialized");
    }
    if (m > ctx->m_synced || !ctx->dXscaled) {
        throw std::runtime_error("linear_cache_eval: basis or X columns not resident");
    }
    const size_t n   = ctx->n;
    const size_t mt  = ctx->max_terms;
    const size_t ldT = ctx->ldT;
    const double tol = ctx->tol;
    cudaStream_t s   = ctx->stream;
    const dim3 block(BLOCK);
    const float *ydot = ctx->dyres ? ctx->dyres : ctx->dy;
    auto t0 = clock_t_::now();
    ctx->n_lc_calls++;

    // E's new columns (rows 0..m; only the upper triangle is read), and the
    // current ybo / E·ybo (E·ybo changes in every entry per append).
    if (ctx->lc_mE < m) {
        CUDA_CHECK(cudaMemcpy2DAsync(ctx->dLE + ctx->lc_mE * mt, mt * sizeof(double),
                                     E + ctx->lc_mE * ldE, ldE * sizeof(double),
                                     m * sizeof(double), m - ctx->lc_mE,
                                     cudaMemcpyHostToDevice, s));
        ctx->lc_mE = m;
    }
    CUDA_CHECK(cudaMemcpyAsync(ctx->dLybo, ybo, m * sizeof(double),
                               cudaMemcpyHostToDevice, s));
    CUDA_CHECK(cudaMemcpyAsync(ctx->dLEybo, Eybo, m * sizeof(double),
                               cudaMemcpyHostToDevice, s));

    // 0. Dead-on-append: basis column r appended as the linear term
    //    (lin_x[r], lin_b[r]) is that candidate itself, in span(Bo) by
    //    construction. Mark its slot dead before the extension below.
    for (size_t r = ctx->lc_terms; r < m; ++r) {
        if (lin_x[r] >= 0) {
            mark_dead(ctx, (size_t)lin_x[r] * mt + (size_t)lin_b[r]);
            ctx->n_lc_dead_append++;
        }
    }
    ctx->lc_terms = m;

    // 1. Extend the live slots by the rows [lc_rows, m).
    const size_t L = ctx->h_llive.size();
    if (ctx->lc_rows < m && L > 0) {
        if (ctx->lc_live_dirty) {
            CUDA_CHECK(cudaMemcpyAsync(ctx->dLlive, ctx->h_llive.data(), L * sizeof(int),
                                       cudaMemcpyHostToDevice, s));
            ctx->lc_live_dirty = false;
        }
        // Row chunks: enough blocks for ~8 per SM, at most 32k rows each (so
        // concurrent blocks share an L2-resident stretch of B / X / Bo).
        const size_t gx  = (L + BLOCK - 1) / BLOCK;
        size_t nch = std::max<size_t>(1, (8 * (size_t)ctx->lc_sms + gx - 1) / gx);
        nch = std::max(nch, (n + 32767) / 32768);
        nch = std::min<size_t>({nch, 65535, (n + 255) / 256});
        const size_t chunk = (n + nch - 1) / nch;
        nch = (n + chunk - 1) / chunk;
        const dim3 grid((unsigned)gx, (unsigned)nch);
        for (size_t r0 = ctx->lc_rows; r0 < m; r0 += LC_RMAX) {
            const int R = (int)std::min<size_t>(LC_RMAX, m - r0);
            const int *lv = ctx->dLlive;
            switch (R) {
            case 1:
                lc_cross_rows_kernel<1><<<grid, block, 0, s>>>(ctx->dB, ctx->dXscaled,
                        ctx->dBo_f64, n, lv, L, mt, r0, chunk, ctx->dLT, ldT);
                break;
            case 2:
                lc_cross_rows_kernel<2><<<grid, block, 0, s>>>(ctx->dB, ctx->dXscaled,
                        ctx->dBo_f64, n, lv, L, mt, r0, chunk, ctx->dLT, ldT);
                break;
            case 3:
                lc_cross_rows_kernel<3><<<grid, block, 0, s>>>(ctx->dB, ctx->dXscaled,
                        ctx->dBo_f64, n, lv, L, mt, r0, chunk, ctx->dLT, ldT);
                break;
            default:
                lc_cross_rows_kernel<4><<<grid, block, 0, s>>>(ctx->dB, ctx->dXscaled,
                        ctx->dBo_f64, n, lv, L, mt, r0, chunk, ctx->dLT, ldT);
                break;
            }
            check_launch();
        }
        const size_t warps_per_block = BLOCK / 32;
        lc_scalars_kernel<<<(unsigned)((L + warps_per_block - 1) / warps_per_block), block, 0, s>>>(
            ctx->dLlive, L, ctx->dLT, ldT, ctx->dLE, mt, ctx->dLybo, ctx->lc_rows, m,
            ctx->dLtt, ctx->dLtet, ctx->dLty);
        check_launch();
        ctx->n_lc_rows += (long)(L * (m - ctx->lc_rows));
    }
    ctx->lc_rows = m;
    lc_phase(ctx, t0, ctx->ms_lc_ext);

    // 2. Classify the candidates: dead ones score 0, absent ones are fresh.
    std::vector<int> &fresh = ctx->h_lpos;      // candidate indices of fresh slots
    fresh.clear();
    for (size_t j = 0; j < P; ++j) {
        const size_t sl = (size_t)src_xcol[j] * mt + (size_t)src_basis[j];
        if (ctx->h_lstate[sl] == 0) {
            fresh.push_back((int)j);
        }
    }

    // 3. Fresh slots, in pieces bounded by the n-row scratch: fill + stats,
    //    T = Boᵀ·Bx (Dgemm, f64), store into the slots, then their scalars.
    if (!fresh.empty()) {
        const double one = 1.0, zero = 0.0;
        std::vector<int> fx, fk, fs;
        const size_t piece = std::min<size_t>(ctx->p_cap, 65535);
        for (size_t f0 = 0; f0 < fresh.size(); f0 += piece) {
            const size_t F = std::min(piece, fresh.size() - f0);
            fx.resize(F);
            fk.resize(F);
            fs.resize(F);
            for (size_t c = 0; c < F; ++c) {
                const int j = fresh[f0 + c];
                fx[c] = src_xcol[j];
                fk[c] = src_basis[j];
                fs[c] = (int)((size_t)fx[c] * mt + (size_t)fk[c]);
            }
            CUDA_CHECK(cudaMemcpyAsync(ctx->dsrc_xcol, fx.data(), F * sizeof(int),
                                       cudaMemcpyHostToDevice, s));
            CUDA_CHECK(cudaMemcpyAsync(ctx->dsrc_basis, fk.data(), F * sizeof(int),
                                       cudaMemcpyHostToDevice, s));
            CUDA_CHECK(cudaMemcpyAsync(ctx->dLidx, fs.data(), F * sizeof(int),
                                       cudaMemcpyHostToDevice, s));
            CUDA_CHECK(cudaMemsetAsync(ctx->ds, 0, F * sizeof(double), s));
            CUDA_CHECK(cudaMemsetAsync(ctx->dybx_raw, 0, F * sizeof(double), s));
            unsigned tiles = (unsigned)std::min<size_t>(64, (n + BLOCK - 1) / BLOCK);
            lc_fill_stats_kernel<<<dim3(std::max(tiles, 1u), (unsigned)F), block, 0, s>>>(
                ctx->dB, ctx->dXscaled, ctx->dy, n, ctx->dsrc_xcol, ctx->dsrc_basis,
                ctx->dBx_f64, ctx->ds, ctx->dybx_raw);
            check_launch();
            CUBLAS_CHECK(cublasDgemm(ctx->handle, CUBLAS_OP_T, CUBLAS_OP_N,
                                     (int)m, (int)F, (int)n, &one, ctx->dBo_f64, (int)n,
                                     ctx->dBx_f64, (int)n, &zero, ctx->dT, (int)ldT));
            lc_store_kernel<<<dim3((unsigned)((ldT + BLOCK - 1) / BLOCK), (unsigned)F),
                              block, 0, s>>>(ctx->dLidx, ctx->dT, ldT, ctx->ds,
                                             ctx->dybx_raw, m, ctx->dLT, ldT,
                                             ctx->dLnb, ctx->dLby);
            check_launch();
            const size_t warps_per_block = BLOCK / 32;
            lc_scalars_kernel<<<(unsigned)((F + warps_per_block - 1) / warps_per_block),
                                block, 0, s>>>(ctx->dLidx, F, ctx->dLT, ldT, ctx->dLE, mt,
                                               ctx->dLybo, 0, m, ctx->dLtt, ctx->dLtet,
                                               ctx->dLty);
            check_launch();
            for (size_t c = 0; c < F; ++c) {
                ctx->h_lstate[fs[c]] = 1;
                ctx->h_llive.push_back(fs[c]);
            }
            // The stage vectors are reused by the next piece: let the async
            // copies out of them finish first.
            CUDA_CHECK(cudaStreamSynchronize(s));
        }
        std::sort(ctx->h_llive.begin(), ctx->h_llive.end());
        ctx->lc_live_dirty = true;
        ctx->n_lc_fresh += (long)fresh.size();
    }
    lc_phase(ctx, t0, ctx->ms_lc_fresh);

    // 4. Score every live requested candidate in closed form.
    std::vector<int> &req = ctx->h_lreq, &pos = ctx->h_lpos;
    req.clear();
    pos.clear();
    for (size_t j = 0; j < P; ++j) {
        const size_t sl = (size_t)src_xcol[j] * mt + (size_t)src_basis[j];
        if (ctx->h_lstate[sl] == 1) {
            req.push_back((int)sl);
            pos.push_back((int)j);
        }
        dsse[j] = 0.0;                          // dead: in span, scores 0
    }
    const size_t Q = req.size();
    ctx->h_lout.resize(Q);
    ctx->h_lflag.resize(Q);
    if (Q > 0) {
        CUDA_CHECK(cudaMemcpyAsync(ctx->dLreq, req.data(), Q * sizeof(int),
                                   cudaMemcpyHostToDevice, s));
        const size_t warps_per_block = BLOCK / 32;
        lc_score_kernel<<<(unsigned)((Q + warps_per_block - 1) / warps_per_block), block, 0, s>>>(
            ctx->dLreq, Q, ctx->dLT, ldT, ctx->dLEybo, m, ctx->dLnb, ctx->dLby,
            ctx->dLtt, ctx->dLtet, ctx->dLty, tol, mars::LINEAR_KAPPA2_MAX,
            ctx->dLout, ctx->dLflag);
        check_launch();
        CUDA_CHECK(cudaMemcpyAsync(ctx->h_lout.data(), ctx->dLout, Q * sizeof(double),
                                   cudaMemcpyDeviceToHost, s));
        CUDA_CHECK(cudaMemcpyAsync(ctx->h_lflag.data(), ctx->dLflag, Q * sizeof(int),
                                   cudaMemcpyDeviceToHost, s));
        CUDA_CHECK(cudaStreamSynchronize(s));
    }
    std::vector<int> fb;                         // indices q of flagged candidates
    for (size_t q = 0; q < Q; ++q) {
        dsse[pos[q]] = ctx->h_lout[q];
        if (ctx->h_lflag[q]) {
            fb.push_back((int)q);
        }
    }
    lc_phase(ctx, t0, ctx->ms_lc_score);

    // 5. Near-degenerate candidates: explicit f64 projection from the cached T,
    //    ΔSSE = <r, y_res>² / (‖r‖² + tol) on the f32-rounded residual r (as
    //    mars::explicit_dsse); ‖r‖² <= tol means in span(Bo): dead for good.
    if (!fb.empty()) {
        std::vector<int> fx, fk, fs;
        std::vector<double> h_s, h_u;
        const size_t piece = std::min<size_t>(ctx->p_cap, 65535);
        bool died = false;
        for (size_t f0 = 0; f0 < fb.size(); f0 += piece) {
            const size_t F = std::min(piece, fb.size() - f0);
            fx.resize(F);
            fk.resize(F);
            fs.resize(F);
            h_s.resize(F);
            h_u.resize(F);
            for (size_t c = 0; c < F; ++c) {
                const int q = fb[f0 + c];
                fs[c] = req[q];
                fx[c] = (int)((size_t)req[q] / mt);
                fk[c] = (int)((size_t)req[q] % mt);
            }
            CUDA_CHECK(cudaMemcpyAsync(ctx->dsrc_xcol, fx.data(), F * sizeof(int),
                                       cudaMemcpyHostToDevice, s));
            CUDA_CHECK(cudaMemcpyAsync(ctx->dsrc_basis, fk.data(), F * sizeof(int),
                                       cudaMemcpyHostToDevice, s));
            CUDA_CHECK(cudaMemcpyAsync(ctx->dLidx, fs.data(), F * sizeof(int),
                                       cudaMemcpyHostToDevice, s));
            f64_reproject(ctx, m, F, ctx->dLT, ldT, ctx->dLidx,
                          FillSrc{ctx->dXscaled, ctx->dsrc_xcol, ctx->dsrc_basis},
                          nullptr, nullptr, ydot, /*want_dot=*/1);
            CUDA_CHECK(cudaMemcpyAsync(h_s.data(), ctx->ds, F * sizeof(double),
                                       cudaMemcpyDeviceToHost, s));
            CUDA_CHECK(cudaMemcpyAsync(h_u.data(), ctx->dybx_raw, F * sizeof(double),
                                       cudaMemcpyDeviceToHost, s));
            CUDA_CHECK(cudaStreamSynchronize(s));
            for (size_t c = 0; c < F; ++c) {
                const int q = fb[f0 + c];
                if (h_s[c] > tol) {
                    dsse[pos[q]] = h_u[c] * h_u[c] / (h_s[c] + tol);
                }
                else {
                    dsse[pos[q]] = 0.0;
                    ctx->h_lstate[fs[c]] = 2;
                    ctx->n_lc_dead++;
                    died = true;
                }
            }
        }
        if (died) {
            auto &lv = ctx->h_llive;
            lv.erase(std::remove_if(lv.begin(), lv.end(),
                                    [&](int sl) { return ctx->h_lstate[sl] != 1; }),
                     lv.end());
            ctx->lc_live_dirty = true;
        }
        ctx->n_lc_fb += (long)fb.size();
        if (fallback_counter) {
            fallback_counter->fetch_add((long)fb.size(), std::memory_order_relaxed);
        }
    }
    lc_phase(ctx, t0, ctx->ms_lc_fb);
}

}  // namespace cuda
}  // namespace mars
