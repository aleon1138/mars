/*
 *  BLAS-style kernels for the MARS forward pass. See kernels.h for layout
 *  contracts. Inner strides are unconditionally 1; the only parameters are
 *  outer (leading) strides that genuinely vary between callers.
 */
#include "kernels.h"
#include <algorithm>  // fill_n
#include <cmath>      // sqrt
#include <cstddef>    // size_t
#include <vector>
#if defined(__AVX__)
#  include <immintrin.h>
#endif

namespace mars {
namespace {

#if defined(__AVX__)
// Horizontal sum of the four f64 lanes.
inline double hsum256_pd(__m256d v)
{
    return (v[0] + v[1]) + (v[2] + v[3]);
}

// Widen 4 contiguous f32 to an f64 vector (cvtps_pd on an unaligned load).
inline __m256d loadu_ps_pd(const float *p)
{
    return _mm256_cvtps_pd(_mm_loadu_ps(p));
}
#endif

/*
 *  Inner kernels along the basis dimension k. AVX2 unrolls 4-wide; the
 *  scalar tail covers the leftover. Hot on every column of orthonormalize().
 *  Bo is f32 storage; values are upcast to f64 on the load (cvtps_pd).
 */

// tc[k] += scalar * (double)bo_row[k] for k in 0..m.
inline void axpy_m(double *tc, const float *bo_row, double scalar, size_t m)
{
    size_t k = 0;
#if defined(__AVX__)
    __m256d bcast = _mm256_set1_pd(scalar);
    for (; k + 4 <= m; k += 4) {
        __m256d t  = _mm256_loadu_pd(tc + k);
        __m256d bo = loadu_ps_pd(bo_row + k);
        _mm256_storeu_pd(tc + k, _mm256_fmadd_pd(bo, bcast, t));
    }
#endif
    for (; k < m; ++k) {
        tc[k] += (double)bo_row[k] * scalar;
    }
}

// dot((double)bo_row[:m], tc[:m]).  Bo is f32, tc is f64. Two accumulators
// break the FMA latency chain (one chain caps at ~1/4 FMA throughput).
inline double dot_bo(const float *bo_row, const double *tc, size_t m)
{
    size_t k = 0;
#if defined(__AVX__)
    __m256d acc0 = _mm256_setzero_pd();
    __m256d acc1 = _mm256_setzero_pd();
    for (; k + 8 <= m; k += 8) {
        acc0 = _mm256_fmadd_pd(loadu_ps_pd(bo_row + k),     _mm256_loadu_pd(tc + k),     acc0);
        acc1 = _mm256_fmadd_pd(loadu_ps_pd(bo_row + k + 4), _mm256_loadu_pd(tc + k + 4), acc1);
    }
    if (k + 4 <= m) {
        acc0 = _mm256_fmadd_pd(loadu_ps_pd(bo_row + k), _mm256_loadu_pd(tc + k), acc0);
        k += 4;
    }
    double acc = hsum256_pd(_mm256_add_pd(acc0, acc1));
#else
    double acc = 0.0;
#endif
    for (; k < m; ++k) {
        acc += (double)bo_row[k] * tc[k];
    }
    return acc;
}

/*
 *  Four dot_bo() reductions sharing the bo_row load+widen: out[c] =
 *  dot((double)bo_row[:m], t{c}[:m]). The four independent accumulators also
 *  break the FMA latency chain. Hot in orthonormalize() Phase 2a, where the
 *  same bo_row is projected against every candidate column's T-coefficients.
 */
inline void dot_bo4(const float *bo_row,
                    const double *t0, const double *t1,
                    const double *t2, const double *t3,
                    size_t m, double out[4])
{
    size_t k = 0;
#if defined(__AVX__)
    __m256d a0 = _mm256_setzero_pd(), a1 = _mm256_setzero_pd();
    __m256d a2 = _mm256_setzero_pd(), a3 = _mm256_setzero_pd();
    for (; k + 4 <= m; k += 4) {
        __m256d bo = loadu_ps_pd(bo_row + k);
        a0 = _mm256_fmadd_pd(bo, _mm256_loadu_pd(t0 + k), a0);
        a1 = _mm256_fmadd_pd(bo, _mm256_loadu_pd(t1 + k), a1);
        a2 = _mm256_fmadd_pd(bo, _mm256_loadu_pd(t2 + k), a2);
        a3 = _mm256_fmadd_pd(bo, _mm256_loadu_pd(t3 + k), a3);
    }
    double s0 = hsum256_pd(a0), s1 = hsum256_pd(a1);
    double s2 = hsum256_pd(a2), s3 = hsum256_pd(a3);
#else
    double s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0;
#endif
    for (; k < m; ++k) {
        const double bo = (double)bo_row[k];
        s0 += bo * t0[k];
        s1 += bo * t1[k];
        s2 += bo * t2[k];
        s3 += bo * t3[k];
    }
    out[0] = s0;
    out[1] = s1;
    out[2] = s2;
    out[3] = s3;
}

/*
 *  dot(a[:m], b[:m]) for f64*f64. Used only for the DGKS gate (tc dot tc),
 *  where both operands are the f64 T workspace. Two accumulators as in dot_bo.
 */
inline double dot_m(const double *a, const double *b, size_t m)
{
    size_t k = 0;
#if defined(__AVX__)
    __m256d acc0 = _mm256_setzero_pd();
    __m256d acc1 = _mm256_setzero_pd();
    for (; k + 8 <= m; k += 8) {
        acc0 = _mm256_fmadd_pd(_mm256_loadu_pd(a + k),     _mm256_loadu_pd(b + k),     acc0);
        acc1 = _mm256_fmadd_pd(_mm256_loadu_pd(a + k + 4), _mm256_loadu_pd(b + k + 4), acc1);
    }
    if (k + 4 <= m) {
        acc0 = _mm256_fmadd_pd(_mm256_loadu_pd(a + k), _mm256_loadu_pd(b + k), acc0);
        k += 4;
    }
    double acc = hsum256_pd(_mm256_add_pd(acc0, acc1));
#else
    double acc = 0.0;
#endif
    for (; k < m; ++k) {
        acc += a[k] * b[k];
    }
    return acc;
}

/*
 *  Column selector for the batched phases below: with Indexed the c-th active
 *  column is cols[c] (the DGKS retry's compacted flagged set), otherwise c
 *  itself (the main pass, compiled exactly as the plain loops it replaced).
 */
template <bool Indexed>
inline size_t sel_col(const int *cols, size_t c)
{
    if constexpr (Indexed) {
        return (size_t)cols[c];
    }
    else {
        (void)cols;
        return c;
    }
}

/*
 *  Phase 1 over a column set: T[:, j] = Bo^T * Bx[:, j] for each selected j,
 *  an f32-in / f64-accumulate GEMM no BLAS offers (sgemm accumulates in f32,
 *  dgemm needs f64 inputs). Bo is row-major, so the outer loop over i loads
 *  bo_row once and reuses it across all nc columns; Bx[i,j] is upcast at the
 *  axpy_m call site. With i outermost every T element is stored on every FMA,
 *  which makes this phase store-port-bound (~2.6x off the FMA peak); an
 *  i-blocked register tile would fix that -- TODO.md item 3.
 */
template <bool Indexed>
inline void bo_t_bx(size_t n, size_t m, const int *cols, size_t nc,
                    const float *Bo, size_t ldBo,
                    const float *Bx, size_t ldBx,
                    double *T, size_t ldT)
{
    for (size_t c = 0; c < nc; ++c) {
        std::fill_n(T + sel_col<Indexed>(cols, c) * ldT, m, 0.0);
    }
    for (size_t i = 0; i < n; ++i) {
        const float *bo_row = Bo + i * ldBo;
        for (size_t c = 0; c < nc; ++c) {
            const size_t j = sel_col<Indexed>(cols, c);
            axpy_m(T + j * ldT, bo_row, (double)Bx[i + j * ldBx], m);
        }
    }
}

/*
 *  Phase 2a over a column set: Bx[:, j] -= Bo * T[:, j] and s_buf[j] =
 *  ||Bx[:, j]||^2 for each selected j, fused into a single sweep over Bo: each
 *  row of Bo is loaded once and reused across all nc columns, so the Bo DRAM
 *  traffic is n*m rather than nc*n*m. The subtract is done in f64, the result
 *  rounded back to f32 on store; s_buf accumulates the *stored* squared values
 *  so the downstream normalization sees the same magnitude that's in memory.
 */
template <bool Indexed>
inline void project_out(size_t n, size_t m, const int *cols, size_t nc,
                        const float *Bo, size_t ldBo,
                        const double *T, size_t ldT,
                        float *Bx, size_t ldBx,
                        double *s_buf)
{
    for (size_t c = 0; c < nc; ++c) {
        s_buf[sel_col<Indexed>(cols, c)] = 0.0;
    }
    for (size_t i = 0; i < n; ++i) {
        const float *bo_row = Bo + i * ldBo;
        size_t c = 0;
        // Blocks of 4 columns share the bo_row load+widen (see dot_bo4).
        for (; c + 4 <= nc; c += 4) {
            size_t j[4];
            for (int q = 0; q < 4; ++q) {
                j[q] = sel_col<Indexed>(cols, c + q);
            }
            double proj[4];
            dot_bo4(bo_row,
                    T + j[0] * ldT, T + j[1] * ldT,
                    T + j[2] * ldT, T + j[3] * ldT,
                    m, proj);
            for (int q = 0; q < 4; ++q) {
                float       *bx    = Bx + j[q] * ldBx;
                const double v     = (double)bx[i] - proj[q];
                const float  v_f32 = (float)v;
                bx[i]    = v_f32;
                const double v_back = (double)v_f32;
                s_buf[j[q]] += v_back * v_back;
            }
        }
        for (; c < nc; ++c) {
            const size_t  jc    = sel_col<Indexed>(cols, c);
            const double *tc    = T  + jc * ldT;
            float        *bx    = Bx + jc * ldBx;
            const double  v     = (double)bx[i] - dot_bo(bo_row, tc, m);
            const float   v_f32 = (float)v;
            bx[i]    = v_f32;
            const double v_back = (double)v_f32;
            s_buf[jc] += v_back * v_back;
        }
    }
}

/*
 *  One classical Gram-Schmidt pass of the f64 column v against the m
 *  orthonormal columns of Bo: c = Bo^T v, then v -= Bo c. Both halves walk Bo
 *  row by row (row-major, stride ldBo), so the whole pass is two streaming
 *  sweeps over Bo regardless of m -- a per-column walk would touch one cache
 *  line per row per column (m times the traffic). Returns the projected energy
 *  sum_j c_j^2 (feeds the DGKS gate) and the residual energy ||v||^2 in v_norm2.
 *  `c` is an m-length workspace.
 */
inline double cgs_project_col(
    size_t n, size_t m,
    double *v,
    const float *Bo, size_t ldBo,
    double *c, double &v_norm2)
{
    std::fill_n(c, m, 0.0);
    for (size_t i = 0; i < n; ++i) {
        axpy_m(c, Bo + i * ldBo, v[i], m);
    }
    v_norm2 = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const double r = v[i] - dot_bo(Bo + i * ldBo, c, m);
        v[i] = r;
        v_norm2 += r * r;
    }
    return dot_m(c, c, m);
}

/*
 *  bx[i] = b[i] * x[i] (f32 multiply, stored f32 -- the same rounding as the
 *  fill in orthonormalize()), and, from the stored values, nb = ||bx||^2 and
 *  by = <bx, y> accumulated in f64.
 */
inline void fill_col_stats(const float *b, const float *x, const float *y,
                           float *bx, size_t n, double &nb, double &by)
{
    size_t i = 0;
#if defined(__AVX__)
    __m256d n0 = _mm256_setzero_pd(), n1 = _mm256_setzero_pd();
    __m256d y0 = _mm256_setzero_pd(), y1 = _mm256_setzero_pd();
    for (; i + 8 <= n; i += 8) {
        const __m256 v8 = _mm256_mul_ps(_mm256_loadu_ps(b + i), _mm256_loadu_ps(x + i));
        _mm256_storeu_ps(bx + i, v8);
        const __m256d vlo = _mm256_cvtps_pd(_mm256_castps256_ps128(v8));
        const __m256d vhi = _mm256_cvtps_pd(_mm256_extractf128_ps(v8, 1));
        n0 = _mm256_fmadd_pd(vlo, vlo, n0);
        n1 = _mm256_fmadd_pd(vhi, vhi, n1);
        y0 = _mm256_fmadd_pd(vlo, loadu_ps_pd(y + i),     y0);
        y1 = _mm256_fmadd_pd(vhi, loadu_ps_pd(y + i + 4), y1);
    }
    double sn = hsum256_pd(_mm256_add_pd(n0, n1));
    double sy = hsum256_pd(_mm256_add_pd(y0, y1));
#else
    double sn = 0.0, sy = 0.0;
#endif
    for (; i < n; ++i) {
        const float v = b[i] * x[i];
        bx[i] = v;
        sn += (double)v * (double)v;
        sy += (double)v * (double)y[i];
    }
    nb = sn;
    by = sy;
}

} // namespace

double dot_widen(const float *a, const float *b, size_t n)
{
    size_t i = 0;
#if defined(__AVX__)
    /*
     *  Two accumulators so the two FMAs per iteration are independent; the
     *  f32->f64 widening (cvtps_pd) keeps the sum at the eps_f32 input floor.
     */
    __m256d acc0 = _mm256_setzero_pd();
    __m256d acc1 = _mm256_setzero_pd();
    for (; i + 8 <= n; i += 8) {
        acc0 = _mm256_fmadd_pd(loadu_ps_pd(a + i),     loadu_ps_pd(b + i),     acc0);
        acc1 = _mm256_fmadd_pd(loadu_ps_pd(a + i + 4), loadu_ps_pd(b + i + 4), acc1);
    }
    double s = hsum256_pd(_mm256_add_pd(acc0, acc1));
#else
    double s = 0.0;
#endif
    for (; i < n; ++i) {
        s += (double)a[i] * (double)b[i];
    }
    return s;
}

double orthonormalize_col(
    size_t n, size_t m,
    double       *v,
    const float  *Bo,   size_t ldBo,
    double tol,
    std::atomic<long> *dgks_counter)
{
    std::vector<double> c(m);
    double v_norm2 = 0.0;
    const double proj_norm2 = cgs_project_col(n, m, v, Bo, ldBo, c.data(), v_norm2);

    /*
     *  DGKS retry: re-orthogonalize once when the residual energy is small
     *  relative to what was projected out. (The retry's own projected energy
     *  is tiny by construction and discarded.)
     */
    if (v_norm2 > tol && v_norm2 * DGKS_GATE_RATIO_SQ < proj_norm2) {
        if (dgks_counter) {
            dgks_counter->fetch_add(1, std::memory_order_relaxed);
        }
        cgs_project_col(n, m, v, Bo, ldBo, c.data(), v_norm2);
    }

    const double w = std::sqrt(v_norm2);
    if (w * w > tol) {
        for (size_t i = 0; i < n; ++i) {
            v[i] /= w;
        }
    }
    return w;
}

void orthonormalize(
    size_t n, size_t m, size_t p,
    const float  *B,    size_t ldB,
    const float  *x,
    const int    *mask,
    const float  *Bo,   size_t ldBo,
    float        *Bx,   size_t ldBx,
    double       *T,    size_t ldT,
    double       *s_buf,
    double tol,
    std::atomic<long> *dgks_counter)
{
    /*
     *  Bx[i, j] = B[i, mask[j]] * x[i]  -- f32 * f32 stored as f32
     */
    for (size_t j = 0; j < p; ++j) {
        const float *b  = B  + mask[j] * ldB;
        float       *bx = Bx + j       * ldBx;
        size_t i = 0;
#if defined(__AVX__)
        for (; i + 8 <= n; i += 8) {
            __m256 b8 = _mm256_loadu_ps(b + i);
            __m256 x8 = _mm256_loadu_ps(x + i);
            _mm256_storeu_ps(bx + i, _mm256_mul_ps(b8, x8));
        }
#endif
        for (; i < n; ++i) {
            bx[i] = b[i] * x[i];
        }
    }

    /*
     *  Phase 1: T = Bo^T * Bx (see bo_t_bx).
     */
    bo_t_bx<false>(n, m, nullptr, p, Bo, ldBo, Bx, ldBx, T, ldT);

    /*
     *  Phase 2a: project out Bo and take each column's squared norm, in one
     *  sweep over Bo (see project_out).
     */
    project_out<false>(n, m, nullptr, p, Bo, ldBo, T, ldT, Bx, ldBx, s_buf);

    /*
     *  Phase 2b: DGKS retry on the columns where most of the energy landed
     *  inside span(Bo). The tol check skips columns we'd discard as degenerate
     *  anyway. See DGKS_GATE_RATIO_SQ in kernels.h.
     *
     *  The flagged columns are re-run through the same batched Phase 1/2a,
     *  compacted to an index list: two sweeps over Bo in total rather than two
     *  per flagged column. The retry is not rare -- on uncentered inputs most
     *  candidates are dominated by their in-span part (x = mean + noise, and
     *  b * mean is in span), and the per-column retry doubled eval(linear).
     */
    thread_local std::vector<int> flagged;
    flagged.clear();
    for (size_t j = 0; j < p; ++j) {
        const double *tc      = T + j * ldT;
        const double  t_norm2 = dot_m(tc, tc, m);
        if (s_buf[j] > tol && s_buf[j] * DGKS_GATE_RATIO_SQ < t_norm2) {
            flagged.push_back((int)j);
        }
    }
    if (!flagged.empty()) {
        if (dgks_counter) {
            dgks_counter->fetch_add((long)flagged.size(), std::memory_order_relaxed);
        }
        const size_t nf = flagged.size();
        bo_t_bx<true>(n, m, flagged.data(), nf, Bo, ldBo, Bx, ldBx, T, ldT);
        project_out<true>(n, m, flagged.data(), nf, Bo, ldBo, T, ldT, Bx, ldBx, s_buf);
    }

    /*
     *  Phase 2c: normalize. f32 multiply; the scale stays f64 only until the
     *  store cast, since the column is already f32-bounded.
     */
    for (size_t j = 0; j < p; ++j) {
        float       *bx    = Bx + j * ldBx;
        const double s     = s_buf[j];
        const float  scale = (s > tol) ? (float)(1.0 / std::sqrt(s + tol)) : 0.0f;
        size_t i = 0;
#if defined(__AVX__)
        __m256 s8 = _mm256_set1_ps(scale);
        for (; i + 8 <= n; i += 8) {
            __m256 bx8 = _mm256_loadu_ps(bx + i);
            _mm256_storeu_ps(bx + i, _mm256_mul_ps(bx8, s8));
        }
#endif
        for (; i < n; ++i) {
            bx[i] *= scale;
        }
    }
}

void gram_col(size_t n, size_t m1, const float *Bo, size_t ldBo, size_t col, double *g)
{
    std::fill_n(g, m1, 0.0);
    for (size_t i = 0; i < n; ++i) {
        const float *row = Bo + i * ldBo;
        axpy_m(g, row, (double)row[col], m1);
    }
}

void linear_dsse(
    size_t n, size_t m, size_t p,
    const float  *B,    size_t ldB,
    const float  *x,
    const int    *mask,
    const float  *Bo,   size_t ldBo,
    const float  *y,
    const float  *yres,
    const double *ybo,
    const double *E,    size_t ldE,
    const double *Eybo,
    float        *Bx,   size_t ldBx,
    double       *T,    size_t ldT,
    double       *work,
    double tol,
    double       *dsse,
    std::atomic<long> *fallback_counter)
{
    double *nb = work;           // (p) ||Bx_j||^2
    double *by = work + p;       // (p) <Bx_j, y>
    double *sb = work + 2 * p;   // (p) explicit-path residual energy
    double *et = work + 3 * p;   // (m) E * T_j

    /*
     *  Fill Bx with the column statistics, then Phase 1: T = Bo^T * Bx.
     */
    for (size_t j = 0; j < p; ++j) {
        fill_col_stats(B + (size_t)mask[j] * ldB, x, y, Bx + j * ldBx, n, nb[j], by[j]);
    }
    bo_t_bx<false>(n, m, nullptr, p, Bo, ldBo, Bx, ldBx, T, ldT);

    /*
     *  Closed form, first order in E. O(m^2) per column for T'E T against the
     *  O(n*m) of Phase 1. Columns too close to span(Bo) for the f64
     *  cancellation in den are queued for the explicit path.
     */
    thread_local std::vector<int> fallback;
    fallback.clear();
    for (size_t j = 0; j < p; ++j) {
        const double *tc = T + j * ldT;
        for (size_t k = 0; k < m; ++k) {
            et[k] = dot_m(E + k * ldE, tc, m);   // E symmetric: column k == row k
        }
        const double den = nb[j] - dot_m(tc, tc, m) + dot_m(tc, et, m);
        if (den * LINEAR_KAPPA2_MAX < nb[j]) {
            fallback.push_back((int)j);
            continue;
        }
        const double u = by[j] - dot_m(tc, ybo, m) + dot_m(tc, Eybo, m);
        dsse[j] = (den > tol) ? u * u / (den + tol) : 0.0;
    }
    if (fallback.empty()) {
        return;
    }
    if (fallback_counter) {
        fallback_counter->fetch_add((long)fallback.size(), std::memory_order_relaxed);
    }

    /*
     *  Explicit path on the near-degenerate columns: project with the Phase 1
     *  T, then a DGKS retry where gated (as in orthonormalize()), and score the
     *  f32 residual against yres in f64.
     */
    const int   *fb = fallback.data();
    const size_t nf = fallback.size();
    project_out<true>(n, m, fb, nf, Bo, ldBo, T, ldT, Bx, ldBx, sb);
    thread_local std::vector<int> retry;
    retry.clear();
    for (size_t c = 0; c < nf; ++c) {
        const size_t  j  = (size_t)fb[c];
        const double *tc = T + j * ldT;
        if (sb[j] > tol && sb[j] * DGKS_GATE_RATIO_SQ < dot_m(tc, tc, m)) {
            retry.push_back((int)j);
        }
    }
    if (!retry.empty()) {
        bo_t_bx<true>(n, m, retry.data(), retry.size(), Bo, ldBo, Bx, ldBx, T, ldT);
        project_out<true>(n, m, retry.data(), retry.size(), Bo, ldBo, T, ldT, Bx, ldBx, sb);
    }
    for (size_t c = 0; c < nf; ++c) {
        const size_t j = (size_t)fb[c];
        if (sb[j] > tol) {
            const double u = dot_widen(Bx + j * ldBx, yres, n);
            dsse[j] = u * u / (sb[j] + tol);
        }
        else {
            dsse[j] = 0.0;
        }
    }
}

} // namespace mars
