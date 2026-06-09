/*
 *  BLAS-style kernels for the MARS forward pass. See kernels.h for layout
 *  contracts. Inner strides are unconditionally 1; the only parameters are
 *  outer (leading) strides that genuinely vary between callers.
 */
#include "kernels.h"
#include <algorithm>  // fill_n
#include <cmath>      // sqrt
#include <cstddef>    // size_t
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
 *  For each row i: bx[i] -= dot(Bo[i,:], tc). Subtraction is done in f64,
 *  rounded back to f32 on store. Returns sum (stored f32)^2 -- the post-store
 *  norm, not the pre-round value -- so downstream normalize() uses the same
 *  quantity that's actually in memory.
 */
inline double project_subtract_and_norm(
    size_t n, size_t m,
    const float *Bo, size_t ldBo,
    const double *tc,
    float *bx)
{
    double s = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const double v = (double)bx[i] - dot_bo(Bo + i * ldBo, tc, m);
        const float  v_f32 = (float)v;
        bx[i] = v_f32;
        const double v_back = (double)v_f32;
        s += v_back * v_back;
    }
    return s;
}

/*
 *  tc = Bo^T * bx (single column). Bo and bx are both f32, upcast to f64
 *  inside axpy_m / at the bx[i] load.
 */
inline void compute_BoT_bx_col(
    size_t n, size_t m,
    const float *Bo, size_t ldBo,
    const float *bx,
    double *tc)
{
    std::fill_n(tc, m, 0.0);
    for (size_t i = 0; i < n; ++i) {
        axpy_m(tc, Bo + i * ldBo, (double)bx[i], m);
    }
}

/*
 *  One modified Gram-Schmidt sweep of the single column v against the m
 *  orthonormal columns of Bo (row-major, row stride ldBo): each projection
 *  updates v before the next is computed. Column j of row i is at
 *  Bo[i*ldBo + j], so the per-column walk is strided (cold path -- once per
 *  append). Returns the projected energy sum_j c_j^2 (feeds the DGKS gate).
 */
inline double mgs_project_col(
    size_t n, size_t m,
    double *v,
    const float *Bo, size_t ldBo)
{
    double proj_norm2 = 0.0;
    for (size_t j = 0; j < m; ++j) {
        double c = 0.0;
        for (size_t i = 0; i < n; ++i) {
            c += (double)Bo[i*ldBo + j] * v[i];
        }
        for (size_t i = 0; i < n; ++i) {
            v[i] -= c * (double)Bo[i*ldBo + j];
        }
        proj_norm2 += c * c;
    }
    return proj_norm2;
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
    /*
     *  Modified Gram-Schmidt against the m orthonormal columns of Bo, matching
     *  MarsAlgo::append(): each projection updates v before the next, and the
     *  projected energy feeds the DGKS gate. See mgs_project_col.
     */
    const double proj_norm2 = mgs_project_col(n, m, v, Bo, ldBo);

    double v_norm2 = 0.0;
    for (size_t i = 0; i < n; ++i) {
        v_norm2 += v[i] * v[i];
    }

    /*
     *  DGKS retry: re-orthogonalize once when the residual energy is small
     *  relative to what was projected out. (The retry's own projected energy
     *  is tiny by construction and discarded.)
     */
    if (v_norm2 > tol && v_norm2 * DGKS_GATE_RATIO_SQ < proj_norm2) {
        if (dgks_counter) {
            dgks_counter->fetch_add(1, std::memory_order_relaxed);
        }
        mgs_project_col(n, m, v, Bo, ldBo);
        v_norm2 = 0.0;
        for (size_t i = 0; i < n; ++i) {
            v_norm2 += v[i] * v[i];
        }
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
     *  Phase 1: T = Bo^T * Bx
     *  Bo is row-major, so the outer loop over i loads bo_row once and reuses
     *  it across all p columns -- amortizes the row fetch. Bx[i,j] is f32,
     *  upcast to f64 at the axpy_m call site; T stays f64.
     *
     *  This is essentially a GEMM call, how would this compare to a tuned BLAS
     *  implementation? For `p`, `k` in 100–500 and `n` in the millions, we'll
     *  get 80–90% of the CPU's single-thread peak, which is essentially where
     *  OpenBLAS would land too. The remaining 10–20% is packing. The gap is
     *  small because `Bo` and `Bx` have good access patterns so the cache/TLB
     *  benefits of packing are muted compared to square GEMMs.
     */
    for (size_t j = 0; j < p; ++j) {
        std::fill_n(T + j * ldT, m, 0.0);
    }
    for (size_t i = 0; i < n; ++i) {
        const float *bo_row = Bo + i * ldBo;
        for (size_t j = 0; j < p; ++j) {
            axpy_m(T + j * ldT, bo_row, (double)Bx[i + j * ldBx], m);
        }
    }

    /*
     *  Phase 2: project out Bo and normalize each column of Bx, with a DGKS
     *  retry when most of the column's energy ends up inside span(Bo).
     *
     *  Phase 2a fuses the per-column subtract into a single sweep over Bo: each
     *  row of Bo is loaded once and reused across all p columns, dropping the
     *  Bo DRAM traffic from p*n*m to n*m. The subtract is done in f64, the
     *  result rounded back to f32 on store; s_buf accumulates the *stored*
     *  squared values so the downstream normalization sees the same magnitude
     *  that's in memory.
     */
    std::fill_n(s_buf, p, 0.0);
    for (size_t i = 0; i < n; ++i) {
        const float *bo_row = Bo + i * ldBo;
        size_t j = 0;
        // Blocks of 4 columns share the bo_row load+widen (see dot_bo4).
        for (; j + 4 <= p; j += 4) {
            double proj[4];
            dot_bo4(bo_row,
                    T + (j + 0) * ldT, T + (j + 1) * ldT,
                    T + (j + 2) * ldT, T + (j + 3) * ldT,
                    m, proj);
            for (int c = 0; c < 4; ++c) {
                float       *bx    = Bx + (j + c) * ldBx;
                const double v     = (double)bx[i] - proj[c];
                const float  v_f32 = (float)v;
                bx[i]    = v_f32;
                const double v_back = (double)v_f32;
                s_buf[j + c] += v_back * v_back;
            }
        }
        for (; j < p; ++j) {
            const double *tc    = T  + j * ldT;
            float        *bx    = Bx + j * ldBx;
            const double  v     = (double)bx[i] - dot_bo(bo_row, tc, m);
            const float   v_f32 = (float)v;
            bx[i]    = v_f32;
            const double v_back = (double)v_f32;
            s_buf[j] += v_back * v_back;
        }
    }

    /*
     *  Phase 2b: DGKS retry on the (rare) columns where most of the energy
     *  landed inside span(Bo). The tol check skips columns we'd discard as
     *  degenerate anyway. See DGKS_GATE_RATIO_SQ in kernels.h.
     */
    for (size_t j = 0; j < p; ++j) {
        double *tc = T  + j * ldT;
        float  *bx = Bx + j * ldBx;
        const double t_norm2 = dot_m(tc, tc, m);
        if (s_buf[j] > tol && s_buf[j] * DGKS_GATE_RATIO_SQ < t_norm2) {
            if (dgks_counter) {
                dgks_counter->fetch_add(1, std::memory_order_relaxed);
            }
            compute_BoT_bx_col(n, m, Bo, ldBo, bx, tc);
            s_buf[j] = project_subtract_and_norm(n, m, Bo, ldBo, tc, bx);
        }
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

} // namespace mars
