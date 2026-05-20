/*
 *  BLAS-style kernels for the MARS forward pass. See kernels.h for layout
 *  contracts. Inner strides are unconditionally 1; the only parameters are
 *  outer (leading) strides that genuinely vary between callers.
 */
#include "kernels.h"
#include <algorithm>  // fill_n
#include <cmath>      // sqrt
#if defined(__AVX__)
#  include <immintrin.h>
#endif

namespace mars {
namespace {

/*
 *  Inner kernels along the basis dimension k. AVX2 unrolls 4-wide; the
 *  scalar tail covers the leftover. Hot on every column of orthonormalize().
 */

// tc[k] += scalar * bo_row[k] for k in 0..m.
inline void axpy_m(double *tc, const double *bo_row, double scalar, int m)
{
    int k = 0;
#if defined(__AVX__)
    __m256d bcast = _mm256_set1_pd(scalar);
    for (; k + 4 <= m; k += 4) {
        __m256d t  = _mm256_loadu_pd(tc + k);
        __m256d bo = _mm256_loadu_pd(bo_row + k);
        _mm256_storeu_pd(tc + k, _mm256_fmadd_pd(bo, bcast, t));
    }
#endif
    for (; k < m; ++k) {
        tc[k] += bo_row[k] * scalar;
    }
}

// dot(bo_row[:m], tc[:m]).
inline double dot_m(const double *bo_row, const double *tc, int m)
{
    int k = 0;
#if defined(__AVX__)
    __m256d acc4 = _mm256_setzero_pd();
    for (; k + 4 <= m; k += 4) {
        __m256d bo = _mm256_loadu_pd(bo_row + k);
        __m256d t  = _mm256_loadu_pd(tc + k);
        acc4 = _mm256_fmadd_pd(bo, t, acc4);
    }
    double acc = (acc4[0] + acc4[1]) + (acc4[2] + acc4[3]);
#else
    double acc = 0.0;
#endif
    for (; k < m; ++k) {
        acc += bo_row[k] * tc[k];
    }
    return acc;
}

// For each row i: bx[i] -= dot(Bo[i,:], tc). Returns sum bx[i]^2 over the
// updated column; lets the caller decide on the DGKS retry / normalization.
inline double project_subtract_and_norm(
    int n, int m,
    const double *Bo, int ldBo,
    const double *tc,
    double *bx)
{
    double s = 0.0;
    for (int i = 0; i < n; ++i) {
        double v = bx[i] - dot_m(Bo + i * ldBo, tc, m);
        bx[i] = v;
        s += v * v;
    }
    return s;
}

// tc = Bo^T * bx (single column).
inline void compute_BoT_bx_col(
    int n, int m,
    const double *Bo, int ldBo,
    const double *bx,
    double *tc)
{
    std::fill_n(tc, m, 0.0);
    for (int i = 0; i < n; ++i) {
        axpy_m(tc, Bo + i * ldBo, bx[i], m);
    }
}

} // namespace

void orthonormalize(
    int n, int m, int p,
    const float  *B,    int ldB,
    const float  *x,
    const int    *mask,
    const double *Bo,   int ldBo,
    double       *Bx,   int ldBx,
    double       *T,    int ldT,
    double       *s_buf,
    double tol,
    std::atomic<long> *dgks_counter)
{
    // ------------------------------------------------------------------------
    // Fill: Bx[i, j] = (double)(B[i, mask[j]] * x[i])
    //   The f32 multiply happens first, then the cast to f64 -- matches the
    //   prior Eigen expression `(B.col(mask[j]).array() * x).cast<double>()`.
    // ------------------------------------------------------------------------
    for (int j = 0; j < p; ++j) {
        const float *b  = B  + mask[j] * ldB;
        double      *bx = Bx + j       * ldBx;
        int i = 0;
#if defined(__AVX__)
        for (; i + 8 <= n; i += 8) {
            __m256 b8 = _mm256_loadu_ps(b + i);
            __m256 x8 = _mm256_loadu_ps(x + i);
            __m256 m8 = _mm256_mul_ps(b8, x8);
            _mm256_storeu_pd(bx + i,     _mm256_cvtps_pd(_mm256_castps256_ps128(m8)));
            _mm256_storeu_pd(bx + i + 4, _mm256_cvtps_pd(_mm256_extractf128_ps(m8, 1)));
        }
        for (; i + 4 <= n; i += 4) {
            __m128 b4 = _mm_loadu_ps(b + i);
            __m128 x4 = _mm_loadu_ps(x + i);
            _mm256_storeu_pd(bx + i, _mm256_cvtps_pd(_mm_mul_ps(b4, x4)));
        }
#endif
        for (; i < n; ++i) {
            bx[i] = (double)(b[i] * x[i]);
        }
    }

    // ------------------------------------------------------------------------
    // Phase 1: T = Bo^T * Bx
    //   Bo is row-major, so the outer loop over i loads bo_row once and reuses
    //   it across all p columns -- amortizes the row fetch.
    //
    // This is essentially a GEMM call, how would this compare to a tuned BLAS
    // implementation? For `p`, `k` in 100–500 and `n` in the millions, we'll
    // get 80–90% of the CPU's single-thread peak, which is essentially where
    // OpenBLAS would land too. The remaining 10–20% is packing. The gap is
    // small because `Bo` and `Bx` have good access patterns so the cache/TLB
    // benefits of packing are muted compared to square GEMMs.
    // ------------------------------------------------------------------------
    for (int j = 0; j < p; ++j) {
        std::fill_n(T + j * ldT, m, 0.0);
    }
    for (int i = 0; i < n; ++i) {
        const double *bo_row = Bo + i * ldBo;
        for (int j = 0; j < p; ++j) {
            axpy_m(T + j * ldT, bo_row, Bx[i + j * ldBx], m);
        }
    }

    // ------------------------------------------------------------------------
    // Phase 2: project out Bo and normalize each column of Bx, with a DGKS
    // retry when most of the column's energy ends up inside span(Bo).
    //
    // Phase 2a fuses the per-column subtract into a single sweep over Bo: each
    // row of Bo is loaded once and reused across all p columns, dropping the
    // Bo DRAM traffic from p*n*m to n*m. The per-column squared norms land in
    // s_buf for the DGKS gate and the final scale step.
    // ------------------------------------------------------------------------
    std::fill_n(s_buf, p, 0.0);
    for (int i = 0; i < n; ++i) {
        const double *bo_row = Bo + i * ldBo;
        for (int j = 0; j < p; ++j) {
            const double *tc = T  + j * ldT;
            double       *bx = Bx + j * ldBx;
            const double  v  = bx[i] - dot_m(bo_row, tc, m);
            bx[i]    = v;
            s_buf[j] += v * v;
        }
    }

    // Phase 2b: DGKS retry on the (rare) columns where most of the energy
    // landed inside span(Bo). The tol check skips columns we'd discard as
    // degenerate anyway. See DGKS_GATE_RATIO_SQ in kernels.h.
    for (int j = 0; j < p; ++j) {
        double *tc = T  + j * ldT;
        double *bx = Bx + j * ldBx;
        const double t_norm2 = dot_m(tc, tc, m);
        if (s_buf[j] > tol && s_buf[j] * DGKS_GATE_RATIO_SQ < t_norm2) {
            if (dgks_counter) {
                dgks_counter->fetch_add(1, std::memory_order_relaxed);
            }
            compute_BoT_bx_col(n, m, Bo, ldBo, bx, tc);
            s_buf[j] = project_subtract_and_norm(n, m, Bo, ldBo, tc, bx);
        }
    }

    // Phase 2c: normalize.
    for (int j = 0; j < p; ++j) {
        double      *bx    = Bx + j * ldBx;
        const double s     = s_buf[j];
        const double scale = (s > tol) ? (1.0 / std::sqrt(s + tol)) : 0.0;
        int i = 0;
#if defined(__AVX__)
        __m256d s4 = _mm256_set1_pd(scale);
        for (; i + 4 <= n; i += 4) {
            __m256d bx4 = _mm256_loadu_pd(bx + i);
            _mm256_storeu_pd(bx + i, _mm256_mul_pd(bx4, s4));
        }
#endif
        for (; i < n; ++i) {
            bx[i] *= scale;
        }
    }
}

} // namespace mars
