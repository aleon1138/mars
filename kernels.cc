/*
 *  Hand-rolled BLAS-style kernels for the MARS forward pass. See kernels.h
 *  for layout contracts. Inner strides are unconditionally 1; the only
 *  parameters are outer (leading) strides that genuinely vary between
 *  callers. Not intended as a general-purpose library.
 */
#include "kernels.h"
#include <cmath>      // sqrt
#include <cstddef>    // size_t
#if defined(__AVX__)
#  include <immintrin.h>
#endif

namespace mars {

void orthonormalize(
    int n, int m, int p,
    const float  *B,    int ldB,
    const float  *x,
    const int    *mask,
    const double *Bo,   int ldBo,
    double       *Bx,   int ldBx,
    double       *T,    int ldT,
    double tol)
{
    // ------------------------------------------------------------------------
    // Fill: Bx[i, j] = (double)(B[i, mask[j]] * x[i])
    //   The f32 multiply happens first, then the cast to f64 -- matches the
    //   prior Eigen expression `(B.col(mask[j]).array() * x).cast<double>()`.
    // ------------------------------------------------------------------------
    for (int j = 0; j < p; ++j) {
        const float *bcol = B + (size_t)mask[j] * (size_t)ldB;
        double      *bxc  = Bx + (size_t)j      * (size_t)ldBx;
        int i = 0;
#if defined(__AVX__)
        for (; i + 4 <= n; i += 4) {
            __m128  b4  = _mm_loadu_ps(bcol + i);
            __m128  x4  = _mm_loadu_ps(x    + i);
            __m256d v   = _mm256_cvtps_pd(_mm_mul_ps(b4, x4));
            _mm256_storeu_pd(bxc + i, v);
        }
#endif
        for (; i < n; ++i) {
            bxc[i] = (double)(bcol[i] * x[i]);
        }
    }

    // ------------------------------------------------------------------------
    // Phase 1: T = Bo^T * Bx
    //   T[k, j] = sum_i Bo[i, k] * Bx[i, j]
    //
    //   Bo is row-major, so Bo[i, :] is contiguous -- 4-wide AVX2 FMA along k
    //   with a broadcasted Bx[i, j] scalar.
    // ------------------------------------------------------------------------
    for (int j = 0; j < p; ++j) {
        double *tc = T + (size_t)j * (size_t)ldT;
        for (int k = 0; k < m; ++k) tc[k] = 0.0;
    }
    for (int i = 0; i < n; ++i) {
        const double *bo_row = Bo + (size_t)i * (size_t)ldBo;
        for (int j = 0; j < p; ++j) {
            double  bx_ij = Bx[(size_t)i + (size_t)j * (size_t)ldBx];
            double *tc    = T  + (size_t)j * (size_t)ldT;
#if defined(__AVX__)
            __m256d bcast = _mm256_set1_pd(bx_ij);
            int k = 0;
            for (; k + 4 <= m; k += 4) {
                __m256d t  = _mm256_loadu_pd(tc + k);
                __m256d bo = _mm256_loadu_pd(bo_row + k);
                t = _mm256_fmadd_pd(bo, bcast, t);
                _mm256_storeu_pd(tc + k, t);
            }
            for (; k < m; ++k) tc[k] += bo_row[k] * bx_ij;
#else
            for (int k = 0; k < m; ++k) tc[k] += bo_row[k] * bx_ij;
#endif
        }
    }

    // ------------------------------------------------------------------------
    // Phase 2: For each column j of Bx, fused project-out + normalize.
    //   sweep i: Bx[i,j] -= dot(Bo[i,:], T[:,j]); s += Bx[i,j]^2
    //   scale  = (s > tol) ? 1/sqrt(s + tol) : 0
    //   sweep i: Bx[i,j] *= scale
    //
    //   Each (i, j) inner loop is a dot product along k -- the same shape as
    //   Phase 1's inner loop, so AVX2 vectorizes the same way.
    // ------------------------------------------------------------------------
    for (int j = 0; j < p; ++j) {
        const double *tc = T  + (size_t)j * (size_t)ldT;
        double       *bx = Bx + (size_t)j * (size_t)ldBx;
        double s = 0.0;

        for (int i = 0; i < n; ++i) {
            const double *bo_row = Bo + (size_t)i * (size_t)ldBo;
            double acc;
#if defined(__AVX__)
            __m256d acc4 = _mm256_setzero_pd();
            int k = 0;
            for (; k + 4 <= m; k += 4) {
                __m256d bo = _mm256_loadu_pd(bo_row + k);
                __m256d t  = _mm256_loadu_pd(tc + k);
                acc4 = _mm256_fmadd_pd(bo, t, acc4);
            }
            acc = (acc4[0] + acc4[1]) + (acc4[2] + acc4[3]);
            for (; k < m; ++k) acc += bo_row[k] * tc[k];
#else
            acc = 0.0;
            for (int k = 0; k < m; ++k) acc += bo_row[k] * tc[k];
#endif
            double v = bx[i] - acc;
            bx[i] = v;
            s += v * v;
        }

        const double scale = (s > tol) ? (1.0 / std::sqrt(s + tol)) : 0.0;

#if defined(__AVX__)
        __m256d sb = _mm256_set1_pd(scale);
        int i = 0;
        for (; i + 4 <= n; i += 4) {
            __m256d b = _mm256_loadu_pd(bx + i);
            _mm256_storeu_pd(bx + i, _mm256_mul_pd(b, sb));
        }
        for (; i < n; ++i) bx[i] *= scale;
#else
        for (int i = 0; i < n; ++i) bx[i] *= scale;
#endif
    }
}

} // namespace mars
