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
 *  Phase 1, register-tiled (CLAUDE.md, 2026-11-04 note). The rows are walked
 *  in panels of P1_ROWS. Per panel, the Bo rows are packed as f64 in k-tile order (P1_KT
 *  consecutive T rows per tile, the panel's rows contiguous within a tile) and
 *  the selected Bx columns are widened to f64. A P1_KT x P1_JT block of T is
 *  then accumulated in registers across the whole panel and added to T once,
 *  so each T load/store serves P1_ROWS FMAs instead of one (the unblocked loop
 *  stored a T element on every FMA and was store-port-bound, ~2.6x off the FMA
 *  peak). Products are exact in f64 (f32 x f32); the sums become two-level
 *  (within the panel, then across panels), which is slightly more accurate
 *  than the old row-sequential order but not bit-identical to it.
 */
constexpr size_t P1_ROWS = 128;   // rows per panel: packed Bo panel m*128*8 B (100 KB at m=100)
constexpr size_t P1_KT   = 12;    // T rows per register tile (three 4-wide f64 vectors)
constexpr size_t P1_JT   = 4;     // T columns per register tile

/*
 *  T[k0 : k0+kn, j] += sum_r bo[r, :] * bx[c][r] for the P1_JT columns tcol[c]
 *  (the f64 AVX2 tile; an AVX-512 16x4 variant measured 15-30% slower on the
 *  EPYC box, in line with the 2026-05-16 AVX-512 note).
 *  bo: the packed tile, P1_KT doubles per row; bx: P1_JT widened columns, each
 *  `ldbx` doubles apart. kn < P1_KT only on the last tile (the packed tail is
 *  zero, so only the store needs the bound).
 */
inline void p1_tile(size_t r, const double *bo, const double *bx, size_t ldbx,
                    double *const tcol[P1_JT], size_t kn)
{
#if defined(__AVX__)
    /*
     *  12 accumulators + 3 Bo vectors + 1 broadcast = all 16 YMM registers:
     *  7 loads per 12 FMAs (an 8x4 tile was 6 per 8, ~10% slower at m=250).
     */
    __m256d a[P1_JT][3];
    for (size_t c = 0; c < P1_JT; ++c) {
        a[c][0] = a[c][1] = a[c][2] = _mm256_setzero_pd();
    }
    const double *xs[P1_JT] = {bx, bx + ldbx, bx + 2 * ldbx, bx + 3 * ldbx};
    for (size_t i = 0; i < r; ++i) {
        const __m256d b0 = _mm256_loadu_pd(bo + i * P1_KT);
        const __m256d b1 = _mm256_loadu_pd(bo + i * P1_KT + 4);
        const __m256d b2 = _mm256_loadu_pd(bo + i * P1_KT + 8);
        for (size_t c = 0; c < P1_JT; ++c) {
            const __m256d x = _mm256_broadcast_sd(xs[c] + i);
            a[c][0] = _mm256_fmadd_pd(b0, x, a[c][0]);
            a[c][1] = _mm256_fmadd_pd(b1, x, a[c][1]);
            a[c][2] = _mm256_fmadd_pd(b2, x, a[c][2]);
        }
    }
    alignas(32) double acc[P1_JT][P1_KT];
    for (size_t c = 0; c < P1_JT; ++c) {
        for (size_t h = 0; h < 3; ++h) {
            _mm256_store_pd(acc[c] + 4 * h, a[c][h]);
        }
    }
#else
    double acc[P1_JT][P1_KT] = {};
    for (size_t i = 0; i < r; ++i) {
        for (size_t c = 0; c < P1_JT; ++c) {
            const double xv = bx[c * ldbx + i];
            for (size_t kk = 0; kk < P1_KT; ++kk) {
                acc[c][kk] += bo[i * P1_KT + kk] * xv;
            }
        }
    }
#endif
    for (size_t c = 0; c < P1_JT; ++c) {
        for (size_t kk = 0; kk < kn; ++kk) {
            tcol[c][kk] += acc[c][kk];
        }
    }
}

/*
 *  Phase 1 over a column set: T[:, j] = Bo^T * Bx[:, j] for each selected j,
 *  an f32-in / f64-accumulate GEMM no BLAS offers (sgemm accumulates in f32,
 *  dgemm needs f64 inputs). Register-tiled over row panels (see p1_tile);
 *  fewer than P1_JT columns (e.g. a small DGKS retry) take the plain loop,
 *  where packing would not pay for itself: with i outermost it loads bo_row
 *  once per row and stores a T element on every FMA.
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
    if (nc < P1_JT) {
        for (size_t i = 0; i < n; ++i) {
            const float *bo_row = Bo + i * ldBo;
            for (size_t c = 0; c < nc; ++c) {
                const size_t j = sel_col<Indexed>(cols, c);
                axpy_m(T + j * ldT, bo_row, (double)Bx[i + j * ldBx], m);
            }
        }
        return;
    }

    const size_t nkt = (m + P1_KT - 1) / P1_KT;
    thread_local std::vector<double> bop, bxp;   // packed Bo panel, widened Bx panel
    bop.resize(nkt * P1_ROWS * P1_KT);
    bxp.resize(nc * P1_ROWS);
    for (size_t i0 = 0; i0 < n; i0 += P1_ROWS) {
        const size_t r = std::min(P1_ROWS, n - i0);
        for (size_t i = 0; i < r; ++i) {
            const float *row = Bo + (i0 + i) * ldBo;
            size_t k = 0;
            for (size_t t = 0; t < nkt; ++t) {
                double *dst = bop.data() + (t * P1_ROWS + i) * P1_KT;
                for (size_t kk = 0; kk < P1_KT; ++kk, ++k) {
                    dst[kk] = (k < m) ? (double)row[k] : 0.0;
                }
            }
        }
        for (size_t c = 0; c < nc; ++c) {
            const float *src = Bx + sel_col<Indexed>(cols, c) * ldBx + i0;
            double      *dst = bxp.data() + c * P1_ROWS;
            for (size_t i = 0; i < r; ++i) {
                dst[i] = (double)src[i];
            }
        }
        size_t c = 0;
        for (; c + P1_JT <= nc; c += P1_JT) {
            for (size_t t = 0; t < nkt; ++t) {
                const size_t k0 = t * P1_KT;
                double *tcol[P1_JT];
                for (size_t q = 0; q < P1_JT; ++q) {
                    tcol[q] = T + sel_col<Indexed>(cols, c + q) * ldT + k0;
                }
                p1_tile(r, bop.data() + t * P1_ROWS * P1_KT, bxp.data() + c * P1_ROWS,
                        P1_ROWS, tcol, std::min(P1_KT, m - k0));
            }
        }
        // Leftover columns (< P1_JT): axpy over the packed panel, one T column
        // at a time (still one T store per P1_KT-wide FMA, per row).
        for (; c < nc; ++c) {
            double       *tc = T + sel_col<Indexed>(cols, c) * ldT;
            const double *xv = bxp.data() + c * P1_ROWS;
            for (size_t t = 0; t < nkt; ++t) {
                const size_t  k0 = t * P1_KT;
                const size_t  kn = std::min(P1_KT, m - k0);
                const double *bo = bop.data() + t * P1_ROWS * P1_KT;
                double acc[P1_KT] = {};
                for (size_t i = 0; i < r; ++i) {
                    for (size_t kk = 0; kk < P1_KT; ++kk) {
                        acc[kk] += bo[i * P1_KT + kk] * xv[i];
                    }
                }
                for (size_t kk = 0; kk < kn; ++kk) {
                    tc[k0 + kk] += acc[kk];
                }
            }
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
    double *c, double &v_norm2, int threads)
{
    // Per-row-block partials of c = Boᵀv and of ||v - Bo c||², combined in
    // block order (row_blocks): thread-count independent.
    const size_t nb = row_block_count(n);
    thread_local std::vector<double> part;
    part.resize(nb * m + nb);
    double *pc = part.data();     // (not `part` inside the regions: thread_local)
    double *pn = part.data() + nb * m;
    row_blocks(n, threads, [&](size_t b, size_t i0, size_t i1) {
        double *cb = pc + b * m;
        std::fill_n(cb, m, 0.0);
        for (size_t i = i0; i < i1; ++i) {
            axpy_m(cb, Bo + i * ldBo, v[i], m);
        }
    });
    std::fill_n(c, m, 0.0);
    for (size_t b = 0; b < nb; ++b) {
        const double *cb = pc + b * m;
        for (size_t k = 0; k < m; ++k) {
            c[k] += cb[k];
        }
    }
    row_blocks(n, threads, [&](size_t b, size_t i0, size_t i1) {
        double s = 0.0;
        for (size_t i = i0; i < i1; ++i) {
            const double r = v[i] - dot_bo(Bo + i * ldBo, c, m);
            v[i] = r;
            s += r * r;
        }
        pn[b] = s;
    });
    v_norm2 = 0.0;
    for (size_t b = 0; b < nb; ++b) {
        v_norm2 += pn[b];
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
    std::atomic<long> *dgks_counter,
    int threads)
{
    std::vector<double> c(m);
    double v_norm2 = 0.0;
    const double proj_norm2 = cgs_project_col(n, m, v, Bo, ldBo, c.data(), v_norm2, threads);

    /*
     *  DGKS retry: re-orthogonalize once when the residual energy is small
     *  relative to what was projected out. (The retry's own projected energy
     *  is tiny by construction and discarded.)
     */
    if (v_norm2 > tol && v_norm2 * DGKS_GATE_RATIO_SQ < proj_norm2) {
        if (dgks_counter) {
            dgks_counter->fetch_add(1, std::memory_order_relaxed);
        }
        cgs_project_col(n, m, v, Bo, ldBo, c.data(), v_norm2, threads);
    }

    const double w = std::sqrt(v_norm2);
    if (w * w > tol) {
        row_blocks(n, threads, [&](size_t, size_t i0, size_t i1) {
            for (size_t i = i0; i < i1; ++i) {
                v[i] /= w;
            }
        });
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

void store_col(size_t n, size_t col, const double *v, float *Bo, size_t ldBo,
               const float *y, double *g, double &yb, int threads)
{
    const size_t m1 = col + 1;
    const size_t w  = m1 + 1;                 // g partial + yb partial per block
    const size_t nb = row_block_count(n);
    thread_local std::vector<double> part;
    part.resize(nb * w);
    double *pp = part.data();    // (not `part` inside the region: thread_local)
    row_blocks(n, threads, [&](size_t b, size_t i0, size_t i1) {
        double *pg = pp + b * w;
        std::fill_n(pg, w, 0.0);
        double s = 0.0;
        for (size_t i = i0; i < i1; ++i) {
            float *row = Bo + i * ldBo;
            const float f = (float)v[i];
            row[col] = f;
            axpy_m(pg, row, (double)f, m1);
            s += (double)f * (double)y[i];
        }
        pg[m1] = s;
    });
    std::fill_n(g, m1, 0.0);
    yb = 0.0;
    for (size_t b = 0; b < nb; ++b) {
        const double *pg = pp + b * w;
        for (size_t k = 0; k < m1; ++k) {
            g[k] += pg[k];
        }
        yb += pg[m1];
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

    explicit_dsse(n, m, fallback.data(), fallback.size(), Bo, ldBo, T, ldT,
                  Bx, ldBx, yres, tol, sb, dsse);
}

namespace {
template <bool Indexed>
void explicit_dsse_impl(size_t n, size_t m, const int *cols, size_t nc,
                        const float *Bo, size_t ldBo, double *T, size_t ldT,
                        float *Bx, size_t ldBx, const float *yres, double tol,
                        double *sb, double *dsse)
{
    project_out<Indexed>(n, m, cols, nc, Bo, ldBo, T, ldT, Bx, ldBx, sb);
    thread_local std::vector<int> retry;
    retry.clear();
    for (size_t c = 0; c < nc; ++c) {
        const size_t  j  = sel_col<Indexed>(cols, c);
        const double *tc = T + j * ldT;
        if (sb[j] > tol && sb[j] * DGKS_GATE_RATIO_SQ < dot_m(tc, tc, m)) {
            retry.push_back((int)j);
        }
    }
    if (!retry.empty()) {
        bo_t_bx<true>(n, m, retry.data(), retry.size(), Bo, ldBo, Bx, ldBx, T, ldT);
        project_out<true>(n, m, retry.data(), retry.size(), Bo, ldBo, T, ldT, Bx, ldBx, sb);
    }
    for (size_t c = 0; c < nc; ++c) {
        const size_t j = sel_col<Indexed>(cols, c);
        if (sb[j] > tol) {
            const double u = dot_widen(Bx + j * ldBx, yres, n);
            dsse[j] = u * u / (sb[j] + tol);
        }
        else {
            dsse[j] = 0.0;
        }
    }
}
} // namespace

void explicit_dsse(size_t n, size_t m, const int *cols, size_t nc,
                   const float *Bo, size_t ldBo, double *T, size_t ldT,
                   float *Bx, size_t ldBx, const float *yres, double tol,
                   double *sb, double *dsse)
{
    if (cols) {
        explicit_dsse_impl<true>(n, m, cols, nc, Bo, ldBo, T, ldT, Bx, ldBx, yres, tol, sb, dsse);
    }
    else {
        explicit_dsse_impl<false>(n, m, cols, nc, Bo, ldBo, T, ldT, Bx, ldBx, yres, tol, sb, dsse);
    }
}

void phase1(size_t n, size_t m, size_t nc, const float *Bo, size_t ldBo,
            const float *Bx, size_t ldBx, double *T, size_t ldT)
{
    bo_t_bx<false>(n, m, nullptr, nc, Bo, ldBo, Bx, ldBx, T, ldT);
}

void fill_stats(const float *b, const float *x, const float *y, float *bx, size_t n,
                double &nb, double &by)
{
    fill_col_stats(b, x, y, bx, n, nb, by);
}

void cross_rows(size_t n, const float *B, size_t ldB, const int *parents, size_t np,
                const float *x, const float *Bo, size_t ldBo, size_t r0, size_t r1,
                double *T, size_t ldT)
{
    constexpr size_t PR = 256;           // rows per panel
    const size_t nr = r1 - r0;
    if (nr == 0 || np == 0) {
        return;
    }
    thread_local std::vector<double> bp, v;
    bp.resize(nr * PR);
    v.resize(PR);
    for (size_t i0 = 0; i0 < n; i0 += PR) {
        const size_t r = std::min(PR, n - i0);
        for (size_t i = 0; i < r; ++i) {        // the new Bo columns, panel-contiguous
            const float *row = Bo + (i0 + i) * ldBo + r0;
            for (size_t kk = 0; kk < nr; ++kk) {
                bp[kk * PR + i] = (double)row[kk];
            }
        }
        for (size_t c = 0; c < np; ++c) {
            const size_t k  = (size_t)parents[c];
            const float *bk = B + k * ldB + i0;
            const float *xx = x + i0;
            for (size_t i = 0; i < r; ++i) {
                v[i] = (double)(bk[i] * xx[i]);  // the f32 product, as the fill forms it
            }
            double *tk = T + k * ldT + r0;
            for (size_t kk = 0; kk < nr; ++kk) {
                tk[kk] += dot_m(v.data(), bp.data() + kk * PR, r);
            }
        }
    }
}

} // namespace mars
