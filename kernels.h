#pragma once

#include <atomic>

namespace mars {

/*
 *  DGKS re-orthogonalization gate. Trigger when residual energy is less
 *  than 1/9 of projection energy (eta^2 = 0.1, eta ~= 0.316). Stricter
 *  than the textbook eta = 1/sqrt(2) (which corresponds to ratio = 1.0):
 *  Bx is per-call scratch and we don't need bit-perfect orthogonality of
 *  it, only protection against catastrophic cancellation. The looser
 *  gate fires on numerically-benign cases and perturbs results within
 *  the f32->f64 noise floor for no real accuracy gain.
 */
constexpr double DGKS_GATE_RATIO_SQ = 9.0;

/*
 *  Interact a candidate regressor `x` with the basis columns selected by
 *  `mask`, orthonormalize against `Bo`, and normalize each column. For each
 *  j in [0, p):
 *      Bx[:,j]  = B[:, mask[j]] .* x                       // fill (f32 * f32 → f32 store)
 *      Bx[:,j] -= Bo * (Bo^T * Bx[:,j])                    // project out Bo's span (f64 arith)
 *      s        = ||Bx[:,j]||^2                            // f64 accumulator over post-store f32
 *      Bx[:,j] *= (s > tol) ? 1/sqrt(s + tol) : 0          // normalize, zero degenerate
 *
 *  Bo and Bx are both stored as f32 to halve the largest buffers; all
 *  arithmetic (projection, dot products, norms) stays in f64 -- Bo and Bx
 *  entries are upcast at the load. The cancellation in (Bx -= Bo*Bo^T*Bx)
 *  is bounded by the DGKS retry, so the rounded f32 storage keeps
 *  O(eps_f32) orthogonality against Bo -- see DGKS_GATE_RATIO_SQ below.
 *
 *  Layouts (caller's responsibility):
 *      B     : (n, *) col-major float;  col stride ldB.   Only columns mask[j] are read.
 *      x     : (n)    float
 *      mask  : (p)    int32, indexes into columns of B
 *      Bo    : (n, m) row-major float;  row stride ldBo
 *      Bx    : (n, p) col-major float;  col stride ldBx       [output]
 *      T     : (m, p) col-major double; col stride ldT        [workspace; overwritten]
 *      s_buf : (p)    double                                  [workspace; overwritten]
 *      tol   : tolerance for treating a column as degenerate;
 *              should be ~O(n * eps_f32^2) for the f32 storage floor.
 *
 *  Uses AVX2 + FMA in the inner loops when available, scalar fallback otherwise.
 *  No heap allocations.
 *
 *  dgks_counter (optional): if non-null, atomically incremented each time the
 *  DGKS re-orthogonalization branch fires for a column. Lets callers measure
 *  numerical cancellation frequency without coupling to a global.
 */
void orthonormalize(
    int n, int m, int p,
    const float  *B,    int ldB,
    const float  *x,
    const int    *mask,
    const float  *Bo,   int ldBo,
    float        *Bx,   int ldBx,
    double       *T,    int ldT,
    double       *s_buf,
    double tol,
    std::atomic<long> *dgks_counter = nullptr);

/*
 *  Dot product of two contiguous f32 vectors, accumulated in f64:
 *
 *      returns  sum_{i<n} (double)a[i] * (double)b[i]
 *
 *  Both inputs are upcast at the load (cvtps_pd) and summed in f64, so the
 *  result carries only the ~eps_f32 floor of the f32 inputs -- the summation
 *  adds nothing on top, unlike an f32 accumulation which grows ~sqrt(n)*eps_f32.
 *  Two accumulators break the FMA dependency chain (one chain would cap at
 *  ~1/4 of FMA throughput). AVX2+FMA when available, scalar fallback otherwise.
 *
 *      a, b : (n) contiguous float
 */
double dot_widen(const float *a, const float *b, int n);

} // namespace mars
