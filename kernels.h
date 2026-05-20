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
 *      Bx[:,j]  = (double)(B[:, mask[j]] .* x)             // fill (f32 * f32 → f64)
 *      Bx[:,j] -= Bo * (Bo^T * Bx[:,j])                    // project out Bo's span
 *      s        = ||Bx[:,j]||^2
 *      Bx[:,j] *= (s > tol) ? 1/sqrt(s + tol) : 0          // normalize, zero degenerate
 *
 *  Layouts (caller's responsibility):
 *      B     : (n, *) col-major float;  col stride ldB.   Only columns mask[j] are read.
 *      x     : (n)    float
 *      mask  : (p)    int32, indexes into columns of B
 *      Bo    : (n, m) row-major double; row stride ldBo
 *      Bx    : (n, p) col-major double; col stride ldBx       [output]
 *      T     : (m, p) col-major double; col stride ldT        [workspace; overwritten]
 *      s_buf : (p)    double                                  [workspace; overwritten]
 *      tol   : tolerance for treating a column as degenerate
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
    const double *Bo,   int ldBo,
    double       *Bx,   int ldBx,
    double       *T,    int ldT,
    double       *s_buf,
    double tol,
    std::atomic<long> *dgks_counter = nullptr);

} // namespace mars
