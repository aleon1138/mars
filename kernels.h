/*
 *  Hand-rolled numerical kernels that replace specific Eigen expressions on
 *  the eval() hot path. Eigen produces correct code but allocates temporaries
 *  for matmul-shaped expressions; in a 64-thread inner loop those allocations
 *  thrash glibc's per-thread arenas and serialize page faults behind the
 *  kernel's mmap_lock. The kernels here take raw pointers + strides (BLAS
 *  style) and do not allocate.
 */
#pragma once

namespace mars {

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
 *      B    : (n, *) col-major float;  col stride ldB.   Only columns mask[j] are read.
 *      x    : (n)    float
 *      mask : (p)    int32, indexes into columns of B
 *      Bo   : (n, m) row-major double; row stride ldBo
 *      Bx   : (n, p) col-major double; col stride ldBx       [output]
 *      T    : (m, p) col-major double; col stride ldT        [workspace; overwritten]
 *      tol  : tolerance for treating a column as degenerate
 *
 *  Uses AVX2 + FMA in the inner loops when available, scalar fallback otherwise.
 *  No heap allocations.
 */
void orthonormalize(
    int n, int m, int p,
    const float  *B,    int ldB,
    const float  *x,
    const int    *mask,
    const double *Bo,   int ldBo,
    double       *Bx,   int ldBx,
    double       *T,    int ldT,
    double tol);

} // namespace mars
