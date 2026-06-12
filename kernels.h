#pragma once

#include <atomic>
#include <cstddef>  // size_t
#include "basis_dtype.h"  // bf16, widen, store

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
 *  B/Bo/Bx are stored as `BT` (f32 or bf16) to shrink the largest buffers; all
 *  arithmetic (projection, dot products, norms) stays in f64 -- entries are
 *  widened at the load and rounded back on store. The cancellation in
 *  (Bx -= Bo*Bo^T*Bx) is bounded by the DGKS retry, so the rounded storage keeps
 *  O(eps_BT) orthogonality against Bo -- see DGKS_GATE_RATIO_SQ below.
 *
 *  Layouts (caller's responsibility):
 *      B     : (n, *) col-major BT;     col stride ldB.   Only columns mask[j] are read.
 *      x     : (n)    float             (the f32 normalized candidate column)
 *      mask  : (p)    int32, indexes into columns of B
 *      Bo    : (n, m) row-major BT;     row stride ldBo
 *      Bx    : (n, p) col-major BT;     col stride ldBx       [output]
 *      T     : (m, p) col-major double; col stride ldT        [workspace; overwritten]
 *      s_buf : (p)    double                                  [workspace; overwritten]
 *      tol   : tolerance for treating a column as degenerate;
 *              should be ~O(n * eps_BT^2) for the storage floor.
 *
 *  Uses AVX2 + FMA in the inner loops for the f32 storage path; bf16 uses the
 *  scalar fallback for now (the SIMD bf16 widen-load comes later). No heap
 *  allocations.
 *
 *  dgks_counter (optional): if non-null, atomically incremented each time the
 *  DGKS re-orthogonalization branch fires for a column. Lets callers measure
 *  numerical cancellation frequency without coupling to a global.
 */
template <class BT>
void orthonormalize(
    size_t n, size_t m, size_t p,
    const BT  *B,    size_t ldB,
    const float *x,
    const int   *mask,
    const BT  *Bo,   size_t ldBo,
    BT        *Bx,   size_t ldBx,
    double    *T,    size_t ldT,
    double    *s_buf,
    double tol,
    std::atomic<long> *dgks_counter = nullptr);

/*
 *  Single-column Gram-Schmidt: orthonormalize one new basis column against the
 *  m existing orthonormal columns of `Bo`, in place. This is the per-append
 *  counterpart of orthonormalize() (which batches p candidate columns for
 *  eval()); it mirrors the math in MarsAlgo::append().
 *
 *  Uses MODIFIED Gram-Schmidt (each projection updates v before the next is
 *  computed) plus one DGKS retry when the residual energy is small relative to
 *  what was projected out (see DGKS_GATE_RATIO_SQ). With Bo orthonormal, MGS
 *  and classical GS coincide up to O(eps); MGS is chosen to match append()
 *  bit-for-bit. On return:
 *      v        <- v_perp / ||v_perp||   if ||v_perp||^2 > tol  (unit residual)
 *               <- v_perp (unchanged)    otherwise              (degenerate)
 *      returns     ||v_perp||            (pre-normalization; caller accepts the
 *                                         column iff w*w > tol and stores v)
 *
 *  All arithmetic is f64; Bo is f32 storage, upcast on the load. Scalar: this
 *  runs once per accepted basis (cold), and the per-column Bo access is strided
 *  (Bo is row-major), so there is nothing to vectorize.
 *
 *  Layouts:
 *      v    : (n)    double, in/out -- the raw new column; residual on return
 *      Bo   : (n, m) row-major float; row stride ldBo
 *      tol  : degeneracy threshold (same scale as orthonormalize()'s tol)
 *
 *  dgks_counter (optional): atomically incremented if the DGKS retry fires.
 */
template <class BT>
double orthonormalize_col(
    size_t n, size_t m,
    double   *v,
    const BT *Bo,   size_t ldBo,
    double tol,
    std::atomic<long> *dgks_counter = nullptr);

/*
 *  Dot product of a contiguous basis column with the contiguous f32 target,
 *  accumulated in f64:
 *
 *      returns  sum_{i<n} (double)a[i] * (double)b[i]
 *
 *  Both inputs are upcast at the load (cvtps_pd) and summed in f64, so the
 *  result carries only the ~eps_f32 floor of the inputs -- the summation
 *  adds nothing on top, unlike an f32 accumulation which grows ~sqrt(n)*eps_f32.
 *  Two accumulators break the FMA dependency chain (one chain would cap at
 *  ~1/4 of FMA throughput). AVX2+FMA when available, scalar fallback otherwise.
 *
 *  The operand types are split so a future bf16 basis build can widen `a`
 *  (the basis column) and `b` (the f32 target) independently at the load.
 *
 *      a : (n) contiguous basis column
 *      b : (n) contiguous f32 target
 */
template <class BT>
double dot_widen(const BT *a, const float *b, size_t n);

} // namespace mars
