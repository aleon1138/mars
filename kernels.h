#pragma once

#include <atomic>
#include <cstddef>  // size_t

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
 *  Relative floor on the hinge-sweep denominator den = ||h+||^2 - ||proj h+||^2,
 *  the energy of the candidate hinge outside span(Bo, Bx). For a hinge inside
 *  the span, den is the difference of two O(||h+||^2) sums and comes out as
 *  noise of either sign at ~eps_f32/sqrt(n) of ||h+||^2 (the orthonormality
 *  defect of the f32 Bo); the numerator uw is noise of the same origin, so
 *  uw^2/den can score anywhere up to ||y_perp||^2 -- observed: in-span cuts at
 *  den/||h+||^2 ~ 1e-11 scoring 1e-2, above every legitimate cut. The absolute
 *  _tol (which grows with n while this floor shrinks) cannot catch it, so
 *  require that at least this fraction of the hinge's energy be new. 8*eps_f32
 *  ~ 1e-6 sits ~4 decades above the floor and well below any cut min_span
 *  admits; below it the term would be a near-duplicate with an O(1/sqrt(den))
 *  coefficient whose gain f32 cannot resolve anyway. Shared by the CPU sweep
 *  (marsalgo.cc) and the CUDA sweep (cuda/hinge_sweep.cu).
 */
constexpr double HINGE_DEN_REL = 8.0 * 1.1920928955078125e-07;  // 8 * FLT_EPSILON

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
 *  O(eps_f32) orthogonality against Bo -- see DGKS_GATE_RATIO_SQ above.
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
 *  The DGKS retry re-runs the flagged columns as one batch (two more sweeps
 *  over Bo, not two per column); its index list is the only heap allocation,
 *  thread-local and grown on demand.
 *
 *  dgks_counter (optional): if non-null, atomically incremented each time the
 *  DGKS re-orthogonalization branch fires for a column. Lets callers measure
 *  numerical cancellation frequency without coupling to a global.
 */
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
    std::atomic<long> *dgks_counter = nullptr);

/*
 *  Single-column Gram-Schmidt: orthonormalize one new basis column against the
 *  m orthonormal columns of `Bo`, in place. This is the per-append counterpart
 *  of orthonormalize() (which batches p candidate columns for eval()) and uses
 *  the same scheme: one classical Gram-Schmidt pass (c = Bo^T v, v -= Bo c)
 *  plus one DGKS retry when the residual energy is small relative to what was
 *  projected out (see DGKS_GATE_RATIO_SQ). With Bo orthonormal to f32 storage
 *  precision, classical and modified GS agree to that same floor; classical GS
 *  is chosen because both halves stream Bo row by row (two sequential sweeps),
 *  whereas modified GS walks Bo column by column -- a strided access that costs
 *  one cache line per row per column, m times the traffic. On return:
 *      v        <- v_perp / ||v_perp||   if ||v_perp||^2 > tol  (unit residual)
 *               <- v_perp (unchanged)    otherwise              (degenerate)
 *      returns     ||v_perp||            (pre-normalization; caller accepts the
 *                                         column iff w*w > tol and stores v)
 *
 *  All arithmetic is f64; Bo is f32 storage, upcast on the load. Allocates an
 *  m-length f64 workspace (cold path: once per accepted basis).
 *
 *  Layouts:
 *      v    : (n)    double, in/out -- the raw new column; residual on return
 *      Bo   : (n, m) row-major float; row stride ldBo
 *      tol  : degeneracy threshold (same scale as orthonormalize()'s tol)
 *
 *  dgks_counter (optional): atomically incremented if the DGKS retry fires.
 */
double orthonormalize_col(
    size_t n, size_t m,
    double       *v,
    const float  *Bo,   size_t ldBo,
    double tol,
    std::atomic<long> *dgks_counter = nullptr);

/*
 *  Fallback threshold of the closed-form linear delta-SSE (linear_dsse()): a
 *  candidate whose residual carries less than 1/LINEAR_KAPPA2_MAX of its energy
 *  (kappa^2 = ||Bx||^2 / den above this) is scored by explicit projection
 *  instead. The closed form subtracts O(||Bx||^2) f64 sums to get den, so its
 *  relative error grows as ~eps_f64*sqrt(n)*kappa^2 (the sequential T and
 *  ||Bx||^2 sums); 1e6 keeps that under ~1e-7 at n=5e5. Exactly in-span
 *  candidates (kappa = inf) always take the explicit path, which zeroes them.
 */
constexpr double LINEAR_KAPPA2_MAX = 1e6;

/*
 *  Linear delta-SSE of p candidate columns, in closed form. For each j, with
 *  Bx_j = B[:, mask[j]] .* x (f32 product, stored f32 -- the same column
 *  orthonormalize() forms and append() would store) and T_j = Bo^T Bx_j:
 *
 *      den = ||Bx_j||^2 - T_j'T_j + T_j'E T_j          (residual energy)
 *      u   = <Bx_j, y>  - T_j'ybo + T_j'E ybo          (residual . y)
 *      dsse[j] = (den > tol) ? u^2 / (den + tol) : 0
 *
 *  This is the projection onto the complement of span(Bo) with the exact Gram
 *  BoᵀBo = I + E taken to first order in E (E ~ eps_f32, so the dropped
 *  term is ~||E||^2*kappa^2, ~5e-9 at the fallback gate): it needs only
 *  Phase 1 (T), not the
 *  projection, the DGKS retry or the normalize, and never rounds the residual
 *  to f32. Columns whose kappa^2 = ||Bx_j||^2 / den exceeds LINEAR_KAPPA2_MAX
 *  fall back to the explicit path: Bx_j -= Bo T_j, a DGKS retry when gated
 *  (see orthonormalize()), then dsse[j] = <r, yres>^2 / (s + tol), with f64
 *  throughout and no f32 normalize scale.
 *
 *  Layouts (caller's responsibility):
 *      B, x, mask, Bo   : as in orthonormalize()
 *      y     : (n) f32 target;  yres : (n) f32 residual target y - Bo*ybo
 *      ybo   : (m) Bo^T y;      E : (m, m) symmetric, col-major, col stride ldE
 *      Eybo  : (m) E * ybo
 *      Bx    : (n, p) col-major float, col stride ldBx   [workspace]
 *      T     : (m, p) col-major double, col stride ldT   [workspace]
 *      work  : (3p + m) double                           [workspace]
 *      dsse  : (p) double                                [output]
 *
 *  fallback_counter (optional): incremented by the number of explicit-path
 *  columns.
 */
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
    std::atomic<long> *fallback_counter = nullptr);

/*
 *  One column of the Gram of Bo: g[k] = sum_i Bo[i,k] * Bo[i,col] for k < m1
 *  (col < m1), f64-accumulated from the f32 storage. append() uses it to extend
 *  E = BoᵀBo - I. One streaming sweep over the first m1 entries of each row.
 */
void gram_col(size_t n, size_t m1, const float *Bo, size_t ldBo, size_t col, double *g);

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
double dot_widen(const float *a, const float *b, size_t n);

} // namespace mars
