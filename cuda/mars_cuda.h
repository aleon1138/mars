#pragma once

/*
 *  Host-facing API for the optional CUDA acceleration of the MARS forward pass.
 *
 *  This header is compiled by the *host* compiler (it is included by
 *  marsalgo.cc / tests), so it must stay free of any CUDA types or headers --
 *  only plain C++ types appear below. The implementation, and the full
 *  definition of `Context` (cuBLAS handle, stream, device buffers), live in
 *  cuda/orthonormalize.cu and are opaque here (PIMPL across the static-lib
 *  boundary).
 *
 *  Scope: a GPU drop-in for mars::orthonormalize() (the linear_only regime),
 *  behind the `cuda=True` flag on MarsAlgo.eval(). The CPU kernel in kernels.cc
 *  stays the correctness oracle; this path must match it within the f32
 *  storage floor (~1e-5). The T = Boᵀ·Bx contraction is therefore accumulated
 *  in f64 -- as blocked f32 GEMMs summed in f64 across K-chunks by default, or
 *  natively with MARS_CUDA_KCHUNK=0 -- because a single f32 accumulation
 *  corrupts the near-zero projection entries (see orthonormalize.cu).
 */

#include <atomic>
#include <cstddef>  // size_t

namespace mars {
namespace cuda {

/*
 *  Opaque per-MarsAlgo GPU context. Owns a cuBLAS handle + stream, resident
 *  device copies of B (f32) and Bo (f32 for the blocked GEMMs, f64 for the
 *  DGKS retry), the scaled candidate X-columns, and per-call scratch sized for
 *  the batch capacity. One context drives one GPU stream's worth of resident
 *  state; it is NOT thread-safe -- the binding serializes the cuda path to a
 *  single host thread.
 */
struct Context;

/*
 *  Allocate a context for an n-row problem with up to `max_terms` basis columns
 *  and degeneracy threshold `tol` (the same _tol MarsAlgo uses). Device buffers
 *  are sized for the worst case (p == max_terms) up front, so no per-call
 *  reallocation occurs. Throws std::runtime_error on any CUDA/cuBLAS failure
 *  (e.g. out-of-memory). Create lazily on first cuda eval so CPU-only fits never
 *  touch the GPU.
 */
Context *context_create(size_t n, size_t max_terms, double tol);

/*
 *  Free all device memory and the cuBLAS handle/stream. Safe on nullptr.
 */
void context_destroy(Context *ctx);

/*
 *  Candidate-column capacity of the batched scratch (>= max_terms). The caller
 *  blocks X-columns so that the total candidate columns per orthonormalize_batch
 *  call does not exceed this.
 */
size_t context_batch_capacity(Context *ctx);

/*
 *  Ensure the device holds columns [0, m) of the current basis. B/Bo columns
 *  are append-only over a fit, so this uploads only the columns appended since
 *  the last call (tracked internally by count) -- cheap to call at the top of
 *  every cuda eval.
 *
 *      B    : (n, *) col-major f32; col stride ldB (== n in practice).
 *      Bo   : (n, m) row-major f32; row stride ldBo (the live bo_stride; may
 *             have grown via restride since the last call -- pass it fresh).
 *
 *  The device keeps B as f32 (col-major) and Bo widened to f64 (col-major), in
 *  its own layout decoupled from the host's ldBo, so a host restride never
 *  invalidates already-uploaded device columns.
 */
void context_sync_basis(Context *ctx, size_t m,
                        const float *B,  size_t ldB,
                        const float *Bo, size_t ldBo);

/*
 *  Upload the (normalized) target `y` (n f32) to the device, once. Idempotent:
 *  re-uploads only when the pointer changes (y is fixed for a fit). Required
 *  before requesting the `ybx` output from orthonormalize().
 */
void context_set_target(Context *ctx, const float *y);

/*
 *  Upload the residual target y_res = y - Bo·ybo (n f32), the vector the
 *  `ybx` outputs of orthonormalize()/orthonormalize_batch() are dotted with
 *  once set (until then they use the target y). y_res changes with every
 *  append, so it is re-uploaded whenever `tag` (the caller's basis count)
 *  changes. Scoring against y_res instead of y keeps the f32 orthogonality
 *  defect of Bo out of the linear ΔSSE at first order (see MarsAlgo::eval()).
 *  The hinge sweep keeps using y.
 */
void context_set_residual(Context *ctx, const float *yres, size_t tag);

/*
 *  Ensure the given X-columns are resident on the device as the *scaled*
 *  candidates x_c = X[:,c]·s[c] (f32). Lazy and idempotent: a column is uploaded
 *  and scaled only the first time it is requested (the scaled candidate is fixed
 *  for a fit), so over a whole fit each used column transfers exactly once
 *  instead of every epoch. Backs the batched path, which indexes these by global
 *  column. The resident buffer is sized (n, p_x) on the first call.
 *
 *      p_x  : total number of X-columns (sizes the resident buffer once).
 *      X    : (n, p_x) col-major f32 host pointer; col stride ldX.
 *      s    : (p_x) per-column 1/rms scale.
 *      cols : (ncols) int32 global column indices to ensure resident.
 */
void context_sync_xcols(Context *ctx, size_t p_x,
                        const float *X, size_t ldX, const float *s,
                        const int *cols, size_t ncols);

/*
 *  GPU counterpart of mars::orthonormalize(). For each j in [0, p):
 *      Bx[:,j]  = B[:, mask[j]] .* x                    (f32 multiply, f32 store)
 *      Bx[:,j] -= Bo * (Boᵀ * Bx[:,j])                  (f64-accumulated T, f32 round)
 *      s        = ||Bx[:,j]||^2  over the rounded f32
 *      Bx[:,j] *= (s > tol) ? 1/sqrt(s + tol) : 0
 *  with one f64 DGKS retry per column whose residual energy is < 1/9 of its
 *  projection energy (matching the CPU gate). The retry restarts from the f64
 *  B·x, not from the f32-projected residual: the f32 projection leaves
 *  out-of-span noise ~eps_f32·√m·‖T‖ that no further projection can remove. Requires context_sync_basis to
 *  have brought columns [0, m) of B/Bo onto the device first.
 *
 *      m    : live orthonormal basis count (columns of resident Bo).
 *      p    : number of candidate columns (== length of mask).
 *      x    : (n) f32 host pointer -- the normalized candidate column.
 *      mask : (p) int32 host pointer -- indexes into columns of B.
 *      Bx   : (n, p) col-major f32 host pointer; col stride ldBx [output].
 *             Pass nullptr to skip the (large) device->host copy -- e.g. in the
 *             linear_only path, where only `ybx` is needed.
 *      ybx  : (p) f64 host pointer [output, optional]. If non-null, the device
 *             computes ybx[j] = Σ_i Bx[i,j]·t[i] (f64 accumulation, matching
 *             mars::dot_widen) and copies back just these p doubles -- so the
 *             n×p Bx matrix never crosses PCIe. t is the residual target if
 *             context_set_residual() was called, else y; requires one of them.
 *      dgks_counter (optional): host atomic, incremented per column the retry
 *             fires on (nullptr in production; used by tests).
 *
 *  The T = Boᵀ·Bx workspace and per-column norms stay on the device. The call
 *  is synchronous: on return, the requested outputs are fully populated. Throws
 *  std::runtime_error on any CUDA/cuBLAS failure.
 */
void orthonormalize(Context *ctx, size_t m, size_t p,
                    const float *x, const int *mask,
                    float *Bx, size_t ldBx,
                    double *ybx = nullptr,
                    std::atomic<long> *dgks_counter = nullptr,
                    bool normalize_resident = false);

/*
 *  GPU hinge sweep for one X column -- the linear_only=false half of
 *  MarsAlgo::eval(). Must directly follow an orthonormalize() call for the
 *  same (m, p, x, mask) made with normalize_resident=true: it reads the
 *  normalized candidate columns Bx and their target dots ybx from the device
 *  scratch that call left behind, plus the resident B / Bo / y.
 *
 *  For each candidate j in [0, p) with parent basis b = B[:, mask[j]], it
 *  evaluates the positive hinge h+ = b .* max(x - h, 0) at every cut h on the
 *  min_span grid of the descending sort of `xraw` (sorted positions head+1,
 *  head+1+min_span, ... below n-end_span, with head = end_span), scoring
 *      sse = uw^2 / (den + tol),  den = ||h+||^2 - ||proj_{Bo,Bx[:,j]} h+||^2,
 *      uw  = <proj h+, proj y> - <h+, y>,
 *  gated by den > max(tol, HINGE_DEN_REL*||h+||^2), and keeps the best cut
 *  (first one on ties), stopping a column once every nonzero row of its
 *  parent lies above the cut -- the same recurrence, gate and support rule as
 *  the CPU sweep in marsalgo.cc. The recurrence runs in f64 as a two-pass
 *  chunked scan over the sorted rows (see hinge_sweep.cu); results agree with
 *  the CPU to f64 reassociation error, not bit-exactly.
 *
 *      xraw      : (n) f32 host pointer -- the RAW X column (sort keys and the
 *                  reported cut values; the scaled candidate is already resident).
 *      ybo       : (m) f64 host pointer -- Boᵀ·y.
 *      hinge_sse : (p) f64 [output] best hinge gain beyond the linear term, 0 if none.
 *      hinge_idx : (p) int [output] sorted position of the best cut, -1 if none.
 *      hinge_cut : (p) f64 [output] xraw at the best cut, NaN if none.
 *
 *  Synchronous. Throws std::runtime_error on any CUDA failure.
 */
void hinge_sweep(Context *ctx, size_t m, size_t p,
                 const float *xraw, const double *ybo,
                 int min_span, int end_span,
                 double *hinge_sse, int *hinge_idx, double *hinge_cut);

/*
 *  Batched linear_only orthonormalize: process P candidate columns (which all
 *  share the resident Bo) in ONE set of GEMMs, returning ybx[j] = scale·Σ
 *  Bx[:,j]·y for each. This amortizes the GEMM / launch / transfer overhead that
 *  dominates the one-eval-at-a-time path. The matrix Bx is never returned
 *  (linear_only). Requires context_set_target(), context_sync_basis() up to m,
 *  and context_sync_xcols() for every X-column referenced by src_xcol.
 *
 *      m         : live orthonormal basis count (resident Bo columns).
 *      P         : number of candidate (output) columns in this block.
 *      src_xcol  : (P) int32 -- output col -> global X column (into the resident
 *                  scaled candidates from context_sync_xcols).
 *      src_basis : (P) int32 -- output col -> B/Bo column in [0, m).
 *      ybx       : (P) f64 host pointer [output].
 *      dgks_counter (optional): as in orthonormalize().
 *
 *  P must be <= the context's batch capacity (context_batch_capacity(); tune with
 *  MARS_CUDA_BLOCK_GB). The caller blocks the candidate columns to respect it.
 */
void orthonormalize_batch(Context *ctx, size_t m, size_t P,
                          const int *src_xcol, const int *src_basis,
                          double *ybx, std::atomic<long> *dgks_counter = nullptr);

} // namespace cuda
} // namespace mars
