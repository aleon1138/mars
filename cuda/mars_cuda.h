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
 *  First pass scope: a GPU drop-in for mars::orthonormalize() only, behind the
 *  `cuda=True` flag on MarsAlgo.eval(). The CPU kernel in kernels.cc stays the
 *  correctness oracle; this path must match it within the f32 storage floor
 *  (~1e-5), which means the T = Boᵀ·Bx contraction MUST accumulate in f64
 *  (cublasDgemm) -- f32 accumulation corrupts the near-zero projection entries.
 */

#include <atomic>
#include <cstddef>  // size_t

namespace mars {
namespace cuda {

/*
 *  Opaque per-MarsAlgo GPU context. Owns a cuBLAS handle + stream, resident
 *  device copies of B (f32) and Bo (widened to f64), and per-call scratch sized
 *  for up to `max_terms` columns. One context drives one GPU stream's worth of
 *  resident state; it is NOT thread-safe -- the binding serializes the cuda
 *  path to a single host thread.
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
 *  GPU counterpart of mars::orthonormalize(). For each j in [0, p):
 *      Bx[:,j]  = B[:, mask[j]] .* x                    (f32 multiply, f32 store)
 *      Bx[:,j] -= Bo * (Boᵀ * Bx[:,j])                  (f64 GEMM, f32 round)
 *      s        = ||Bx[:,j]||^2  over the rounded f32
 *      Bx[:,j] *= (s > tol) ? 1/sqrt(s + tol) : 0
 *  with one DGKS retry per column whose residual energy is < 1/9 of its
 *  projection energy (matching the CPU gate). Requires context_sync_basis to
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
 *             computes ybx[j] = Σ_i Bx[i,j]·y[i] (f64 accumulation, matching
 *             mars::dot_widen) and copies back just these p doubles -- so the
 *             n×p Bx matrix never crosses PCIe. Requires context_set_target().
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
                    std::atomic<long> *dgks_counter = nullptr);

/*
 *  Batched linear_only orthonormalize: process P candidate columns drawn from a
 *  block of `nb` X-columns (which all share the resident Bo) in ONE set of GEMMs,
 *  returning ybx[j] = scale·Σ Bx[:,j]·y for each. This amortizes the GEMM /
 *  launch / transfer overhead that dominates the one-eval-at-a-time path. The
 *  matrix Bx is never returned (linear_only). Requires context_set_target() and
 *  context_sync_basis() up to m first.
 *
 *      m         : live orthonormal basis count (resident Bo columns).
 *      P         : number of candidate (output) columns across the block.
 *      nb        : number of distinct X-columns in the block (<= P).
 *      Xblock    : (n, nb) col-major f32 host pointer -- the raw X columns; col
 *                  stride ldX. Scaled on-device by s_block (so x = X·s is
 *                  f32-rounded before multiplying by B, matching the per-eval path).
 *      s_block   : (nb) per-column 1/rms scale.
 *      src_xcol  : (P) int32 -- output col -> block-local X column in [0, nb).
 *      src_basis : (P) int32 -- output col -> B/Bo column in [0, m).
 *      ybx       : (P) f64 host pointer [output].
 *      dgks_counter (optional): as in orthonormalize().
 *
 *  P must be <= the context's batch capacity (budget / ~16n bytes, never below
 *  max_terms; tune with MARS_CUDA_BLOCK_GB). The caller blocks the X-columns to
 *  respect that cap.
 */
void orthonormalize_batch(Context *ctx, size_t m, size_t P, size_t nb,
                          const float *Xblock, size_t ldX, const float *s_block,
                          const int *src_xcol, const int *src_basis,
                          double *ybx, std::atomic<long> *dgks_counter = nullptr);

} // namespace cuda
} // namespace mars
