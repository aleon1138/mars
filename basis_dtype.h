#pragma once

namespace mars {

/*
 *  Storage element type of the basis buffers B / Bo / Bx (and the per-thread
 *  Bx scratch). Today this is f32; a future low-precision build flips it to a
 *  16-bit type (bf16) to halve the load volume of the bandwidth-bound forward
 *  pass. All arithmetic stays f32/f64 -- the 16-bit form is storage only.
 *
 *  Every read of a stored basis element goes through widen(), every store
 *  through narrow(), so the precision conversion lives in exactly two places:
 *
 *      widen  : storage -> f32 working precision  (decode; exact for bf16->f32)
 *      narrow : f32/f64 working value -> storage  (encode; round for ->bf16)
 *
 *  For basis_t == float both are the identity (float<->double round-trips are
 *  exact and the optimizer elides them), so wrapping a read/store changes no
 *  bits and no codegen. The wrappers exist so the bf16 build only has to change
 *  this header plus the SIMD load/store helpers, not every call site.
 *
 *  Note widen() returns float (working precision), not double: callers that
 *  feed an f64 accumulator promote the result implicitly (float * double ->
 *  double), which reproduces the existing `(double)elem * acc` pattern exactly;
 *  callers doing f32 elementwise products (the Bx fill / normalize, the append
 *  basis build) keep their original float rounding sequence.
 */
using basis_t = float;

inline float   widen (basis_t v)
{
    return v;
}
inline basis_t narrow(double  v)
{
    return (basis_t)v;
}

} // namespace mars
