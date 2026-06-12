#pragma once

#include <cstdint>
#include <cstring>  // memcpy

namespace mars {

/*
 *  Storage element types for the basis buffers B / Bo / Bx (and the per-thread
 *  Bx scratch). The buffers can be stored as f32 (full precision, the default)
 *  or bf16 (half the bytes, to cut the load volume of the bandwidth-bound
 *  forward pass). bf16 is *storage only* -- all arithmetic stays f32/f64.
 *
 *  Every read of a stored element goes through widen() (decode to f32 working
 *  precision), every store through store() (encode the f32/f64 working value).
 *  Both dispatch on the element type, so templated kernels written on a storage
 *  type `BT` need no explicit precision handling beyond these two calls:
 *
 *      widen(BT)        -> float   decode (exact for bf16->f32)
 *      store(BT&, double)           encode (round-to-nearest-even for ->bf16)
 *
 *  widen() returns float (working precision), not double: callers feeding an f64
 *  accumulator promote implicitly (float * double -> double), reproducing the
 *  old `(double)elem * acc` pattern exactly; callers doing f32 elementwise
 *  products keep their original float rounding sequence. For BT == float both
 *  ops are the identity (float<->double round-trips are exact and elided), so a
 *  float-typed kernel is bit-for-bit what it was before the seam.
 *
 *  bf16 is the truncated top 16 bits of an IEEE-754 f32 (1 sign, 8 exponent, 7
 *  mantissa) -- same dynamic range as f32, ~2-3 decimal digits of precision.
 */
struct bf16 {
    uint16_t bits;
};

// --- decode (widen): stored element -> f32 working precision ----------------
inline float widen(float v)
{
    return v;
}
inline float widen(bf16 v)
{
    uint32_t u = (uint32_t)v.bits << 16;
    float f;
    std::memcpy(&f, &u, sizeof(f));
    return f;
}

// --- encode (store): f32/f64 working value -> stored element -----------------
inline void store(float &dst, double v)
{
    dst = (float)v;
}
inline void store(bf16 &dst, double v)
{
    const float f = (float)v;
    if (f != f) {            // NaN -> canonical quiet bf16 NaN (preserve NaN-ness)
        dst.bits = 0x7FC0;
        return;
    }
    uint32_t u;
    std::memcpy(&u, &f, sizeof(u));
    // round-to-nearest-even on the discarded low 16 bits
    const uint32_t bias = ((u >> 16) & 1u) + 0x7FFFu;
    u += bias;
    dst.bits = (uint16_t)(u >> 16);
}

} // namespace mars
