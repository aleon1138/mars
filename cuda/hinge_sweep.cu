/*
 *  CUDA hinge sweep -- the linear_only=false half of MarsAlgo::eval(). See
 *  cuda/mars_cuda.h for the host contract and the sweep in marsalgo.cc for the
 *  CPU oracle this must match.
 *
 *  The CPU sweep walks the rows in descending order of x and, per candidate
 *  column j, carries a recurrence over sorted position i:
 *
 *      g[c] += b_i * Bo[k[i], c]                 (c < m; c == m uses Bx[k[i], j])
 *      f[c] += d_i * g[c]                        d_i = x[k[i-1]] - x[k[i]]
 *
 *  so that at cut i, f is the projection of the positive hinge h+ onto
 *  span(Bo, Bx[:,j]) and the score needs ||f||^2 and <f, [ybo; ybx_j]>, plus
 *  five scalars of the same shape (b2, vb as g-type sums over b*b and b*y;
 *  bd, w as the matching f-type sums; and ||h+||^2 driven by bd and b2). The
 *  recurrence is linear, so it chunks: for a chunk of sorted rows [a, b) run
 *  from a zero state, the true state at any i in the chunk is the local state
 *  plus the carry-in advanced by the same recurrence. The carry-in is the
 *  state the CPU holds when it is about to process row a (f after step a-1,
 *  g = sum over rows t < a), and the carry-out is that same state at b:
 *
 *      g_b = g_a + g_loc,   f_b = f_a + D(a,b) * g_a + f_loc,
 *      h2_b = h2_a + D(a,b) * (2*bd_a + D(a,b) * b2_a) + h2_loc,
 *
 *  with D(a,b) = sum of the f32 deltas d over [a, b) -- the deltas the chunk's
 *  own steps apply, d_a included and d_b excluded (an off-by-one here applies
 *  d_a*g_a twice; it only shows on columns without ties). Using the summed f32
 *  deltas (not x_a - x_b) keeps every row's effective weight identical to the
 *  CPU's, so the two agree to f64 reassociation error.
 *
 *  Three device passes over the sorted rows do the work:
 *    pass 1   per (chunk, column): the local end state from zero (no scoring);
 *    propagate  per (column, lane): a sequential walk over the chunks that
 *             turns the local states into carry-ins (C x (m+3) x p work);
 *    pass 2   per (chunk, column): rerun the recurrence from the carry-in and
 *             score every on-grid cut, keeping the chunk's best.
 *  A final kernel picks the best chunk per column (first on ties, as the CPU's
 *  strict `>` does).
 *
 *  Thread mapping: one block per (chunk, group of JB columns); thread tid owns
 *  lanes c = tid + q*blockDim: c < m is Bo column c, c == m the Bx column,
 *  c == m+1 the parent-b lane (b2/bd, plus ||h+||^2) and c == m+2 the y lane
 *  (vb/w). Each step the block reads one gathered Bo row (contiguous: Bo is
 *  kept row-major on the device for this) shared by its JB columns, as the
 *  CPU does. Rows are staged a tile at a time in shared memory so the gather
 *  latency is paid with many loads in flight, not once per sequential step.
 *  Everything accumulates in f64 (f/g in f32 are a rejected failure mode, see
 *  CLAUDE.md); on a weak-f64 GPU that FMA rate is the floor of this kernel.
 *
 *  Support tracking: i_last[j] is the last sorted position whose parent entry
 *  is nonzero. Cuts past it are degenerate (the CPU stops there), so chunks
 *  beyond it are skipped per column and the grid is clipped to i <= i_last.
 */

#include "../kernels.h"   // mars::HINGE_DEN_REL
#include "context.cuh"

#include <cub/cub.cuh>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace mars {
namespace cuda {

namespace {

constexpr int JB        = 4;     // candidate columns per block (share the gathered Bo row)
constexpr int MAX_LANES = 1024;  // m + 3 <= MAX_LANES (CPT <= 4 at blockDim 256)
constexpr int PROP_BLOCK = 256;

// --- prep kernels -----------------------------------------------------------

// Sort keys: the raw X column with -0.0 folded into +0.0 (they compare equal
// to the CPU argsort), and the identity as the sort values.
__global__ void sort_keys_kernel(const float *xraw, float *keys, int *seq, size_t n)
{
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    unsigned u = __float_as_uint(xraw[i]);
    if (u == 0x80000000u) {
        u = 0u;
    }
    keys[i] = __uint_as_float(u);
    seq[i]  = (int)i;
}

// xs[t] = x[k[t]] (the scaled candidate in sorted order); pos = k^-1.
__global__ void perm_kernel(const int *k, const float *x, float *xs, int *pos, size_t n)
{
    const size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n) {
        return;
    }
    const int r = k[t];
    xs[t]  = x[r];
    pos[r] = (int)t;
}

// d[t] = xs[t-1] - xs[t] in f32 (the CPU's `d`), d[0] = 0.
__global__ void delta_kernel(const float *xs, float *d, size_t n)
{
    const size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n) {
        return;
    }
    d[t] = (t > 0) ? xs[t - 1] - xs[t] : 0.0f;
}

// dsum[ck] = sum of d over [a, a+L), a = ck*L -- the carry's D(a, b), the
// deltas the chunk's own steps apply. Only chunks with a next chunk are
// consumed; the rest are written as 0.
__global__ void dsum_kernel(const float *d, double *dsum, int L, int tail)
{
    const int ck = blockIdx.x;
    const int a  = ck * L;
    const int b  = a + L;
    double acc = 0.0;
    if (b <= tail) {
        for (int s = a + (int)threadIdx.x; s < b; s += blockDim.x) {
            acc += (double)d[s];
        }
    }
    acc = blockReduceSum(acc);
    if (threadIdx.x == 0) {
        dsum[ck] = acc;
    }
}

__inline__ __device__ int warpReduceMax(int v)
{
    for (int off = warpSize / 2; off > 0; off >>= 1) {
        v = max(v, __shfl_down_sync(0xffffffffu, v, off));
    }
    return v;
}

// i_last[j] = max sorted position t with B[k[t], mask[j]] != 0, clamped to
// tail-1 (positions in the end span are never swept), or -1. Reads the parent
// column and the inverse permutation contiguously.
__global__ void i_last_kernel(const float *B, size_t n, const int *mask,
                              const int *pos, int tail, int *i_last)
{
    __shared__ int sh[32];
    const int j = blockIdx.x;
    const float *col = B + (size_t)mask[j] * n;
    int best = -1;
    for (size_t r = threadIdx.x; r < n; r += blockDim.x) {
        if (col[r] != 0.0f) {
            best = max(best, pos[r]);
        }
    }
    best = warpReduceMax(best);
    const int lane = threadIdx.x % warpSize;
    const int wid  = threadIdx.x / warpSize;
    if (lane == 0) {
        sh[wid] = best;
    }
    __syncthreads();
    const int nwarps = (blockDim.x + warpSize - 1) / warpSize;
    best = (threadIdx.x < nwarps) ? sh[lane] : -1;
    if (wid == 0) {
        best = warpReduceMax(best);
        if (lane == 0) {
            i_last[j] = min(best, tail - 1);
        }
    }
}

// --- the sweep --------------------------------------------------------------

// First grid position >= i0 (the grid is head+1, head+1+min_span, ...).
__device__ __forceinline__ int first_grid_at(int i0, int head, int min_span)
{
    const int g0 = head + 1;
    if (i0 <= g0) {
        return g0;
    }
    return g0 + ((i0 - g0 + min_span - 1) / min_span) * min_span;
}

/*
 *  One block per (chunk ck, column group jg). PASS1: start from zero, no
 *  scoring, write the local end state (after the chunk's last step) as the
 *  chunk's carry record. PASS2: start from the carry-in, score on-grid cuts,
 *  write the chunk's best per column.
 *
 *  Dynamic shared memory layout (TS = rows per tile, Lpad = CPT*blockDim).
 *  The block-uniform per-row values are staged as f64: converted once per
 *  tile rather than by every thread at every step (f32->f64 conversion runs
 *  at the f64 rate, so nine of them per step would cost as much as the eight
 *  FMAs). Only the per-thread Bo value is converted in the step loop.
 *      double s_acc[TS][JB][4]    (PASS2) per-slot ff, fy, h2, w
 *      double s_d  [TS], s_y[TS]  delta and gathered y
 *      double s_b  [JB][TS]       gathered parent b per column
 *      double s_bx [JB][TS]       gathered Bx per column
 *      float  s_bo [TS][Lpad]     gathered Bo rows (lanes c < m)
 *      int    s_k  [TS]           row index of each sorted position
 */
template <int CPT, bool PASS1>
__global__ void __launch_bounds__(256)
sweep_kernel(const float *__restrict__ Bo_rm, int ldBo,
             const float *__restrict__ B, const float *__restrict__ Bx,
             const float *__restrict__ y, const int *__restrict__ k,
             const float *__restrict__ d, const int *__restrict__ mask,
             const double *__restrict__ ybo, const double *__restrict__ ybx,
             const int *__restrict__ i_last,
             size_t n, int m, int p, int L, int tail, int head, int min_span, int TS,
             double tol,
             double *__restrict__ carry_f, double *__restrict__ carry_g,
             double *__restrict__ carry_h2,
             double *__restrict__ best_sse, int *__restrict__ best_idx)
{
    extern __shared__ __align__(16) unsigned char smem_raw[];
    const int BD   = blockDim.x;
    const int Lc   = m + 3;
    const int Lpad = CPT * BD;
    double *s_acc = reinterpret_cast<double *>(smem_raw);          // [TS][JB][4] (PASS2 only)
    double *s_d   = s_acc + (PASS1 ? 0 : (size_t)TS * JB * 4);     // [TS]
    double *s_y   = s_d + TS;                                      // [TS]
    double *s_b   = s_y + TS;                                      // [JB][TS]
    double *s_bx  = s_b + JB * TS;                                 // [JB][TS]
    float  *s_bo  = reinterpret_cast<float *>(s_bx + JB * TS);     // [TS][Lpad]
    int    *s_k   = reinterpret_cast<int *>(s_bo + (size_t)TS * Lpad);

    const int ck = blockIdx.x;
    const int jg = blockIdx.y;
    const int a  = ck * L;
    const int b  = min(a + L, tail);
    const int tid = threadIdx.x;

    // Columns of this block. PASS1 needs a chunk only if its carry-out feeds a
    // later chunk the column still sweeps; PASS2 needs it if any cut in it is
    // at or before the column's last nonzero parent row.
    int  jcol[JB], ilast[JB];
    bool act[JB];
    bool any = false;
    #pragma unroll
    for (int jj = 0; jj < JB; ++jj) {
        const int j = jg * JB + jj;
        jcol[jj]  = j;
        ilast[jj] = -1;
        act[jj]   = false;
        if (j < p) {
            ilast[jj] = i_last[j];
            act[jj]   = PASS1 ? (a + L <= ilast[jj]) : (a <= ilast[jj]);
        }
        any |= act[jj];
    }
    if (!any) {
        return;
    }

    // Lane kinds: 0 Bo column c, 1 Bx column, 2 parent-b lane, 3 y lane, 4 idle.
    int    kind[CPT];
    double wgt[CPT];
    #pragma unroll
    for (int q = 0; q < CPT; ++q) {
        const int c = tid + q * BD;
        kind[q] = (c < m) ? 0 : (c == m) ? 1 : (c == m + 1) ? 2 : (c == m + 2) ? 3 : 4;
        wgt[q]  = (c < m) ? ybo[c] : 0.0;
    }
    double ybxj[JB];
    #pragma unroll
    for (int jj = 0; jj < JB; ++jj) {
        ybxj[jj] = act[jj] ? ybx[jcol[jj]] : 0.0;
    }

    // State. k0/k1 (||h+||^2 halves) live only in the parent-b lane.
    double f[CPT][JB], g[CPT][JB], k0[JB], k1[JB];
    #pragma unroll
    for (int jj = 0; jj < JB; ++jj) {
        k0[jj] = 0.0;
        k1[jj] = 0.0;
        #pragma unroll
        for (int q = 0; q < CPT; ++q) {
            f[q][jj] = 0.0;
            g[q][jj] = 0.0;
        }
        if (!PASS1 && act[jj]) {
            const size_t base = ((size_t)ck * p + jcol[jj]) * Lc;
            #pragma unroll
            for (int q = 0; q < CPT; ++q) {
                const int c = tid + q * BD;
                if (c < Lc) {
                    f[q][jj] = carry_f[base + c];
                    g[q][jj] = carry_g[base + c];
                }
            }
            k0[jj] = carry_h2[(size_t)ck * p + jcol[jj]];
        }
    }

    double bsse[JB];
    int    bidx[JB];
    #pragma unroll
    for (int jj = 0; jj < JB; ++jj) {
        bsse[jj] = 0.0;
        bidx[jj] = -1;
    }
    if (!PASS1) {
        for (int e = tid; e < TS * JB * 4; e += BD) {
            s_acc[e] = 0.0;
        }
    }

    const int lane_w = tid & 31;
    int prev_first = 0, prev_nslots = 0;   // the previous tile's grid, for the scorer

    for (int i0 = a; i0 < b; i0 += TS) {
        const int i1 = min(i0 + TS, b);
        const int nt = i1 - i0;

        // --- phase A1: sorted-row indices + deltas; score the previous tile ---
        for (int t = tid; t < nt; t += BD) {
            s_k[t] = k[i0 + t];
            s_d[t] = (double)d[i0 + t];
        }
        if (!PASS1) {
            // Thread jj scores column jj. Unrolled with a thread match (not
            // bsse[tid]) so the per-column arrays stay in registers.
            #pragma unroll
            for (int jj = 0; jj < JB; ++jj) {
                if (tid == jj && act[jj]) {
                    for (int s = 0; s < prev_nslots; ++s) {
                        const int i = prev_first + s * min_span;
                        if (i > ilast[jj]) {
                            break;
                        }
                        double *e = s_acc + ((size_t)s * JB + jj) * 4;
                        const double ff = e[0], fy = e[1], h2 = e[2], w = e[3];
                        const double uw  = fy - w;
                        const double den = h2 - ff;
                        const double sse = den > fmax(tol, mars::HINGE_DEN_REL * h2)
                                           ? (uw * uw) / (den + tol) : 0.0;
                        if (sse > bsse[jj]) {
                            bsse[jj] = sse;
                            bidx[jj] = i;
                        }
                        e[0] = e[1] = e[2] = e[3] = 0.0;
                    }
                }
            }
        }
        __syncthreads();

        // --- phase A2: gathers through s_k ---
        for (int t = tid; t < nt; t += BD) {
            s_y[t] = (double)y[s_k[t]];
        }
        #pragma unroll
        for (int jj = 0; jj < JB; ++jj) {
            if (act[jj]) {
                const float *bcol  = B  + (size_t)mask[jcol[jj]] * n;
                const float *bxcol = Bx + (size_t)jcol[jj] * n;
                for (int t = tid; t < nt; t += BD) {
                    const int kk = s_k[t];
                    s_b [jj * TS + t] = (double)bcol[kk];
                    s_bx[jj * TS + t] = (double)bxcol[kk];
                }
            }
        }
        #pragma unroll
        for (int q = 0; q < CPT; ++q) {
            if (kind[q] == 0) {
                const int c = tid + q * BD;
                #pragma unroll 8
                for (int t = 0; t < nt; ++t) {
                    s_bo[(size_t)t * Lpad + c] = Bo_rm[(size_t)s_k[t] * ldBo + c];
                }
            }
        }
        __syncthreads();

        // --- phase B: the sequential recurrence over the tile ---
        const int first = first_grid_at(i0, head, min_span);
        int next_grid = first;
        int slot = 0;
        for (int t = 0; t < nt; ++t) {
            const int    i  = i0 + t;
            const double dv = s_d[t];
            const double yv = s_y[t];
            double bv[JB], bxv[JB];
            bool   go[JB];
            #pragma unroll
            for (int jj = 0; jj < JB; ++jj) {
                bv[jj]  = s_b [jj * TS + t];
                bxv[jj] = s_bx[jj * TS + t];
                go[jj]  = act[jj] && (i <= ilast[jj]);   // block-uniform
            }
            double bo[CPT];
            #pragma unroll
            for (int q = 0; q < CPT; ++q) {
                bo[q] = (kind[q] == 0) ? (double)s_bo[(size_t)t * Lpad + tid + q * BD] : 0.0;
            }

            // f-part: f += d*g (k0/k1 first: they use bd before and b2 after
            // the previous step, i.e. the current f and g, as the CPU does).
            #pragma unroll
            for (int q = 0; q < CPT; ++q) {
                #pragma unroll
                for (int jj = 0; jj < JB; ++jj) {
                    if (go[jj]) {
                        if (kind[q] == 2) {
                            k0[jj] = fma(dv * dv, g[q][jj], k0[jj]);
                            k1[jj] = fma(dv * 2.0, f[q][jj], k1[jj]);
                        }
                        f[q][jj] = fma(dv, g[q][jj], f[q][jj]);
                    }
                }
            }

            if (!PASS1 && i == next_grid) {
                #pragma unroll
                for (int jj = 0; jj < JB; ++jj) {
                    if (go[jj]) {
                        double pf = 0.0, py = 0.0;
                        #pragma unroll
                        for (int q = 0; q < CPT; ++q) {
                            if (kind[q] <= 1) {
                                const double w = (kind[q] == 0) ? wgt[q] : ybxj[jj];
                                pf = fma(f[q][jj], f[q][jj], pf);
                                py = fma(f[q][jj], w, py);
                            }
                        }
                        pf = warpReduceSum(pf);
                        py = warpReduceSum(py);
                        double *e = s_acc + ((size_t)slot * JB + jj) * 4;
                        if (lane_w == 0) {
                            atomicAdd(e + 0, pf);
                            atomicAdd(e + 1, py);
                        }
                        #pragma unroll
                        for (int q = 0; q < CPT; ++q) {
                            if (kind[q] == 2) {
                                e[2] = k0[jj] + k1[jj];
                            }
                            else if (kind[q] == 3) {
                                e[3] = f[q][jj];
                            }
                        }
                    }
                }
                ++slot;
                next_grid += min_span;
            }

            // g-part: g += b * value
            #pragma unroll
            for (int q = 0; q < CPT; ++q) {
                #pragma unroll
                for (int jj = 0; jj < JB; ++jj) {
                    if (go[jj]) {
                        const double v = (kind[q] == 0) ? bo[q]
                                         : (kind[q] == 1) ? bxv[jj]
                                         : (kind[q] == 2) ? bv[jj]
                                         : (kind[q] == 3) ? yv : 0.0;
                        g[q][jj] = fma(bv[jj], v, g[q][jj]);
                    }
                }
            }
        }
        prev_first  = first;
        prev_nslots = slot;
        __syncthreads();
    }

    if (PASS1) {
        // The local end state (after step b-1) is the chunk's carry record.
        #pragma unroll
        for (int jj = 0; jj < JB; ++jj) {
            if (!act[jj]) {
                continue;
            }
            const size_t base = ((size_t)ck * p + jcol[jj]) * Lc;
            #pragma unroll
            for (int q = 0; q < CPT; ++q) {
                const int c = tid + q * BD;
                if (c < Lc) {
                    carry_f[base + c] = f[q][jj];
                    carry_g[base + c] = g[q][jj];
                }
                if (kind[q] == 2) {
                    carry_h2[(size_t)ck * p + jcol[jj]] = k0[jj] + k1[jj];
                }
            }
        }
    }
    else {
        // Score the last tile, then publish the chunk's best per column.
        #pragma unroll
        for (int jj = 0; jj < JB; ++jj) {
            if (tid == jj && act[jj]) {
                for (int s = 0; s < prev_nslots; ++s) {
                    const int i = prev_first + s * min_span;
                    if (i > ilast[jj]) {
                        break;
                    }
                    const double *e = s_acc + ((size_t)s * JB + jj) * 4;
                    const double ff = e[0], fy = e[1], h2 = e[2], w = e[3];
                    const double uw  = fy - w;
                    const double den = h2 - ff;
                    const double sse = den > fmax(tol, mars::HINGE_DEN_REL * h2)
                                       ? (uw * uw) / (den + tol) : 0.0;
                    if (sse > bsse[jj]) {
                        bsse[jj] = sse;
                        bidx[jj] = i;
                    }
                }
                best_sse[(size_t)ck * p + jcol[jj]] = bsse[jj];
                best_idx[(size_t)ck * p + jcol[jj]] = bidx[jj];
            }
        }
    }
}

// Turn the per-chunk local end states (pass 1) into carry-ins, in place. One
// thread per (column j, lane c), walking the chunks the column sweeps.
__global__ void propagate_kernel(double *carry_f, double *carry_g, double *carry_h2,
                                 const double *dsum, const int *i_last,
                                 int p, int m, int L)
{
    const int    Lc = m + 3;
    const size_t e  = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= (size_t)p * Lc) {
        return;
    }
    const int j  = (int)(e / Lc);
    const int c  = (int)(e - (size_t)j * Lc);
    const int il = i_last[j];
    if (il < 0) {
        return;
    }
    const int  C_j   = il / L;          // last chunk the column sweeps
    const bool blane = (c == m + 1);
    double F = 0.0, G = 0.0, H = 0.0;
    for (int ck = 0; ck < C_j; ++ck) {
        const size_t idx = ((size_t)ck * p + j) * Lc + c;
        const double lf = carry_f[idx];
        const double lg = carry_g[idx];
        carry_f[idx] = F;
        carry_g[idx] = G;
        const double dl = dsum[ck];
        if (blane) {
            const size_t hidx = (size_t)ck * p + j;
            const double lh = carry_h2[hidx];
            carry_h2[hidx] = H;
            H = H + dl * (2.0 * F + dl * G) + lh;
        }
        F = F + dl * G + lf;
        G = G + lg;
    }
    const size_t idx = ((size_t)C_j * p + j) * Lc + c;
    carry_f[idx] = F;
    carry_g[idx] = G;
    if (blane) {
        carry_h2[(size_t)C_j * p + j] = H;
    }
}

// Best chunk per column (first on ties), and the cut value at that position.
__global__ void final_kernel(const double *best_sse, const int *best_idx,
                             const int *i_last, const int *k, const float *xraw,
                             int p, int L,
                             double *h_sse, int *h_idx, double *h_cut)
{
    const int j = (int)((size_t)blockIdx.x * blockDim.x + threadIdx.x);
    if (j >= p) {
        return;
    }
    const int il = i_last[j];
    double bs = 0.0;
    int    bi = -1;
    if (il >= 0) {
        const int C_j = il / L;
        for (int ck = 0; ck <= C_j; ++ck) {
            const double s = best_sse[(size_t)ck * p + j];
            const int    i = best_idx[(size_t)ck * p + j];
            if (i >= 0 && s > bs) {
                bs = s;
                bi = i;
            }
        }
    }
    h_sse[j] = bs;
    h_idx[j] = bi;
    h_cut[j] = (bi >= 0) ? (double)xraw[k[bi]] : nan("");
}

// --- host helpers -----------------------------------------------------------

// Chunk-sized buffers: (C, p, m+3) carries, (C, p) per-chunk scalars and
// bests, (C) delta sums. Grown (never shrunk) to the current call's need.
void ensure_chunk_buffers(Context *ctx, size_t C, size_t p, size_t Lc)
{
    const size_t need_c  = C * p * Lc;
    const size_t need_cp = C * p;
    if (need_c > ctx->carry_cap) {
        cudaFree(ctx->dcarry_f);
        cudaFree(ctx->dcarry_g);
        ctx->dcarry_f = ctx->dcarry_g = nullptr;
        ctx->carry_cap = 0;
        CUDA_CHECK(cudaMalloc(&ctx->dcarry_f, need_c * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&ctx->dcarry_g, need_c * sizeof(double)));
        ctx->carry_cap = need_c;
    }
    if (need_cp > ctx->chunkp_cap) {
        cudaFree(ctx->dcarry_h2);
        cudaFree(ctx->dbest_sse);
        cudaFree(ctx->dbest_idx);
        ctx->dcarry_h2 = ctx->dbest_sse = nullptr;
        ctx->dbest_idx = nullptr;
        ctx->chunkp_cap = 0;
        CUDA_CHECK(cudaMalloc(&ctx->dcarry_h2, need_cp * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&ctx->dbest_sse, need_cp * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&ctx->dbest_idx, need_cp * sizeof(int)));
        ctx->chunkp_cap = need_cp;
    }
    if (C > ctx->chunk_cap) {
        cudaFree(ctx->ddsum);
        ctx->ddsum = nullptr;
        ctx->chunk_cap = 0;
        CUDA_CHECK(cudaMalloc(&ctx->ddsum, C * sizeof(double)));
        ctx->chunk_cap = C;
    }
}

void ensure_sweep_buffers(Context *ctx)
{
    if (ctx->dxraw) {
        return;
    }
    const size_t n = ctx->n;
    CUDA_CHECK(cudaMalloc(&ctx->dxraw,     n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ctx->dkeys,     n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ctx->dkeys_out, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ctx->dseq,      n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&ctx->dk,        n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&ctx->dpos,      n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&ctx->dxs,       n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ctx->dd,        n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ctx->dybo,      ctx->max_terms * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&ctx->di_last,   ctx->p_cap * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&ctx->dh_sse,    ctx->p_cap * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&ctx->dh_idx,    ctx->p_cap * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&ctx->dh_cut,    ctx->p_cap * sizeof(double)));
    size_t bytes = 0;
    CUDA_CHECK(cub::DeviceRadixSort::SortPairsDescending(
                   nullptr, bytes, ctx->dkeys, ctx->dkeys_out, ctx->dseq, ctx->dk,
                   (int)n, 0, 32, ctx->stream));
    CUDA_CHECK(cudaMalloc(&ctx->dsort_tmp, bytes));
    ctx->sort_tmp_bytes = bytes;
}

inline double wall_ms(const std::chrono::steady_clock::time_point &t0)
{
    return std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now() - t0).count();
}

template <int CPT>
void launch_passes(Context *ctx, size_t m, size_t p, int BD, int TS, int L, int C,
                   int n_jg, int tail, int head, int min_span,
                   size_t smem1, size_t smem2, double *t_pass1, double *t_pass2)
{
    cudaStream_t s = ctx->stream;
    const dim3 grid((unsigned)C, (unsigned)n_jg);
    auto t0 = std::chrono::steady_clock::now();
    sweep_kernel<CPT, true><<<grid, BD, smem1, s>>>(
        ctx->dBo_rm, (int)ctx->max_terms, ctx->dB, ctx->dBx_f32, ctx->dy,
        ctx->dk, ctx->dd, ctx->dmask, ctx->dybo, ctx->dybx, ctx->di_last,
        ctx->n, (int)m, (int)p, L, tail, head, min_span, TS, ctx->tol,
        ctx->dcarry_f, ctx->dcarry_g, ctx->dcarry_h2, ctx->dbest_sse, ctx->dbest_idx);
    check_launch();
    const size_t prop_n = p * (m + 3);
    propagate_kernel<<<(unsigned)((prop_n + PROP_BLOCK - 1) / PROP_BLOCK), PROP_BLOCK, 0, s>>>(
        ctx->dcarry_f, ctx->dcarry_g, ctx->dcarry_h2, ctx->ddsum, ctx->di_last,
        (int)p, (int)m, L);
    check_launch();
    if (ctx->profile) {
        CUDA_CHECK(cudaStreamSynchronize(s));
        *t_pass1 += wall_ms(t0);
        t0 = std::chrono::steady_clock::now();
    }
    sweep_kernel<CPT, false><<<grid, BD, smem2, s>>>(
        ctx->dBo_rm, (int)ctx->max_terms, ctx->dB, ctx->dBx_f32, ctx->dy,
        ctx->dk, ctx->dd, ctx->dmask, ctx->dybo, ctx->dybx, ctx->di_last,
        ctx->n, (int)m, (int)p, L, tail, head, min_span, TS, ctx->tol,
        ctx->dcarry_f, ctx->dcarry_g, ctx->dcarry_h2, ctx->dbest_sse, ctx->dbest_idx);
    check_launch();
    if (ctx->profile) {
        CUDA_CHECK(cudaStreamSynchronize(s));
        *t_pass2 += wall_ms(t0);
    }
}

}  // namespace

void sweep_free(Context *ctx)
{
    cudaFree(ctx->dxraw);
    cudaFree(ctx->dkeys);
    cudaFree(ctx->dkeys_out);
    cudaFree(ctx->dseq);
    cudaFree(ctx->dk);
    cudaFree(ctx->dpos);
    cudaFree(ctx->dxs);
    cudaFree(ctx->dd);
    cudaFree(ctx->dybo);
    cudaFree(ctx->di_last);
    cudaFree(ctx->dsort_tmp);
    cudaFree(ctx->dcarry_f);
    cudaFree(ctx->dcarry_g);
    cudaFree(ctx->dcarry_h2);
    cudaFree(ctx->ddsum);
    cudaFree(ctx->dbest_sse);
    cudaFree(ctx->dbest_idx);
    cudaFree(ctx->dh_sse);
    cudaFree(ctx->dh_idx);
    cudaFree(ctx->dh_cut);
    ctx->dxraw = ctx->dkeys = ctx->dkeys_out = ctx->dxs = ctx->dd = nullptr;
    ctx->dseq = ctx->dk = ctx->dpos = ctx->di_last = ctx->dbest_idx = ctx->dh_idx = nullptr;
    ctx->dybo = ctx->dcarry_f = ctx->dcarry_g = ctx->dcarry_h2 = ctx->ddsum = nullptr;
    ctx->dbest_sse = ctx->dh_sse = ctx->dh_cut = nullptr;
    ctx->dsort_tmp = nullptr;
    ctx->sort_tmp_bytes = ctx->carry_cap = ctx->chunkp_cap = ctx->chunk_cap = 0;
}

void hinge_sweep(Context *ctx, size_t m, size_t p,
                 const float *xraw, const double *ybo,
                 int min_span, int end_span,
                 double *hinge_sse, int *hinge_idx, double *hinge_cut)
{
    if (p == 0) {
        return;
    }
    const size_t n  = ctx->n;
    cudaStream_t s  = ctx->stream;
    const int    Lc = (int)m + 3;
    if (Lc > MAX_LANES) {
        throw std::runtime_error("cuda hinge_sweep: basis too large (m + 3 > 1024 lanes)");
    }
    if (p > ctx->p_cap) {
        throw std::runtime_error("cuda hinge_sweep: p exceeds the context's column capacity");
    }
    ensure_sweep_buffers(ctx);
    const auto t_start = std::chrono::steady_clock::now();

    const int head = end_span;
    const int tail = (int)n - end_span;
    if (tail <= 1 || head + 1 >= tail) {
        // No admissible cut: every position is inside an end span.
        for (size_t j = 0; j < p; ++j) {
            hinge_sse[j] = 0.0;
            hinge_idx[j] = -1;
            hinge_cut[j] = NAN;
        }
        return;
    }

    // Block shape: one lane per (m+3) accumulator pair, CPT lanes per thread.
    int BD, CPT;
    if (ctx->sweep_block) {
        BD  = ctx->sweep_block;
        CPT = (Lc + BD - 1) / BD;
        if (CPT > 4) {
            throw std::runtime_error("cuda hinge_sweep: MARS_CUDA_SWEEP_BLOCK too small for m");
        }
        CPT = (CPT == 3) ? 4 : CPT;   // instantiated for 1, 2, 4 lanes per thread
    }
    else if (Lc <= 256) {
        BD = ((Lc + 31) / 32) * 32;
        CPT = 1;
    }
    else if (Lc <= 512) {
        BD = 256;
        CPT = 2;
    }
    else {
        BD = 256;
        CPT = 4;
    }
    const int Lpad = BD * CPT;
    // Rows per tile. The tile bounds shared memory (and so residency), but the
    // sweep is bound by per-step f64 work, not occupancy: 8/16/32 rows measured
    // identical at n=5e5, m=100 and 64 was 15% slower, so 16 by default.
    int TS = std::min(16, std::max(4, (8192 / Lpad) & ~3));
    if (ctx->sweep_tile) {
        TS = ctx->sweep_tile;
    }
    const size_t smem_rows = (size_t)TS * Lpad * sizeof(float);
    const size_t smem_side = (size_t)TS * sizeof(int) +
                             (size_t)TS * (2 + 2 * JB) * sizeof(double);
    const size_t smem1 = smem_rows + smem_side;
    const size_t smem2 = smem1 + (size_t)TS * JB * 4 * sizeof(double);

    // Chunking: enough blocks to fill the GPU several times over, chunks long
    // enough that the per-chunk carry work (propagate, C x p x Lc) stays small.
    const int n_jg = (int)((p + JB - 1) / JB);
    int L;
    if (ctx->sweep_chunk) {
        L = (int)ctx->sweep_chunk;
    }
    else {
        const int C_target = std::max(1, 6144 / n_jg);
        L = (tail + C_target - 1) / C_target;
        L = std::min(8192, std::max(128, L));
    }
    const int C = (tail + L - 1) / L;

    ensure_chunk_buffers(ctx, (size_t)C, p, (size_t)Lc);

    // Uploads: the raw column (sort keys + cut values) and Boᵀy.
    CUDA_CHECK(cudaMemcpyAsync(ctx->dxraw, xraw, n * sizeof(float),
                               cudaMemcpyHostToDevice, s));
    CUDA_CHECK(cudaMemcpyAsync(ctx->dybo, ybo, m * sizeof(double),
                               cudaMemcpyHostToDevice, s));

    // Sort + sorted-order prep.
    const unsigned grid_n = (unsigned)((n + BLOCK - 1) / BLOCK);
    sort_keys_kernel<<<grid_n, BLOCK, 0, s>>>(ctx->dxraw, ctx->dkeys, ctx->dseq, n);
    check_launch();
    size_t bytes = ctx->sort_tmp_bytes;
    CUDA_CHECK(cub::DeviceRadixSort::SortPairsDescending(
                   ctx->dsort_tmp, bytes, ctx->dkeys, ctx->dkeys_out, ctx->dseq, ctx->dk,
                   (int)n, 0, 32, s));
    perm_kernel<<<grid_n, BLOCK, 0, s>>>(ctx->dk, ctx->dx, ctx->dxs, ctx->dpos, n);
    check_launch();
    delta_kernel<<<grid_n, BLOCK, 0, s>>>(ctx->dxs, ctx->dd, n);
    check_launch();
    dsum_kernel<<<(unsigned)C, BLOCK, 0, s>>>(ctx->dd, ctx->ddsum, L, tail);
    check_launch();
    i_last_kernel<<<(unsigned)p, BLOCK, 0, s>>>(ctx->dB, n, ctx->dmask, ctx->dpos, tail,
                                                ctx->di_last);
    check_launch();
    double t_sort = 0.0, t_pass1 = 0.0, t_pass2 = 0.0;
    if (ctx->profile) {
        CUDA_CHECK(cudaStreamSynchronize(s));
        t_sort = wall_ms(t_start);
    }

    switch (CPT) {
        case 1:
            launch_passes<1>(ctx, m, p, BD, TS, L, C, n_jg, tail, head, min_span,
                             smem1, smem2, &t_pass1, &t_pass2);
            break;
        case 2:
            launch_passes<2>(ctx, m, p, BD, TS, L, C, n_jg, tail, head, min_span,
                             smem1, smem2, &t_pass1, &t_pass2);
            break;
        default:
            launch_passes<4>(ctx, m, p, BD, TS, L, C, n_jg, tail, head, min_span,
                             smem1, smem2, &t_pass1, &t_pass2);
            break;
    }

    const auto t_fin = std::chrono::steady_clock::now();
    final_kernel<<<(unsigned)((p + BLOCK - 1) / BLOCK), BLOCK, 0, s>>>(
        ctx->dbest_sse, ctx->dbest_idx, ctx->di_last, ctx->dk, ctx->dxraw,
        (int)p, L, ctx->dh_sse, ctx->dh_idx, ctx->dh_cut);
    check_launch();
    CUDA_CHECK(cudaMemcpyAsync(hinge_sse, ctx->dh_sse, p * sizeof(double),
                               cudaMemcpyDeviceToHost, s));
    CUDA_CHECK(cudaMemcpyAsync(hinge_idx, ctx->dh_idx, p * sizeof(int),
                               cudaMemcpyDeviceToHost, s));
    CUDA_CHECK(cudaMemcpyAsync(hinge_cut, ctx->dh_cut, p * sizeof(double),
                               cudaMemcpyDeviceToHost, s));
    CUDA_CHECK(cudaStreamSynchronize(s));

    if (ctx->profile) {
        ctx->ms_sweep_sort  += t_sort;
        ctx->ms_sweep_pass1 += t_pass1;
        ctx->ms_sweep_pass2 += t_pass2;
        ctx->ms_sweep_misc  += wall_ms(t_fin);
        ctx->n_sweeps++;
    }
    if (std::getenv("MARS_CUDA_VERBOSE") && ctx->n_sweeps <= 1) {
        std::fprintf(stderr,
                     "[mars-cuda] hinge sweep: n=%zu m=%zu p=%zu lanes=%d block=%d cpt=%d "
                     "tile=%d chunk=%d chunks=%d smem=%zu B\n",
                     n, m, p, Lc, BD, CPT, TS, L, C, smem2);
    }
}

}  // namespace cuda
}  // namespace mars
