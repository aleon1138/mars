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

constexpr int MAX_LANES = 1024;  // m + 1 + 2*JB <= MAX_LANES (CPT <= 4 at blockDim 256)
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

// Select arr[idx] from a register array with an unrolled chain, so a
// per-thread runtime index never forces the array into local memory.
template <int N, typename T>
__device__ __forceinline__ T sel(const T (&arr)[N], int idx)
{
    T v = arr[0];
    #pragma unroll
    for (int e = 1; e < N; ++e) {
        v = (idx == e) ? arr[e] : v;
    }
    return v;
}

/*
 *  One block per (chunk ck, column group jg of JB columns). PASS1: start from
 *  zero, no scoring, write the local end state (after the chunk's last step)
 *  as the chunk's carry record. PASS2: start from the carry-in, score on-grid
 *  cuts, write the chunk's best per column.
 *
 *  Thread lanes (c = tid + q*blockDim, q < CPT), LANES = m + 1 + 2*JB:
 *      c < m           Bo column c, all JB columns in slots f[q][jj], g[q][jj]
 *      c == m          Bx lane, all JB columns
 *      c == m+1+jj     parent-b lane of column jj, in slot 0 only (b2, bd and
 *                      the k0/k1 halves of ||h+||^2); the four lanes sit in
 *                      consecutive threads so their k0/k1 FMAs share two warp
 *                      instructions instead of one thread issuing eight
 *      c == m+1+JB+jj  y lane of column jj, slot 0 only (vb, w)
 *  The carry record per column keeps the canonical LCAR = m+3 layout
 *  (Bo columns, Bx, b, y), independent of JB.
 *
 *  Slot 0 is gated by "any column active" (block-uniform) so the b/y lanes'
 *  own column is always advanced; the other slots by that column's own gate.
 *  Updates past a column's i_last are never read, so the over-gating is
 *  harmless.
 *
 *  Dynamic shared memory (TS = rows per tile, Lpad = CPT*blockDim, PW = the
 *  threads holding Bo/Bx lanes (all of them when CPT > 1, m+1 otherwise),
 *  PST = PW rounded up plus a 4-double pad against bank conflicts, SMAX =
 *  grid slots per reduction batch):
 *      double s_acc [SMAX][JB][4]        (PASS2) per-slot ff, fy, h2, w
 *      double s_part[SMAX][JB][2][PST]   (PASS2) per-thread ff/fy partials
 *      double s_d[TS], s_y[TS]           delta and gathered y
 *      double s_b[JB][TS], s_bx[JB][TS]  gathered parent b and Bx per column
 *      float  s_bo[TS][Lpad]             gathered Bo rows (lanes c < m)
 *      int    s_k[TS]                    row index of each sorted position
 *  The block-uniform per-row values are staged as f64 (f32->f64 conversion
 *  runs at the f64 rate; only the per-thread Bo value is converted per step).
 *
 *  On-grid scoring (PASS2): at each grid point every thread writes its two
 *  partials per column to s_part; after SMAX grid points (or at the chunk
 *  end) the block reduces them cooperatively -- GRP threads per (slot, column,
 *  value) each summing a strided run, then a GRP-wide shuffle -- which costs
 *  ~4x fewer f64 instructions than a per-point warp-shuffle reduction.
 */
template <int CPT, int JB, bool PASS1>
__global__ void __launch_bounds__(256)
sweep_kernel(const float *__restrict__ Bo_rm, int ldBo,
             const float *__restrict__ B, const float *__restrict__ Bx,
             const float *__restrict__ y, const int *__restrict__ k,
             const float *__restrict__ d, const int *__restrict__ mask,
             const double *__restrict__ ybo, const double *__restrict__ ybx,
             const int *__restrict__ i_last,
             size_t n, int m, int p, int L, int tail, int head, int min_span,
             int TS, int SMAX, int GRP, int PW, int PST, double tol,
             double *__restrict__ carry_f, double *__restrict__ carry_g,
             double *__restrict__ carry_h2,
             double *__restrict__ best_sse, int *__restrict__ best_idx)
{
    extern __shared__ __align__(16) unsigned char smem_raw[];
    const int BD   = blockDim.x;
    const int LCAR = m + 3;
    const int Lpad = CPT * BD;
    double *s_acc  = reinterpret_cast<double *>(smem_raw);                 // [SMAX][JB][4]
    double *s_part = s_acc + (PASS1 ? 0 : (size_t)SMAX * JB * 4);          // [SMAX][JB][2][PST]
    double *s_d    = s_part + (PASS1 ? 0 : (size_t)SMAX * JB * 2 * PST);   // [TS]
    double *s_y    = s_d + TS;                                             // [TS]
    double *s_b    = s_y + TS;                                             // [JB][TS]
    double *s_bx   = s_b + JB * TS;                                        // [JB][TS]
    float  *s_bo   = reinterpret_cast<float *>(s_bx + JB * TS);            // [TS][Lpad]
    int    *s_k    = reinterpret_cast<int *>(s_bo + (size_t)TS * Lpad);    // [TS]

    const int ck  = blockIdx.x;
    const int jg  = blockIdx.y;
    const int a   = ck * L;
    const int b   = min(a + L, tail);
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

    // Lane kinds: 0 Bo column c, 1 Bx, 2 parent-b of column my, 3 y of column
    // my, 4 idle. wgt is the ybo weight of a Bo lane.
    int    kind[CPT], my[CPT];
    double wgt[CPT];
    #pragma unroll
    for (int q = 0; q < CPT; ++q) {
        const int c = tid + q * BD;
        kind[q] = (c < m) ? 0 : (c == m) ? 1 : (c < m + 1 + JB) ? 2
                  : (c < m + 1 + 2 * JB) ? 3 : 4;
        my[q]   = (kind[q] == 2) ? c - (m + 1) : (kind[q] == 3) ? c - (m + 1 + JB) : 0;
        wgt[q]  = (c < m) ? ybo[c] : 0.0;
    }
    double ybxj[JB];
    #pragma unroll
    for (int jj = 0; jj < JB; ++jj) {
        ybxj[jj] = act[jj] ? ybx[jcol[jj]] : 0.0;
    }

    // State. k0/k1 (||h+||^2 halves) live in the parent-b lanes' slot 0.
    double f[CPT][JB], g[CPT][JB], k0 = 0.0, k1 = 0.0;
    #pragma unroll
    for (int q = 0; q < CPT; ++q) {
        #pragma unroll
        for (int jj = 0; jj < JB; ++jj) {
            f[q][jj] = 0.0;
            g[q][jj] = 0.0;
        }
        if (!PASS1) {
            const int c = tid + q * BD;
            if (kind[q] <= 1) {
                #pragma unroll
                for (int jj = 0; jj < JB; ++jj) {
                    if (act[jj]) {
                        const size_t base = ((size_t)ck * p + jcol[jj]) * LCAR + c;
                        f[q][jj] = carry_f[base];
                        g[q][jj] = carry_g[base];
                    }
                }
            }
            else if (kind[q] <= 3) {
                const int  jc  = jg * JB + my[q];
                const bool am  = sel(act, my[q]);
                if (am) {
                    const size_t base = ((size_t)ck * p + jc) * LCAR + (kind[q] == 2 ? m + 1 : m + 2);
                    f[q][0] = carry_f[base];
                    g[q][0] = carry_g[base];
                    if (kind[q] == 2) {
                        k0 = carry_h2[(size_t)ck * p + jc];
                    }
                }
            }
        }
    }

    double bsse[JB];
    int    bidx[JB];
    #pragma unroll
    for (int jj = 0; jj < JB; ++jj) {
        bsse[jj] = 0.0;
        bidx[jj] = -1;
    }

    // Reduce + score the current batch of grid slots (PASS2; block-uniform).
    // Three barriers: partials complete; sums complete; scoring complete
    // before s_acc/s_part are reused.
    int slot = 0, batch_first = 0;
    auto flush = [&]() {
        __syncthreads();
        const int T      = slot * JB * 2;          // (slot, column, value) triples
        const int per    = BD / GRP;               // triples per round
        const int rounds = (T + per - 1) / per;
        const int part   = tid % GRP;
        for (int r = 0; r < rounds; ++r) {
            const int  tr  = r * per + tid / GRP;
            const bool on  = tr < T;
            double acc = 0.0;
            if (on) {
                const double *row = s_part + (size_t)tr * PST;
                for (int e = part; e < PW; e += GRP) {
                    acc += row[e];
                }
            }
            for (int off = GRP / 2; off > 0; off >>= 1) {
                acc += __shfl_down_sync(0xffffffffu, acc, off, GRP);
            }
            if (on && part == 0) {
                s_acc[(size_t)(tr >> 1) * 4 + (tr & 1)] = acc;
            }
        }
        __syncthreads();
        #pragma unroll
        for (int jj = 0; jj < JB; ++jj) {
            if (tid == jj && act[jj]) {
                for (int s = 0; s < slot; ++s) {
                    const int i = batch_first + s * min_span;
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
            }
        }
        slot = 0;
        __syncthreads();
    };

    bool all_done = false;
    for (int i0 = a; i0 < b && !all_done; i0 += TS) {
        const int i1 = min(i0 + TS, b);
        const int nt = i1 - i0;

        // --- phase A1: sorted-row indices + deltas ---
        for (int t = tid; t < nt; t += BD) {
            s_k[t] = k[i0 + t];
            s_d[t] = (double)d[i0 + t];
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
        int next_grid = first_grid_at(i0, head, min_span);
        for (int t = 0; t < nt; ++t) {
            const int    i  = i0 + t;
            const double dv = s_d[t];
            const double yv = s_y[t];
            double bv[JB], bxv[JB];
            bool   go[JB];
            bool   go_any = false;
            #pragma unroll
            for (int jj = 0; jj < JB; ++jj) {
                bv[jj]  = s_b [jj * TS + t];
                bxv[jj] = s_bx[jj * TS + t];
                go[jj]  = act[jj] && (i <= ilast[jj]);   // block-uniform
                go_any |= go[jj];
            }
            if (!go_any) {
                all_done = true;
                break;
            }
            // Per-lane slot-0 operands and the Bo value.
            double bo[CPT], mult0[CPT], v0[CPT];
            #pragma unroll
            for (int q = 0; q < CPT; ++q) {
                bo[q] = (kind[q] == 0) ? (double)s_bo[(size_t)t * Lpad + tid + q * BD] : 0.0;
                const double bmy = sel(bv, my[q]);
                mult0[q] = (kind[q] <= 1) ? bv[0] : (kind[q] <= 3) ? bmy : 0.0;
                v0[q]    = (kind[q] == 0) ? bo[q] : (kind[q] == 1) ? bxv[0]
                           : (kind[q] == 2) ? bmy : (kind[q] == 3) ? yv : 0.0;
            }

            // f-part: f += d*g. k0/k1 first: they use bd before and b2 after
            // the previous step, i.e. the current f and g, as the CPU does.
            #pragma unroll
            for (int q = 0; q < CPT; ++q) {
                if (kind[q] == 2) {
                    k0 = fma(dv * dv, g[q][0], k0);
                    k1 = fma(dv * 2.0, f[q][0], k1);
                }
                f[q][0] = fma(dv, g[q][0], f[q][0]);
                #pragma unroll
                for (int jj = 1; jj < JB; ++jj) {
                    if (go[jj]) {
                        f[q][jj] = fma(dv, g[q][jj], f[q][jj]);
                    }
                }
            }

            if (!PASS1 && i == next_grid) {
                if (slot == 0) {
                    batch_first = i;
                }
                #pragma unroll
                for (int jj = 0; jj < JB; ++jj) {
                    double pf = 0.0, py = 0.0;
                    if (go[jj]) {
                        #pragma unroll
                        for (int q = 0; q < CPT; ++q) {
                            if (kind[q] <= 1) {
                                const double w = (kind[q] == 0) ? wgt[q] : ybxj[jj];
                                pf = fma(f[q][jj], f[q][jj], pf);
                                py = fma(f[q][jj], w, py);
                            }
                            else if (kind[q] == 2 && my[q] == jj) {
                                s_acc[((size_t)slot * JB + jj) * 4 + 2] = k0 + k1;
                            }
                            else if (kind[q] == 3 && my[q] == jj) {
                                s_acc[((size_t)slot * JB + jj) * 4 + 3] = f[q][0];
                            }
                        }
                    }
                    if (tid < PW) {
                        double *pr = s_part + ((size_t)slot * JB + jj) * 2 * PST + tid;
                        pr[0]   = pf;
                        pr[PST] = py;
                    }
                }
                ++slot;
                next_grid += min_span;
                if (slot == SMAX) {
                    flush();
                }
            }

            // g-part: g += b * value.
            #pragma unroll
            for (int q = 0; q < CPT; ++q) {
                g[q][0] = fma(mult0[q], v0[q], g[q][0]);
                #pragma unroll
                for (int jj = 1; jj < JB; ++jj) {
                    if (go[jj]) {
                        const double v = (kind[q] == 0) ? bo[q] : (kind[q] == 1) ? bxv[jj] : 0.0;
                        g[q][jj] = fma(bv[jj], v, g[q][jj]);
                    }
                }
            }
        }
        __syncthreads();
    }

    if (PASS1) {
        // The local end state (after step b-1) is the chunk's carry record.
        #pragma unroll
        for (int q = 0; q < CPT; ++q) {
            const int c = tid + q * BD;
            if (kind[q] <= 1) {
                #pragma unroll
                for (int jj = 0; jj < JB; ++jj) {
                    if (act[jj]) {
                        const size_t base = ((size_t)ck * p + jcol[jj]) * LCAR + c;
                        carry_f[base] = f[q][jj];
                        carry_g[base] = g[q][jj];
                    }
                }
            }
            else if (kind[q] <= 3) {
                const int  jc = jg * JB + my[q];
                const bool am = sel(act, my[q]);
                if (am) {
                    const size_t base = ((size_t)ck * p + jc) * LCAR + (kind[q] == 2 ? m + 1 : m + 2);
                    carry_f[base] = f[q][0];
                    carry_g[base] = g[q][0];
                    if (kind[q] == 2) {
                        carry_h2[(size_t)ck * p + jc] = k0 + k1;
                    }
                }
            }
        }
    }
    else {
        if (slot > 0) {
            flush();
        }
        #pragma unroll
        for (int jj = 0; jj < JB; ++jj) {
            if (tid == jj && act[jj]) {
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

// Dynamic shared memory above the 48 KB default needs a per-kernel opt-in
// (up to the device's limit: 64 KB on sm_75, 227 KB on sm_120).
template <typename K>
void allow_smem(K kernel, size_t bytes)
{
    if (bytes > 48 * 1024) {
        int dev = 0, cap = 0;
        CUDA_CHECK(cudaGetDevice(&dev));
        CUDA_CHECK(cudaDeviceGetAttribute(&cap, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev));
        if ((size_t)cap < bytes) {
            throw std::runtime_error("cuda hinge_sweep: block needs more shared memory than "
                                     "the device allows (reduce MARS_CUDA_SWEEP_TILE/JB)");
        }
        CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        (int)bytes));
    }
}

struct SweepShape {
    int BD, CPT, TS1, TS2, SMAX, GRP, PW, PST;
    size_t smem1, smem2;
};

template <int CPT, int JB>
void launch_passes(Context *ctx, size_t m, size_t p, const SweepShape &sh,
                   int L, int C, int n_jg, int tail, int head, int min_span,
                   double *t_pass1, double *t_pass2)
{
    const int BD = sh.BD;
    const size_t smem1 = sh.smem1, smem2 = sh.smem2;
    cudaStream_t s = ctx->stream;
    const dim3 grid((unsigned)C, (unsigned)n_jg);
    allow_smem(sweep_kernel<CPT, JB, true>,  smem1);
    allow_smem(sweep_kernel<CPT, JB, false>, smem2);
    auto t0 = std::chrono::steady_clock::now();
    sweep_kernel<CPT, JB, true><<<grid, BD, smem1, s>>>(
        ctx->dBo_rm, (int)ctx->max_terms, ctx->dB, ctx->dBx_f32, ctx->dy,
        ctx->dk, ctx->dd, ctx->dmask, ctx->dybo, ctx->dybx, ctx->di_last,
        ctx->n, (int)m, (int)p, L, tail, head, min_span, sh.TS1, sh.SMAX, sh.GRP,
        sh.PW, sh.PST, ctx->tol,
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
    sweep_kernel<CPT, JB, false><<<grid, BD, smem2, s>>>(
        ctx->dBo_rm, (int)ctx->max_terms, ctx->dB, ctx->dBx_f32, ctx->dy,
        ctx->dk, ctx->dd, ctx->dmask, ctx->dybo, ctx->dybx, ctx->di_last,
        ctx->n, (int)m, (int)p, L, tail, head, min_span, sh.TS2, sh.SMAX, sh.GRP,
        sh.PW, sh.PST, ctx->tol,
        ctx->dcarry_f, ctx->dcarry_g, ctx->dcarry_h2, ctx->dbest_sse, ctx->dbest_idx);
    check_launch();
    if (ctx->profile) {
        CUDA_CHECK(cudaStreamSynchronize(s));
        *t_pass2 += wall_ms(t0);
    }
}

template <int JB>
void launch_cpt(Context *ctx, size_t m, size_t p, const SweepShape &sh,
                int L, int C, int n_jg, int tail, int head, int min_span,
                double *t_pass1, double *t_pass2)
{
    switch (sh.CPT) {
        case 1:
            launch_passes<1, JB>(ctx, m, p, sh, L, C, n_jg, tail, head, min_span, t_pass1, t_pass2);
            break;
        case 2:
            launch_passes<2, JB>(ctx, m, p, sh, L, C, n_jg, tail, head, min_span, t_pass1, t_pass2);
            break;
        default:
            launch_passes<4, JB>(ctx, m, p, sh, L, C, n_jg, tail, head, min_span, t_pass1, t_pass2);
            break;
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
    const int    JB = (ctx->sweep_jb == 8) ? 8 : 4;       // columns per block
    const int    Lc = (int)m + 3;                          // carry record length
    const int    lanes = (int)m + 1 + 2 * JB;              // thread lanes
    if (lanes > MAX_LANES) {
        throw std::runtime_error("cuda hinge_sweep: basis too large (m + 1 + 2*JB > 1024 lanes)");
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

    // Block shape: one lane per accumulator pair, CPT lanes per thread.
    SweepShape sh;
    if (ctx->sweep_block) {
        sh.BD  = ctx->sweep_block;
        sh.CPT = (lanes + sh.BD - 1) / sh.BD;
        if (sh.CPT > 4) {
            throw std::runtime_error("cuda hinge_sweep: MARS_CUDA_SWEEP_BLOCK too small for m");
        }
        sh.CPT = (sh.CPT == 3) ? 4 : sh.CPT;   // instantiated for 1, 2, 4 lanes per thread
    }
    else {
        // The fewest lanes per thread that fit a <= 256-thread block, and the
        // smallest block for that: idle padding lanes still execute the FMAs.
        sh.CPT = 1;
        sh.BD  = ((lanes + 31) / 32) * 32;
        if (sh.BD > 256) {
            sh.CPT = 2;
            sh.BD  = (((lanes + 1) / 2 + 31) / 32) * 32;
        }
        if (sh.BD > 256) {
            sh.CPT = 4;
            sh.BD  = (((lanes + 3) / 4 + 31) / 32) * 32;
        }
    }
    const int BD   = sh.BD;
    const int Lpad = BD * sh.CPT;
    /*
     *  Shared memory is sized per pass against a residency budget (~34 KB per
     *  block, i.e. 3 blocks per SM on a 100 KB carve-out): measured, pass 2
     *  loses ~2x when its block grows past that (one resident block), and
     *  pass 1 loses ~30% when the row tile shrinks below 16 rows. So pass 1
     *  takes a 16-row tile (32 KB at 512 lanes), and pass 2 takes a one-slot
     *  reduction batch, a partials buffer only as wide as the threads holding
     *  Bo/Bx lanes, and whatever tile then fits the budget.
     */
    const size_t budget = 34 * 1024;
    const size_t per_row = (size_t)Lpad * sizeof(float) + sizeof(int) +
                           (size_t)(2 + 2 * JB) * sizeof(double);
    auto tile_for = [&](size_t bytes) {
        int t = (int)(bytes / per_row);
        t = std::min(16, std::max(4, t & ~3));
        return ctx->sweep_tile ? ctx->sweep_tile : t;
    };
    sh.TS1 = tile_for(budget);
    sh.PW  = (sh.CPT == 1) ? std::min(BD, (int)m + 1) : BD;
    sh.PST = ((sh.PW + 3) & ~3) + 4;
    const size_t per_slot = (size_t)JB * (4 + 2 * sh.PST) * sizeof(double);
    sh.SMAX = ctx->sweep_batch ? ctx->sweep_batch : 1;
    sh.GRP  = 1;
    while (sh.GRP * 2 <= std::min(32, BD / (sh.SMAX * JB * 2))) {
        sh.GRP *= 2;
    }
    const size_t partials = (size_t)sh.SMAX * per_slot;
    sh.TS2   = tile_for(partials < budget ? budget - partials : 0);
    sh.smem1 = (size_t)sh.TS1 * per_row;
    sh.smem2 = (size_t)sh.TS2 * per_row + partials;

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

    if (JB == 8) {
        launch_cpt<8>(ctx, m, p, sh, L, C, n_jg, tail, head, min_span, &t_pass1, &t_pass2);
    }
    else {
        launch_cpt<4>(ctx, m, p, sh, L, C, n_jg, tail, head, min_span, &t_pass1, &t_pass2);
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
                     "[mars-cuda] hinge sweep: n=%zu m=%zu p=%zu jb=%d lanes=%d block=%d cpt=%d "
                     "tiles=%d/%d batch=%d grp=%d pw=%d chunk=%d chunks=%d smem=%zu/%zu B\n",
                     n, m, p, JB, lanes, BD, sh.CPT, sh.TS1, sh.TS2, sh.SMAX, sh.GRP, sh.PW,
                     L, C, sh.smem1, sh.smem2);
    }
}

}  // namespace cuda
}  // namespace mars
