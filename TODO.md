# TODO

Ideas and notes for future performance/quality work, not planned for a specific
release. Completed work lives in git history; rejected approaches and why they
were rejected live in CLAUDE.md's performance notes. Items are ordered by
expected value per unit of risk as of the 2026-10-23 review. Timings below are
single-threaded on the EPYC 9845 server (`tests/bench_sweep`, n=5e5, pinned to
one core); see the benchmarking note at the end before trusting any A/B.

## 1. Hinge sweep (CPU hinge fits): resolved, with a lesson

The sweep (`covariates_impl` over sorted rows) was ~75% of `eval(hinge)` and ran
~3× slower than the identical loop in a standalone microbench. Bisection
(2026-10-23) stripped the on-grid work, support bookkeeping, scalar
accumulators and gathers one at a time: together they cost <5%. Replacing the
gathered `Bo` row by an L1-resident one cut the sweep from 2.8 s to 0.93 s,
yet no prefetch variant changed anything -- because **the binary contained no
prefetch instructions**. GCC 13 deletes a loop of `__builtin_prefetch` calls
inside a lambda at every -O level (plain loops, single calls and inline asm
survive; clang keeps all forms). The pre-review code had a single call that
survived but covered only the first line of each row.

With `prefetch_line()` (inline asm) prefetching the full `Bo` row and the
`b`/`bx`/`y` gathers 4 rows ahead, the sweep at n=5e5 is 1.3 s vs 4.6 s before
the review at m=100 and 6.7 s vs 21.9 s at m=250, bit-identical; tuning
confirmed distance 4, 4 columns per row and the forced inline. The
orthonormalize GEMMs (item 3) are now ~45% of a CPU hinge eval. Remaining headroom is small: the compute floor of the
loop (2 loads, 1 widening load, 2 FMAs, 2 stores per 4 lanes) is ~0.9 s at
this size. What is left is the store traffic of `f`/`g` (one load + one
store each per FMA); amortizing it across 2 sorted rows per step is possible
algebraically (`f += (d1+d2)·g + d2·b1·x1`) but changes the f64 summation
order and needs the on-grid positions handled exactly -- low priority.

**Lesson:** after any change to prefetch code, check
`objdump -d marslib*.so | grep -c prefetcht0`.

## 2. CUDA: the hinge sweep

Only `linear_only` runs on the GPU (`fit(cuda=True)`); the hinge sweep stays on
the CPU. Porting it is a much bigger pass (sort / segmented scan / per-cut
argmax) and the next CUDA frontier. With `append()` now 15× faster on the CPU
(it had become the dominant per-epoch cost under the GPU path), re-profile a
production `linear_only` fit with `MARS_CUDA_PROFILE=1` before deciding; the
GPU phases were flat (T GEMM 28%, round/norm 24%, fill 23%, DGKS 19%,
projection 6%) and Python orchestration ~16% as of 2026-06-17.

Lesson kept from that profile (commit `3d27433`): the random-`Bo` benches never
fire the DGKS retry, so a perf benchmark must exercise it or it measures a path
production never takes. `CudaKernelsTest.BatchDgksRetryMatchesPerColumn` covers
the batched retry.

Small CUDA items, all untested on-device until the GPU is free:

- **Deterministic reductions.** `round_reduce_kernel` sums `ds`/`ybx_raw` with
  f64 `atomicAdd` across up to 64 row tiles, so GPU results vary at f64 ULP from
  run to run (not just vs the CPU). A per-tile partials buffer (64×P doubles)
  plus a fixed-order second pass makes them reproducible; cheap.
- **`fill` f64-write drop (~7% of GPU time).** `fill` writes both an f32 and an
  f64 `Bx`; on the blocked-f32 path the f64 copy is read only by `round_reduce`
  as the minuend of the fused subtract. Reading the f32 `Bx` instead needs the
  f32 projection to land in its own buffer (it currently overwrites `dBx_f32`).
  Bit-identical result, delicate aliasing change, low ROI.
- **`context_sync_basis` host gather.** The new `Bo` column is gathered on the
  host from the row-major `Bo` (one cache line per row); `append()` has the same
  column contiguous as f64 `v` just before scattering it. Staging a contiguous
  f32 copy on `MarsData` and uploading that avoids the gather. A few ms/epoch.
- **DGKS 19%** is genuine f64 re-orthogonalization of real near-collinear
  columns; cutting it further needs an algorithmic change (re-orthogonalize
  against only the relevant `Bo` columns). High risk for the remaining slice.

## 3. Orthonormalize Phase 1/2a GEMMs are store-port-bound (CPU)

`eval(linear)` is 1.14 s at n=5e5, m=100 and 5.8 s at m=250 (23 ms per basis),
~2.6× off the FMA bound. Phase 1 (`axpy_m` building `T = Boᵀ·Bx` with `i`
outermost) stores a `T` element on every FMA and x86 retires ~1 store/cycle;
Phase 2a is load-bound the same way. Keeping `i` outermost reads `Bo`/`Bx`
exactly once, but `T` (m×p f64) cannot live in registers.

**Fix:** an `i`-blocked, register-tiled kernel: hold a `T` tile in YMM
accumulators across a panel of 8–16 rows, so each `T` load/store serves 8–16
FMAs and the `Bo` panel is re-read from L1. Same shape for Phase 2a. ~100 lines
of AVX with three tail cases; tune the panel on the server (the arm64 dev box
takes the scalar path).

**Why it can't just call BLAS:** `Bo`/`Bx` are f32 storage but `T` must
accumulate in f64 (the precision contract behind the f32 buffers); `sgemm`
accumulates in f32 and `dgemm` needs f64 inputs.

**Value:** the whole cost of CPU `linear_only` (production uses the GPU there)
and, since the sweep fix, ~45% of a CPU hinge eval (1.1 of 2.4 s at m=100,
5.8 of 12.5 s at m=250). Expected ~2× on Phase 1/2a, so ~1.3× on a hinge fit.

## 4. Python orchestration: `bmask` list comprehensions

`fit()` rebuilds the mask each epoch with two Python list comprehensions per
input (`basic_filter`, `aux_filter`) over all bases; ~130 ms/epoch on the
production fit, ~16% under the GPU path and growing as the GPU gets faster.
Precompute per epoch `degree = [len(b) for b in basis]` and a `(p, M)` bool
`contains[i, k] = (i in basis[k])`, then vectorize the basic filter:

```python
bmask &= (degree[None, :] < max_degree) & (~contains | self_interact)
```

`aux_filter` is user-supplied and stays scalar, but call it only on the entries
still true after the basic filter.

## 5. MARS algorithmic improvements

Motivation: stock MARS becomes numerically unreliable at interaction order ≥ 3
on correlated features — the Gram matrix loses effective rank, GCV rankings get
noisy, and the backward pass cannot be trusted to remove redundant terms.

Two concrete data points from the 2026-10-22 review, to re-check the symptom
against before investing here:

- The hinge sweep *was* selecting numerically-in-span cuts (den/‖h+‖² ~ 1e-11,
  spurious gain up to 1e-2, above legitimate cuts) whenever the cut passed
  below a small-support parent. Order-≥3 parents have the smallest support, so
  they were hit most. Now gated (CLAUDE.md, 2026-10-22). Re-run the failing
  cases first; part of the symptom may be gone.
- A plain hinge fit (n=2e5, p=40, 41 terms, then `prune`) kept four hinges of
  the same input at adjacent tail knots (1.83, 1.92, 2.01, 2.07) on one parent
  with coefficients −21, +165, −350, +211: knots placed `min_span` apart across
  *epochs* on the same (input, parent), fitting a wiggle in a 2% tail. This is
  the failure chain items 1–2 below target (nothing forbids a new knot next to
  an existing one on the same input/parent; `end_span` is a fraction of n, not
  Friedman's formula).

### 5.1 Fix `endspan`

Applied at the boundaries of the data: the minimum gap between the
smallest/largest data point and the first/last candidate knot per variable.
Boundary knots are asymmetrically dangerous: a knot near min(x) has support
reaching past every data point and dominates extrapolation while being backed
by few observations. Friedman's default:

$$\text{endspan} = 3 - \log_2(\alpha / d)$$

with $d$ the number of inputs and $\alpha$ the same small probability as in
`minspan` (0.05): a constant plus a $\log d$ correction, independent of $N$,
typically 10–30 samples. Today `tail_span` is a *fraction* of n (2% → 10k rows
at n=5e5). Audit: scale with $d$ per the formula; apply `endspan` and `minspan`
together (`sorted_x[endspan : N - endspan : minspan]`); consider a minimum gap
to knots already in the model on the same (input, parent).

### 5.2 Knot tuning: 1-D refinement after each forward pick

After the forward pass picks $t^*$ from the discrete grid, refine it by a line
search (Brent / golden section) for the RSS-minimizing position in
$(\tau_{i-1}, \tau_{i+1})$. Friedman rejected this in 1991 on cost; with cached
partial residuals each evaluation is $O(M)$ incremental and Brent converges in
5–10 steps, so it is sub-millisecond even at $N = 10^6$. Bakin/Hegland/Osborne
(2000) report it improves forward-pass quality. Validate on a signal with a knot
at a non-grid position (e.g. $f(x) = 1[x > 0.4137]$ with a quantile grid that
misses it): same test MSE with fewer terms. Do after 5.1.

### 5.3 BMARS basis reformulation

A change of basis internal to the linear algebra: the user still sees hinge
terms and the API/serialization are unchanged, but the matrices factored and
updated are formed in a compactly-supported linear B-spline basis per variable
(tent functions on a quantile knot grid, $K_v \in [64, 256]$), where the Gram
is banded and well conditioned. Any grid-knot hinge is an exact combination of
two adjacent B-splines plus a global linear term (de Boor §IX), so a
precomputed banded change-of-basis $W^v$ maps hinges to B-spline coefficients;
tensor-product interactions use the Kronecker factors applied as banded
matvecs, never materialized. Items 5.1–5.2 reduce the *frequency* of
conditioning failures; this removes the cause (unbounded hinge support).

Stage it: (a) univariate linear B-splines, snap-to-grid knots, replace only the
forward-pass inner products — must match hinge MARS to machine precision on
well-conditioned data; (b) tensor-product interactions (the Kronecker matvec is
the only new piece); (c) knot tuning in the B-spline basis; (d) optionally
quadratic B-splines for $C^1$. Keep the hinge representation primary (store and
serialize hinges, derive B-spline coefficients); add an explicit linear term so
boundary half-tents extrapolate cleanly; unit-test $B_j(x) = \beta_j^\top
\Phi(x)$ pointwise. Validate in three regimes: well-conditioned low order
(match), ill-conditioned order 4 on shared-factor features (hinge GCV path
plateaus and spikes, B-spline path smooth), and a real surface under bootstrap
(more stable selected terms). A multi-month project; keep the hinge path as the
oracle throughout.

## Dropped

- **`argsort` LRU cache (Fast-MARS).** `argsort` is now an O(n) stable radix
  sort (~5 ms at n=5e5 vs 51 ms for the indirect `stable_sort`); caching
  permutations no longer pays for its memory.
- **Row-order sensitivity of the constructor reductions.** They are f64 sums
  (~√n·eps_f64); the row-order dependence `tests/repro_shuffle.py` measures
  comes from tie-breaking in the sort (by row index), not from these.

## Benchmarking note

On the shared EPYC server, timings differ by ±20% between cores and between
concurrent runs — enough to invert small A/B results. Build each variant as its
own binary, `taskset -c <one core>`, run variants back to back on that core,
and diff `bench_sweep`'s dump files for bit-exactness. The bench's basis must
stay well conditioned (random hinges on random parents make the summed
delta-SSEs noise-dominated and useless for A/B).

## References

- Friedman, J.H. (1991), "Multivariate Adaptive Regression Splines",
  *Annals of Statistics* 19(1):1–141. `minspan`/`endspan` in §3; knot tuning
  discussed and rejected in §3.5 on cost grounds.
- Friedman, J.H. (1993), "Fast MARS", Stanford Tech Report LCS110. Cached
  partial residuals; the rank-1 update framework used here.
- Bakin, S., Hegland, M., Osborne, M.R. (2000), "Parallel MARS Algorithm Based
  on B-splines", *Computational Statistics* 15:463–484. The BMARS paper, with
  the knot-tuning experiment.
- de Boor, "A Practical Guide to Splines" (revised ed., 2001), §IX: the
  truncated-power ↔ B-spline change of basis.
