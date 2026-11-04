# TODO

Open ideas for future performance/quality work, not planned for a specific
release. Only open work lives here: completed work is in git history, and the
measurements, lessons and rejected approaches it produced are in CLAUDE.md's
notes (dated). Items are ordered by expected value per unit of risk as of the
2026-11-04 review. Timings are single-threaded on the EPYC 9845 server
(`tests/bench_sweep`, n=5e5, pinned to one core) unless noted; see the
benchmarking note in CLAUDE.md before trusting any A/B.

## 1. Linear-only scoring: numerics and speed (2026-11-04 review)

Found by a numpy model of both paths against an f64 ground truth on the same
f32 data (fit-like basis, correlated features with mean/std = 3; n=2e4, m=120
and n=6e3, m=400). The GPU numbers are emulated with OpenBLAS sgemm, not
measured on the device. κ = ‖Bx‖ / ‖residual‖.

**Numerics** (#1, the residual target, and #2, the GPU DGKS retry restarting
from the f64 `B·x`, landed 2026-11-04 -- CLAUDE.md note)

2. **Exactly in-span candidates on the GPU: thin margin.** They still score 0,
   but in the model their residual reached 0.4·tol at m=400, n=6e3 (`tol` sits
   at its f32 floor at small n and the f32-projection noise grows as √m). Add
   a duplicate-candidate case at m≈1000, small n to `test_cuda`.
3. **f32 normalize scale** adds ±1.2e-7 relative per-candidate noise to
   ΔSSE (both paths). Free fix: `ΔSSE = <r,y>²/(s+tol)` in f64.
4. **`grid.y = P` can exceed 65535** in the batched GPU path: `p_cap =
   8e9/(16n)` passes it for n < ~7600, so a large block fails to launch.
   Clamp `p_cap` or loop over `grid.y`.

**Speed**

- **C. Cache `T` across epochs, and the GPU closed form (next after item 3).**
  B landed on the CPU (2026-11-04, CLAUDE.md). `Bo`, `B` and the scaled X
  never change, so `T[k', (k,c)]` never changes: per epoch only a new row for
  every candidate plus the new parent's columns are needed, two
  (m×n)·(n×p_x) GEMMs instead of p_x·m single-column passes (~m/2 less work).
  Memory m_max²·p_x f64 (128 MB at 400·400·100). Keep `TᵀT`, `TᵀET`, `Tᵀybo`,
  `TᵀE·ybo`, `‖Bx‖²`, `<Bx,y>` per candidate and extend them incrementally
  (O(m) per candidate per epoch via E's new row), or a Cholesky of `BoᵀBo`.
  **The GPU closed form belongs here:** `den` takes `T` errors at first order,
  and the blocked-f32 `T` (~eps_f32·L/√(3n) relative per coherent entry, so
  ~2e-7·κ² on `den` at n=5e5, L=2048) is too coarse, while a full Dgemm `T`
  costs more than it saves on the weak-f64 card. With the cache, each epoch's
  new rows are two thin GEMMs, cheap enough in f64.
- **D. Centering x is not worth it alone:** exact in exact arithmetic
  (b_k·const ∈ span) but only cut the DGKS rate from 89% to 86%.
- Smaller: ~244 sgemm + accumulate launches per batch at n=5e5 (a
  strided-batched sgemm plus one reduction would do); a host sync mid
  `project_reduce` for the DGKS gate.

## 2. CUDA follow-ups

The GPU hinge sweep landed on 2026-11-03 (CLAUDE.md note of that date): 76 ms
per X column at n=5e5, m=100 (28x one CPU core) and 386 ms at m=250. Pass 1
(carries) is at ~1.9x the f64 FMA floor and the grid scoring adds ~12 / 31 ms;
the f64-only levers are spent. Open items, in order:

1. **Blocked f32 for pass 1 -- the remaining factor of several, and a numerics
   decision (not yet taken).** With the carry defined at the end of step b-1,
   pass 1 is plain sums over the chunk (`g_loc = Σ_t b_t·v_t`, `f_loc = Σ_t
   b_t·v_t·D(t,b-1)` with per-chunk suffix sums of `d` as weights), i.e. a
   batched GEMM over pre-permuted rows that could run as f32 (64× the f64
   rate on the sm_120 box) with f64 cross-chunk accumulation, like the T
   GEMM. Within-chunk f32 error of ~sqrt(L)·eps_f32 on O(sqrt(L/n)) partials
   is ~1e-7..1e-9 relative on the O(1) carries, against the 1e-6 relative
   `HINGE_DEN_REL` gate: near-gate scores would move at the 1e-3..1e-1 level,
   the same order the GPU orthonormalize's f32 projection already introduces
   (~5e-4 relative). Build it opt-in (`MARS_CUDA_SWEEP_F32=1`), measure speed
   (pass 1 from ~27 ms to a few ms at m=100), and run the same fit-quality
   study the T GEMM got (R², basis selection on well-conditioned, correlated
   and collinear stresses) before considering a default. Pass 2's chunk-local
   part could follow (its local contribution is O(L/n) of the carried state,
   so f32 there is ~1e-10 relative), which would make the whole sweep
   tensor-core shaped.
2. **Profile a whole hinge fit** (`MARS_CUDA_PROFILE=1 python3
   tests/bench_eval.py --cuda --n 500000 --p 20 --max-terms 101 --repeat 1`)
   and attack what dominates there. Candidates now that an eval is <0.1 s:
   `append()` (28 ms at m=100, CPU classical GS), `context_sync_basis` (a host
   gather plus a strided row-major scatter per appended column, 308 ms over
   the 100-column bench), and the per-column serialization -- the cuda path
   evaluates X columns one `eval()` at a time on one stream, so for small m
   the GPU is underfilled; sweeping several X columns per launch (grid.z) or
   overlapping the CPU-side work of column i+1 with the GPU's column i would
   help there.
3. **Deterministic orthonormalize reductions** (`round_reduce_kernel` f64
   atomics): GPU results vary at the ULP between runs, which forces every
   downstream test to use tolerances instead of equality. A per-tile partials
   buffer plus a fixed-order second pass; cheap.
4. Cache the sort permutation (and its inverse) per X column across epochs;
   it never changes within a fit (sort+prep is 1.5 ms per eval -- low priority).
5. `m + 1 + 2·JB > 1024` lanes throws; lift by instantiating more lanes per
   thread. The 8-columns-per-block variant at four lanes per thread spills
   (only reachable with `MARS_CUDA_SWEEP_JB=8` above m≈500).
6. Smaller orthonormalize items from the 2026-06-17 profile (GPU phases were
   flat: T GEMM 28%, round/norm 24%, fill 23%, DGKS 19%, projection 6%):
   - **`fill` f64-write drop (~7%).** `fill` writes both an f32 and an f64
     `Bx`; on the blocked-f32 path the f64 copy is read only by `round_reduce`
     as the minuend of the fused subtract. Reading the f32 `Bx` instead needs
     the f32 projection in its own buffer (it currently overwrites `dBx_f32`).
     Bit-identical result, delicate aliasing change, low ROI.
   - **`context_sync_basis` host gather.** The new `Bo` column is gathered on
     the host from the row-major `Bo` (one cache line per row); `append()` has
     the same column contiguous as f64 `v` just before scattering it. Staging a
     contiguous f32 copy on `MarsData` and uploading that avoids the gather.
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

**Value:** since the closed-form linear ΔSSE (2026-11-04), Phase 1 is nearly
the whole cost of CPU `linear_only`; and, since the sweep fix, ~45% of a CPU
hinge eval (1.1 of 2.4 s at m=100,
5.8 of 12.5 s at m=250). Expected ~2× on Phase 1/2a, so ~1.3× on a hinge fit.

## 4. CPU hinge sweep: f/g store traffic (low priority)

The sweep runs at its compute floor since the 2026-10-22 prefetch fix (1.3 s at
n=5e5, m=100 vs a ~0.9 s floor for 2 loads, 1 widening load, 2 FMAs and 2
stores per 4 lanes). What is left is the store traffic of `f`/`g` (one load +
one store each per FMA). Amortizing it across 2 sorted rows per step is
possible algebraically (`f += (d1+d2)·g + d2·b1·x1`) but changes the f64
summation order and needs the on-grid positions handled exactly.

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
  the failure chain items 5.1–5.2 below target (nothing forbids a new knot next to
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
