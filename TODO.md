# TODO

Open ideas for future performance and quality work. None of them is planned
for a specific release. This file holds only open work: completed work is in
the git history, and the measurements, lessons and rejected approaches from
that work are in the dated notes in docs/notes.md. Items are in order of expected
value per unit of risk, as of the 2026-11-04 review. Unless noted otherwise,
timings are single-threaded on the EPYC 9845 server (`tests/bench_sweep`,
n=5e5, pinned to one core). Read the benchmarking note in docs/notes.md before you
trust any A/B.

## 0. Where we left off (2026-11-04)

The 2026-11-04 round reworked `linear_only` end to end (commits
00e4870..4bbb549, and the docs/notes.md notes of that date). Its changes were the
residual target, the f64 restart of the GPU DGKS retry, closed-form scoring,
the register-tiled Phase 1, the CPU and GPU T caches, the threaded `append()`
and dead-on-append. Whole linear-only fits went from:

- GPU, n=5e5, p=50, 53 terms: 5.12 → 0.70 s
- CPU, n=2e5, p=16, 72 terms, 16 threads: 1.37 → 0.54 s

The models are identical before and after. Every test passes on the sm_120
box (14 `test_cuda`, 23 `unittest`, 42 pytest).

Suggested next steps, in order:

1. **Profile a whole hinge fit** on the CPU and the GPU with
   `tests/bench_eval.py` (`--cuda`, `MARS_CUDA_PROFILE=1`). See §2 item 2. A
   hinge fit (`linear_only=False`) is the normal MARS use. That round left it
   alone, apart from ~10% from the Phase 1 tiling and the threaded `append()`.
   The profile decides between the two big hinge-path items: blocked-f32
   pass 1 of the GPU sweep (§2 item 1) and CPU Phase 2a tiling (§3, needs a
   quiet box).
2. The small `linear_only` items in §1: the f64 ΔSSE on the uncached paths
   (#3), the GPU in-span margin test (#2), the GPU basis upload.
3. On a quiet box: §3, and re-check the 12×4 vs 8×4 Phase 1 tile.

Workflow reminders:

- The sandbox has no GPU. Compile CUDA code here (a build dir under the
  session scratchpad, `--target test_cuda`). The user runs the device tests
  and benches.
- Run CPU A/B tests in a scratch copy of the tree, not by editing the sources
  in place.
- Write commit messages to a file for `git commit -F`.
- Keep the CPU and GPU paths separate (CLAUDE.md, Architecture).
- `tests/bench_linear.py` A/Bs a `linear_only` fit with and without the T
  cache (CPU or `--cuda`) and checks that the models match.

## 1. Linear-only scoring: numerics and speed (2026-11-04 review)

A numpy model of both paths found these items. It compared each path against
an f64 ground truth on the same f32 data: a fit-like basis and correlated
features with mean/std = 3, at n=2e4, m=120 and at n=6e3, m=400. The GPU
numbers come from an emulation with OpenBLAS sgemm, not from the device.
κ = ‖Bx‖ / ‖residual‖.

**Numerics.** Item #1 (the residual target) and item #2 (the GPU DGKS retry
restarting from the f64 `B·x`) landed on 2026-11-04 (see docs/notes.md).

2. **Exactly in-span candidates on the GPU: thin margin.** They still score
   0. But in the numpy model their residual reached 0.4·tol at m=400, n=6e3.
   At small n, `tol` sits at its f32 floor, and the noise of the f32
   projection grows as √m. Add a duplicate-candidate case at m≈1000 and small
   n to `test_cuda`.
3. **f32 normalize scale.** On the uncached explicit paths, the f32 normalize
   scale adds ±1.2e-7 relative noise to the ΔSSE of each candidate. These
   paths are the CPU with `MARS_LINEAR_CACHE_GB=0` and, on the GPU,
   `orthonormalize_batch` and the per-column `eval(cuda=True)`. The fix is
   free: compute `ΔSSE = <r,y>²/(s+tol)` in f64, as the fallback of the cached
   paths already does.

**Speed**

- **Where a cached GPU `linear_only` fit spends its time** (n=5e5, p=50, 53
  terms, 0.70 s). No single item dominates.
  - Outside eval, ~0.38 s: the threaded `append()` and the Python loop.
  - Inside eval: the basis upload (next item) 68–130 ms, fresh slots 74 ms,
    row extension 52 ms, the explicit fallback 43 ms and scoring 2 ms. The
    fallback ran on 28 candidates. All of them were dead: products reached in
    two orders.
- **GPU `context_sync_basis` gathers each new `Bo` column on one thread.** It
  reads down the row-major host `Bo` with a stride, then syncs. That costs
  ~1.3–2.5 ms per append at n=5e5 (68–130 ms per fit, depending on the run).
  Fix it inside the CUDA path, because the CPU and GPU paths stay separate
  and the CPU `append()` must not carry GPU work. For example, use a pinned
  staging buffer that several host threads fill, or one `cudaMemcpy2DAsync`
  straight from the strided host column.
- **CPU follow-up: batch the Phase 1 columns of fresh parents.** The Phase 1
  column of a newly requested parent is a single column, so it takes the
  store-bound axpy loop. The fresh parents of several X columns share `Bo`.
  Batching them into one tiled pass would cut the remaining per-epoch cost.
- **Centering x is not worth it on its own.** It is exact in exact arithmetic
  (b_k·const ∈ span), but it only cut the DGKS rate from 89% to 86%.
- Smaller items:
  - At n=5e5, a batch makes ~244 sgemm + accumulate launches. A
    strided-batched sgemm plus one reduction would do.
  - `project_reduce` syncs with the host midway for the DGKS gate.

## 2. CUDA follow-ups

The GPU hinge sweep landed on 2026-11-03 (see the docs/notes.md note of
that date). It takes 76 ms per X column at n=5e5, m=100 (28x one CPU core) and 386
ms at m=250. Pass 1 (the carries) runs at ~1.9x the f64 FMA floor. The grid
scoring adds ~12 ms at m=100 and ~31 ms at m=250. The f64-only levers are
spent. Open items, in order:

1. **Blocked f32 for pass 1.** This is where the remaining several-fold
   speedup is, and it needs a numerics decision that we have not made yet.
   With the carry defined at the end of step b-1, pass 1 is plain sums over
   the chunk: `g_loc = Σ_t b_t·v_t` and `f_loc = Σ_t b_t·v_t·D(t,b-1)`, with
   per-chunk suffix sums of `d` as weights. That is a batched GEMM over
   pre-permuted rows. Like the T GEMM, it could run in f32 (64× the f64 rate
   on the sm_120 box) with f64 accumulation across chunks.

   The f32 error within a chunk is ~sqrt(L)·eps_f32 on partials of
   O(sqrt(L/n)). On the O(1) carries that is ~1e-7..1e-9 relative, against
   the 1e-6 relative `HINGE_DEN_REL` gate. So near-gate scores would move at
   the 1e-3..1e-1 level. That is the same order that the f32 projection in
   the GPU orthonormalize already introduces (~5e-4 relative).

   To do:
   - Build it opt-in (`MARS_CUDA_SWEEP_F32=1`).
   - Measure the speed. Pass 1 should go from ~27 ms to a few ms at m=100.
   - Run the fit-quality study that the T GEMM got: R² and basis selection
     on well-conditioned, correlated and collinear stresses.
   - Only then consider making it the default.

   The chunk-local part of pass 2 could follow. Its local contribution is
   O(L/n) of the carried state, so f32 there is ~1e-10 relative. That would
   make the whole sweep tensor-core shaped.
2. **Profile a whole hinge fit** and attack what dominates it. On the GPU:

   ```bash
   MARS_CUDA_PROFILE=1 python3 tests/bench_eval.py --cuda --n 500000 --p 20 --max-terms 101 --repeat 1
   ```

   For the CPU, run the same command without `--cuda`. Now that an eval takes
   <0.1 s, the candidates are:
   - `append()`. It has been threaded since 2026-11-04: ~1.5 ms per call at
     n=5e5, m≈50 on 16 threads, vs ~17 ms on one.
   - `context_sync_basis` (see §1).
   - The per-column serialization. The cuda path evaluates the X columns one
     `eval()` at a time on one stream, so at small m the GPU is underfilled.
     Two things would help there: sweeping several X columns per launch
     (grid.z), or overlapping the CPU-side work of column i+1 with the GPU's
     column i.
3. **Deterministic orthonormalize reductions.** The f64 atomics in
   `round_reduce_kernel` make GPU results vary at the ULP between runs. So
   every downstream test must use tolerances instead of equality. The fix is
   cheap: a per-tile partials buffer plus a second pass in fixed order.
4. Cache the sort permutation (and its inverse) per X column across epochs.
   It never changes within a fit. This is low priority: sort+prep is 1.5 ms
   per eval.
5. The sweep throws when `m + 1 + 2·JB > 1024` lanes. Lift the limit by
   instantiating more lanes per thread. The 8-columns-per-block variant spills
   at four lanes per thread, but only `MARS_CUDA_SWEEP_JB=8` above m≈500
   reaches it.
6. Smaller orthonormalize items from the 2026-06-17 profile. The GPU phases
   were flat: T GEMM 28%, round/norm 24%, fill 23%, DGKS 19%, projection 6%.
   - **Drop the f64 write in `fill` (~7%).** `fill` writes both an f32 and an
     f64 `Bx`. On the blocked-f32 path, only `round_reduce` reads the f64
     copy, as the minuend of the fused subtract. To read the f32 `Bx`
     instead, the f32 projection needs its own buffer (today it overwrites
     `dBx_f32`). The result is bit-identical, but the aliasing change is
     delicate and the return is low.
   - **`context_sync_basis` host gather.** See §1. Fix it on the CUDA side:
     a contiguous copy staged by `append()` would break the CPU/GPU split.
   - **DGKS (19%).** This is genuine f64 re-orthogonalization of columns that
     really are near-collinear. Cutting it further needs an algorithmic
     change: re-orthogonalize against only the relevant `Bo` columns. That is
     high risk for the slice that remains.

## 3. Register-tile Phase 2a (CPU hinge eval, re-measure on a quiet box)

Phase 1 (`T = Boᵀ·Bx`) has been register-tiled since 2026-11-04 (docs/notes.md),
which gave ~2× on `eval(linear)`. Phase 2a (`project_out`: `Bx -= Bo·T` plus
the norms) is still the row-at-a-time `dot_bo4` loop. Only the hinge eval and
the fallback of the closed form use it. The loop is load-bound at ~1.25 loads
per FMA.

The same treatment applies with the roles swapped. For each output tile of 4
rows × 3 columns, keep 12 YMM accumulators along k (7 loads per 12 FMAs). Do
the horizontal sum at the end, and the subtract, round and norm as now.

The expected gain is ~10% of a hinge eval. That is too small to see while the
EPYC box is under heavy shared load (±30% run to run on 2026-11-04). When the
box is quiet, measure it, and re-check the choice of the 12×4 vs the 8×4
Phase 1 tile (~10–15%, measured under that load).

## 4. CPU hinge sweep: f/g store traffic (low priority)

The sweep has run at its compute floor since the 2026-10-22 prefetch fix: 1.3
s at n=5e5, m=100, against a ~0.9 s floor for 2 loads, 1 widening load, 2 FMAs
and 2 stores per 4 lanes. What is left is the store traffic of `f` and `g`
(one load and one store each per FMA). The algebra allows amortizing it
across 2 sorted rows per step (`f += (d1+d2)·g + d2·b1·x1`). But that changes
the f64 summation order, and it must handle the on-grid positions exactly.

## 5. MARS algorithmic improvements

Motivation: on correlated features, stock MARS becomes numerically unreliable
at interaction order ≥ 3. The Gram matrix loses effective rank, the GCV
rankings get noisy, and the backward pass cannot be trusted to remove
redundant terms.

The 2026-10-22 review found two concrete data points. Re-check the symptom
against them before investing here:

- Whenever the cut passed below a small-support parent, the hinge sweep *was*
  selecting cuts that are numerically in span (den/‖h+‖² ~ 1e-11). Their
  spurious gain reached 1e-2, above the legitimate cuts. Order-≥3 parents have
  the smallest support, so they were hit most. The sweep now gates these cuts
  (docs/notes.md, 2026-10-22). Re-run the failing cases first, because part of
  the symptom may be gone.
- A plain hinge fit (n=2e5, p=40, 41 terms, then `prune`) kept four hinges of
  the same input on one parent, at adjacent tail knots (1.83, 1.92, 2.01,
  2.07), with coefficients −21, +165, −350, +211. Different *epochs* placed
  these knots `min_span` apart on the same (input, parent), and together they
  fit a wiggle in a 2% tail. Items 5.1–5.2 below target this failure chain.
  Nothing forbids a new knot next to an existing one on the same input and
  parent, and `end_span` is a fraction of n, not Friedman's formula.

### 5.1 Fix `endspan`

`endspan` applies at the boundaries of the data. It is the minimum gap, per
variable, between the smallest (largest) data point and the first (last)
candidate knot. Boundary knots are asymmetrically dangerous. A knot near
min(x) has support that reaches past every data point, so it dominates
extrapolation while few observations back it. Friedman's default is:

$$\text{endspan} = 3 - \log_2(\alpha / d)$$

Here $d$ is the number of inputs, and $\alpha$ is the same small probability
as in `minspan` (0.05). The result is a constant plus a $\log d$ correction.
It does not depend on $N$ and is typically 10–30 samples. Today `tail_span` is
a *fraction* of n (2%, which is 10k rows at n=5e5). The audit:

- Scale `endspan` with $d$ per the formula.
- Apply `endspan` and `minspan` together
  (`sorted_x[endspan : N - endspan : minspan]`).
- Consider a minimum gap to the knots that the model already has on the same
  (input, parent).

### 5.2 Knot tuning: 1-D refinement after each forward pick

After the forward pass picks $t^*$ from the discrete grid, refine it with a
line search (Brent or golden section) for the position in
$(\tau_{i-1}, \tau_{i+1})$ that minimizes the RSS. Friedman rejected this in
1991 because of the cost. With cached partial residuals, each evaluation is
an $O(M)$ incremental update, and Brent converges in 5–10 steps. So the
refinement takes under a millisecond even at $N = 10^6$. Bakin, Hegland and
Osborne (2000) report that it improves forward-pass quality.

Validate it on a signal with a knot at a position off the grid, for example
$f(x) = 1[x > 0.4137]$ with a quantile grid that misses it. Success is the
same test MSE with fewer terms. Do this after 5.1.

### 5.3 BMARS basis reformulation

This is a change of basis inside the linear algebra only. The user still sees
hinge terms, and the API and serialization do not change. But the code forms
the matrices that it factors and updates in a compactly supported linear
B-spline basis per variable: tent functions on a quantile knot grid,
$K_v \in [64, 256]$. In that basis the Gram is banded and well conditioned.

Any hinge at a grid knot is an exact combination of two adjacent B-splines
plus a global linear term (de Boor §IX). So a precomputed banded change of
basis $W^v$ maps hinges to B-spline coefficients. Tensor-product interactions
apply the Kronecker factors as banded matvecs and never materialize them.
Items 5.1–5.2 make conditioning failures *less frequent*. This item removes
their cause, the unbounded support of hinges.

This is a multi-month project. Keep the hinge path as the oracle throughout,
and stage the work:

- (a) Univariate linear B-splines with snap-to-grid knots, replacing only the
  inner products of the forward pass. This stage must match hinge MARS to
  machine precision on well-conditioned data.
- (b) Tensor-product interactions. The Kronecker matvec is the only new piece.
- (c) Knot tuning in the B-spline basis.
- (d) Optionally, quadratic B-splines for $C^1$.

Design rules:

- Keep the hinge representation primary: store and serialize hinges, and
  derive the B-spline coefficients from them.
- Add an explicit linear term, so that the boundary half-tents extrapolate
  cleanly.
- Unit-test $B_j(x) = \beta_j^\top \Phi(x)$ pointwise.

Validate in three regimes:

- Well-conditioned, low order: expect a match.
- Ill-conditioned, order 4, on shared-factor features: expect the hinge GCV
  path to plateau and spike while the B-spline path stays smooth.
- A real surface under bootstrap: expect more stable selected terms.

## References

- Friedman, J.H. (1991), "Multivariate Adaptive Regression Splines",
  *Annals of Statistics* 19(1):1–141. §3 covers `minspan` and `endspan`.
  §3.5 discusses knot tuning and rejects it because of the cost.
- Friedman, J.H. (1993), "Fast MARS", Stanford Tech Report LCS110. Cached
  partial residuals, and the rank-1 update framework that this code uses.
- Bakin, S., Hegland, M., Osborne, M.R. (2000), "Parallel MARS Algorithm Based
  on B-splines", *Computational Statistics* 15:463–484. The BMARS paper, with
  the knot-tuning experiment.
- de Boor, "A Practical Guide to Splines" (revised ed., 2001), §IX: the
  truncated-power ↔ B-spline change of basis.
