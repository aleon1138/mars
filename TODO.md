# TODO

Ideas and notes for future performance work. Not planned for a specific release.

## BF16 support for X

**Goal:** ~2× speedup on the forward pass by halving memory bandwidth for the X
  matrix.

**Why it should help:** `covariates()` is ~1 FLOP/byte, which puts it near
  memory-bound territory at multi-core. Halving X bytes ≈ halving the load
  volume.

**Design:**
- bf16 only for X storage and as inputs to FMA. Accumulators stay fp32 — bf16's
  8-bit mantissa is not enough to accumulate n squared residuals without
  precision loss.
- Template `MarsAlgo` on input dtype (fp32 or bf16), dispatch from pybind based
  on numpy dtype.
- Most of the non-hot-loop code stays dtype-agnostic via templates. The real
  work is in `covariates()`.

**Complications:**
- numpy has no native bf16; need `ml_dtypes.bfloat16` or torch tensors on the
  Python side.
- Native bf16 FMA (`VDPBF16PS`) requires AVX-512 BF16 — i.e., Intel Sapphire
  Rapids+ or AMD Zen 4+. On older hardware, load bf16, convert to fp32 via
  widening cast, then normal FMA. Still wins on memory but not on compute.

## AVX-512 path

**When it makes sense:** only together with bf16, on hardware that supports
AVX-512 BF16 natively (e.g., EPYC 9845 / Zen 5c, Sapphire Rapids+). Plain
AVX-512 without bf16 is probably a wash because:
- Memory bandwidth is the real bottleneck at multi-core.
- Frequency downclocking on Skylake-X/Cascade Lake eats most of the width gain.
- AVX-512 absent from most consumer Intel (12th–14th gen).

**Design:**
- Keep current AVX2 kernel as the default path — portable and fast enough on
  most hardware.
- Add an AVX-512 + bf16 kernel behind `#ifdef __AVX512BF16__` or a runtime CPUID
  check.
- Repo stays public-friendly: anyone can compile with `-march=native` and get
  the best their CPU supports.

**Dev note:** i7-7820X (Skylake-X) has AVX-512F but no BF16 and heavy
  downclocking, so it's dev/test only. EPYC 9845 is where performance testing
  happens.

## CUDA port

**Realistic speedup:** 10–30× on the forward pass vs. current multi-core AVX2.
  Not 100× — MARS isn't pure matmul; there's sorting, dependency chains between
  epochs, and control flow for cut-point selection. Your current code already
  hits decent CPU throughput.

**Why it could still be worth it:**
- 96GB VRAM is enough to keep the entire working set (X, B, Bo, y, internals)
  resident — no PCIe transfers during a fit. That alone could be 2–3× if
  current runs are memory-bandwidth-limited.
- Tensor cores eat bf16×bf16 → fp32 natively, which is exactly the precision
  regime we want.

**Design:**
- Port incrementally. Keep CPU orchestration, move only the hottest inner loop
  (`covariates()`) to CUDA first. Measure before doing more.
- `covariates()` maps well to segmented scan primitives (CUB / thrust).
- Orthonormalization is straightforward cuBLAS.
- Per-epoch dependencies prevent parallelism across epochs, but within an epoch
  the (x_col, basis_col) candidate grid is embarrassingly parallel — same
  parallelism OpenMP is already exploiting.

**Effort estimate:** multi-week project, not a weekend.

## Replace Eigen with a custom mini-BLAS

**Goal:** drop Eigen entirely. Smaller binary, ~80% faster builds, no hidden
  temporaries, full control over allocation.

**Why it's tractable:** the hot loop (`covariates_impl`) is already
  hand-rolled AVX/FMA — Eigen isn't doing the heavy lifting there. The
  remaining ~70 Eigen touchpoints in `marsalgo.cc` are mostly book-keeping:
  views, slices, element-wise scaling, and norm reductions. There is exactly
  one BLAS-like op in the codebase: `Bx.transpose() * y` at line 387
  (matrix-vector product, p ≤ 400 columns), plus a couple of
  `colwise().squaredNorm()` calls at construction. A minimal BLAS providing
  GEMV, AXPY, dot/norm, and a few cwise kernels is enough.

**Why it's worth doing:**
- Killed the per-call temporary allocations problem at the source (see
  2026-05-16 perf investigation): no more reliance on jemalloc as a runtime
  prerequisite.
- Eigen's `<Eigen/Dense>` is ~80% of build time. Compiling marsalgo.cc drops
  from seconds to fractions of a second.
- All allocation lives in `MarsScratch` (already pooled on `MarsAlgo`),
  making the algorithm provably allocation-free on the hot path.
- Easier to reason about and to step through in a debugger.

**Why not Blaze:** same expression-template / hidden-temporary design as
  Eigen. Would not eliminate the underlying problem. The point of doing this
  is to *own* the linear algebra, not to swap dependencies.

**Design sketch:**
- `mars_la.h` — thin header with `Mat<T>`, `Vec<T>`, `View<T>` (non-owning
  pointer + shape + stride), and the handful of kernels we need.
- Replace `Map<>`, `Ref<>`, `Block<>` usage with plain pointer+stride views.
- Keep the column-major / row-major distinction explicit at the type level
  so we don't lose Eigen's compile-time correctness.
- One GEMV kernel, vectorized; everything else is straightforward loops.

**Effort estimate:** 2-3 weeks for a clean, tested replacement. Mostly
  mechanical once the view/kernel surface is settled. The risk is in subtle
  precision differences vs. Eigen's reductions — needs the
  `tests/repro_shuffle.py` row-order invariance check + a tight numerical
  comparison against the current implementation on a saved dataset.

# MARS improvements — TODO

Context for future-me: this is a roadmap of four improvements to the
`aleon1138/mars` codebase, ordered roughly from cheapest to most invasive.
The first three sit on top of the existing hinge-basis design; the fourth
(BMARS) replaces the internal linear algebra wholesale. They are independent
in principle — you can ship (1)–(3) without touching (4) — but (3) and (4)
compose particularly well.

The underlying motivation across all four is the same observation: stock MARS
becomes numerically unreliable at interaction order $\geq 3$ on correlated
features. The Gram matrix loses effective rank, GCV rankings become noisy,
and the backward pass cannot be trusted to remove redundant terms. Each item
below addresses some part of that failure chain.

---

## 1. Fix `endspan` logic

Aplied at the **boundaries** of the data: minimum gap between the
smallest/largest data point and the first/last candidate knot for each
variable.

### Why a separate parameter

Boundary knots are asymmetrically dangerous. A hinge with knot $t$ near $\min
(x_v)$ has support $[t, \infty)$ that extends past every data point — it will
dominate extrapolation at the boundary even though it's supported by very few
in-sample observations. The cost of a bad boundary knot is therefore much
higher than the cost of a bad interior knot, which justifies a separate
(and larger) gap.

### Friedman's default formula

$$
\text{endspan} = 3 - \log_2(\alpha / d)
$$

where $d$ is the number of input variables and $\alpha$ is the same small
probability as in `minspan` (default 0.05). Note this is a fixed constant plus
a $\log d$ correction — it does not depend on $N$. Typical values are in the
range 10–30.

Things to audit when you get back to this:

- Check that `endspan` scales with $d$ (number of input variables) per the
  formula, not just $N$ or constants.
- Check that `endspan` and `minspan` are **both** applied — the slice should
  be `sorted_x[endspan : N - endspan : minspan]`, not one or the other.

---

## 2. Knot tuning — 1D refinement after each forward pick

After the forward pass picks a knot $t^* \in \{\tau_i\}$ from the discrete
candidate grid, do a local 1D line search to refine $t^*$ to the
RSS-minimizing position in the open interval $(\tau_{i-1}, \tau_
{i+1})$. Brent's method or golden-section search both work; bracket is the two
adjacent grid points.

Friedman discussed and rejected knot tuning in 1991 on cost grounds — the
line search is $\mathcal{O}(N)$ per step and he wanted forward selection cheap.
On modern hardware (especially with cached partial residuals from the fast-MARS
framework already being used), each line search is sub-millisecond even for $N
= 10^6$. The cost calculus is inverted.

Bakin/Hegland/Osborne (2000) re-investigated this in the BMARS paper and
reported it improves the forward-pass quality. They have a figure
"Results of experiment with knot tuning procedure" in the published version.

After the discrete pick, you have:

- The chosen variable $v$ and the chosen grid knot $\tau_i$.
- The current basis $\{B_1, \ldots, B_M\}$ and the residual vector $r$ from
  the previous step.
- Cached $\langle B_j, B_{M+1}(t)\rangle$ for the candidate basis function
  $B_{M+1}(t) = (\text{parent}) \cdot (x_v - t)_+$.

Define $\rho(t) = \text{RSS}$ when $B_{M+1}(t)$ is added to the model with
its optimal least-squares coefficient. This is a smooth function of $t$ on
$(\tau_{i-1}, \tau_{i+1})$ (the breakpoint at $\tau_i$ is removed when you
allow $t$ to move continuously). Run Brent on $\rho(t)$ in that interval.

Cost analysis: each evaluation of $\rho(t)$ at a new $t$ requires recomputing
the inner products $\langle B_j, B_{M+1}(t)\rangle$ for $j = 1, \ldots, M$
and $\langle y, B_{M+1}(t)\rangle$. That's $\mathcal{O}(MN)$ if done naively,
but only the data points in the support of $B_{M+1}(t)$ contribute and the
support changes incrementally with $t$ — so with sorted indexing you can
update incrementally in $\mathcal{O}(M)$ per Brent step. Brent typically
converges in 5–10 steps. Total added cost per forward step:
$\mathcal{O}(M)$ in the best case, $\mathcal{O}(MN/K)$ if you're lazy where
$K$ is the candidate grid size.

### Implementation order

Do this **after** (1) and (2) are working — knot tuning amplifies the
quality of the discrete pick, which itself depends on a clean candidate
filter. Wire in the line search as a refinement step that runs after the
discrete forward pass and before committing the basis function.

### Validation

Same Friedman-benchmark setup as before. With knot tuning on, the test-set
MSE should improve at fixed $M$ (model size). Equivalently, you should reach
the same MSE with fewer basis functions. The improvement is largest when
the underlying signal has knot-like discontinuities at non-grid positions
(easy to construct synthetically: $f(x) = \mathbb{1}[x > 0.4137]$ and let
the grid be aligned to quantiles that don't include 0.4137).

---

## 3. BMARS basis reformulation

### What it is

A change of basis that's **purely internal to the linear algebra**. The user
still sees a hinge-basis MARS model, the API and serialization are
unchanged, but the matrices you actually factor and update are formed in a
compactly-supported B-spline basis where the Gram is banded and well-
conditioned.

### Why this is the real fix

Items (1)–(3) reduce the *frequency* of conditioning failures. They don't
fix the underlying problem, which is that truncated power basis functions
$(x - t)_+$ have **unbounded support** — they grow without bound for
$x > t$. Every pair of such basis functions has substantial overlap in
their tails, and at high interaction order the joint support is enormous.

B-splines have **compact support** — they are exactly zero outside a bounded
interval. Two B-splines have non-zero inner product only if their supports
physically overlap. The Gram matrix becomes banded (1D) or sparse with
structured nonzero pattern (tensor product). Sparse Cholesky on the B-spline
Gram is numerically much better-behaved than dense Cholesky on the hinge
Gram, and the gap *widens* with interaction order rather than narrowing.

### The change-of-basis identity (the math)

For each variable $v$, set up an internal linear B-spline basis
$\{\phi_1^v, \ldots, \phi_{K_v}^v\}$ on a knot grid
$\tau_1^v < \tau_2^v < \cdots < \tau_{K_v}^v$. Choose the grid once at
startup as evenly-spaced quantiles of $x_v$. $K_v \in [64, 256]$ is plenty
for typical financial features.

A linear B-spline $\phi_i^v$ centered at $\tau_i^v$ is the tent function

$$
\phi_i^v(x) = \begin{cases}
(x - \tau_{i-1}^v) / (\tau_i^v - \tau_{i-1}^v) & \tau_{i-1}^v \leq x \leq \tau_i^v \\
(\tau_{i+1}^v - x) / (\tau_{i+1}^v - \tau_i^v) & \tau_i^v \leq x \leq \tau_{i+1}^v \\
0 & \text{otherwise}
\end{cases}
$$

The key identity: any hinge $(x_v - \tau_i^v)_+$ with knot at a grid point
can be written exactly as a linear combination of two adjacent B-splines
plus a global linear term:

$$
(x_v - \tau_i^v)_+ \;=\; \alpha_i \cdot x_v + \beta_i + \sum_{j \geq i} w_j^{(i)} \, \phi_j^v(x_v)
$$

The weights $w_j^{(i)}, \alpha_i, \beta_i$ are determined by matching values
at the grid points. Precompute these once at startup. Stack into a
lower-triangular banded matrix $W^v \in \mathbb{R}^{K_v \times K_v}$.

For tensor-product hinges (interactions), the joint expansion is the tensor
product of the univariate expansions. The change-of-basis matrix is
$W^{v_1} \otimes W^{v_2} \otimes \cdots \otimes W^{v_k}$, but you never
materialize the Kronecker product — apply the factors as a sequence of
banded matvecs.

### Algorithm

1. **Startup (once)**:
   - For each variable $v$: compute knot grid $\{\tau_i^v\}$ as quantiles.
   - Build $\Phi^v \in \mathbb{R}^{N \times K_v}$, the B-spline design matrix.
     Each row has at most 2 nonzeros; store sparse.
   - Compute $\Phi^{v\top}\Phi^v \in \mathbb{R}^{K_v \times K_v}$ — banded.
   - Compute the change-of-basis matrix $W^v$.

2. **Forward pass**:
   - Each existing hinge basis function $B_j$ is stored *additionally* as a
     coefficient vector $\beta_j$ in the (tensor) B-spline basis.
   - To evaluate a candidate hinge $B_{M+1}$: compute its $\beta_{M+1}$ via
     the change-of-basis identity (cheap, sparse).
   - Inner products: $\langle B_j, B_{M+1}\rangle = \beta_j^\top G \beta_{M+1}$
     where $G$ is the (precomputed, banded) B-spline Gram. Banded matvec.
   - Run the standard fast-MARS rank-1 update for the RSS reduction, but
     using these Gram entries instead of dense ones.
   - The internal Cholesky factor is on the B-spline Gram, not the hinge Gram.

3. **Backward pass / GCV**:
   - Unchanged in structure. The rank-1 downdates now operate on the
     well-conditioned B-spline Gram, so GCV rankings are trustworthy.

4. **Output**:
   - Return the model in hinge form for interpretability and API
     compatibility. The B-spline representation is purely internal.

### Why "minimum viable cut" matters

Don't try to do the full thing in one go. Stage it:

**Stage 4a — univariate, linear B-splines, snap-to-grid knots.**
~A few hundred lines. Replaces just the inner-product computation in the
forward pass for univariate (non-interaction) terms. Validate on the
Friedman benchmark — should match standard MARS to machine precision on
well-conditioned cases and outperform it on ill-conditioned ones.

**Stage 4b — extend to tensor-product interactions.**
The Kronecker matvec is the only really new piece. Cost per inner product
for an order-$k$ interaction is $\mathcal{O}(k \cdot K \cdot \text{nnz})$,
which beats the hinge version's dense $\mathcal{O}(N)$ for any $K \ll N$.

**Stage 4c — integrate with knot tuning (item 3).**
Knot tuning in the B-spline basis is a 1D line search over the
representation coefficients $\beta$, which is even cheaper than in the
hinge basis because the support is bounded.

**Stage 4d (optional) — quadratic B-splines for $C^1$ continuity.**
This recovers Friedman's cubic-smoothing benefit (BMARS as published is
purely $C^0$). A hinge isn't exactly representable in quadratic B-splines
on a finite grid, but the error is a piecewise-quadratic correction that
can be absorbed into the basis. Probably not worth doing until 4a–4c are
solid and you have a concrete need for $C^1$.

### Practical complications

**Knot grid resolution vs. accuracy.** Snapping candidate knots to a grid
of $K$ points gives up Friedman's continuous knot search. In practice this
is fine — Friedman's own fast-MARS does subsampling that's morally
equivalent — and item (3) above (knot tuning) gives you sub-grid resolution
back where it matters.

**Boundary handling.** Linear B-splines at the edges of the grid are
half-tents. Add an explicit linear term to the basis to handle extrapolation
cleanly, otherwise candidates near the boundary look weird.

**The $W^v$ change-of-basis matrices.** Precompute once. Watch for
off-by-one errors in the Kronecker composition for interactions — write
unit tests that verify $B_j(x) = \beta_j^\top \Phi(x)$ pointwise on a
random sample of $x$ values.

**Hinge-basis interpretability.** Don't lose this. Always store the hinge
representation as the primary; the B-spline coefficients are derived. When
serializing fitted models, serialize the hinges. When loading, recompute
the B-spline coefficients from the hinges.

### Stack-specific notes

- Blaze `CompressedMatrix` handles the banded B-spline Gram cleanly. The
  existing `LLT` factorization works without modification.
- The Kronecker matvecs for tensor-product interactions are inner-loop
  AVX-512 candidates. Sort the B-spline support by knot index so memory
  access is contiguous; this is essentially free cache-friendliness.
- The f64 accumulation refactor done for the cross-machine (i7-7820X vs
  EPYC 9845) divergence issue is more useful here than in the original code,
  because the hinge-basis Gram was throwing away precision before the
  accumulation step — so f64 accumulation was a partial fix. With B-spline
  basis the conditioning win compounds with the accumulation precision win.
- BLIS gives the dense ops for the final QR/Cholesky on the active model;
  banded structure mostly bypasses BLAS in favor of hand-rolled banded
  kernels. Worth profiling.

### Validation

Three regimes to test:

1. **Well-conditioned, low-order**: fit standard Friedman benchmark with
   $k=2$ interactions. B-spline backend should match hinge backend to
   machine precision on test MSE.

2. **Ill-conditioned, high-order**: construct a synthetic with $k=4$
   interactions on highly correlated features (e.g., features generated as
   $x_i = z + \epsilon_i$ for shared $z$). Hinge backend's GCV path should
   show the characteristic noisy-plateau-then-spike pattern as conditioning
   fails. B-spline backend's GCV path should be smooth and monotone in the
   forward phase.

3. **Real data**: pick one of the binance execution-cost surfaces. Compare
   hinge backend vs B-spline backend on test-set MSE and on the stability
   of the fitted model under bootstrap resampling. The B-spline backend
   should produce models that are more stable across bootstrap samples
   (fewer different basis functions selected), which is the practical
   manifestation of better conditioning.

---

## Implementation order

1. **Item 1 (`minspan`)** — easiest, biggest first improvement, catches
   regressions in the candidate-generation code path. Do this first.
2. **Item 2 (`endspan` audit/fix)** — do alongside item 1 since they share
   the candidate-slice logic.
3. **Item 3 (knot tuning)** — depends on (1) and (2) being correct.
   Independent of item 4.
4. **Item 4a (univariate BMARS)** — biggest win, but most invasive.
   Validate against items 1–3 in well-conditioned regime first.
5. **Item 4b (tensor-product BMARS)** — extension of 4a.
6. **Item 4c (knot tuning in B-spline basis)** — composition step.
7. **Item 4d (quadratic B-splines)** — only if needed.

Items 1, 2, 3 can each ship as standalone improvements with their own
benchmarks. Item 4 is a multi-month project; stage it carefully and keep
the hinge-basis code path working as the validation oracle.

---

## References

- Friedman, J.H. (1991), "Multivariate Adaptive Regression Splines",
  *Annals of Statistics* 19(1):1–141. The original paper. `minspan` and
  `endspan` formulas are in §3. Knot tuning is discussed and rejected
  in §3.5 on cost grounds.

- Friedman, J.H. (1993), "Fast MARS", Stanford Tech Report LCS110.
  Cached partial residuals; the rank-1 update framework you're already
  using.

- Bakin, S., Hegland, M., Osborne, M.R. (2000), "Parallel MARS Algorithm
  Based on B-splines", *Computational Statistics* 15:463–484. The BMARS
  paper. Has the knot-tuning experiment for B-spline basis.

- For the change-of-basis identity between truncated powers and B-splines:
  de Boor, "A Practical Guide to Splines" (revised ed., 2001), §IX.
  Standard reference; the identity is classical.

# Smaller items / known issues

Collected during the DGKS re-orthogonalization review (2026-05-17 session).
Each item is independent and small; none of them block correctness today.

## `eval()`'s bootstrap covariates_impl call passes the wrong `ym`

**Location:** `marsalgo.cc:484`, the first call to `covariates_impl<true>` before
the cut sweep loop.

**Issue:** The `ym` parameter is hardcoded to `ybx[0]` but should be `ybx[j]`
(the projection of the candidate column being swept). The call is tagged
`<true>` so it pays the cost of computing `o.ff`/`o.fy`, but the return value
is discarded. Currently harmless because `k0 = 0` keeps `f[m]` zero and the
SSE math reads from `o` only in the main loop body.

**Fix:** Switch to `<false>` (saves the per-call SSE arithmetic) and either
fix `ybx[0]` to `ybx[j]` or drop the argument since the result is unused.
One-line change.

## `append()` recomputes the full `ybo` projection on every call

**Location:** `marsalgo.cc:586-587`.

**Issue:**
```cpp
VectorXd ybo = _data->Bo.leftCols(_m+1).transpose() * _data->y.matrix();
```
This is an O(n·m) matvec, but only the new last entry changed; the leading
m entries are still valid in `_data->ybo`. Wasted work proportional to model
size.

**Fix:** Compute only the new entry and append it:
```cpp
const double new_ybo = _data->Bo.col(_m).dot(_data->y.matrix());
const double mse = (1. - _data->ybo.squaredNorm() - new_ybo*new_ybo) / n;
if (mse >= -_tol) {
    _data->ybo.conservativeResize(_m+1);
    _data->ybo[_m] = new_ybo;
    ...
}
```

## `prune()` is O(M⁴)

**Location:** `mars.py:515-525`, the backward-elimination loop.

**Issue:** Each elimination step calls `_gcv_sse` for every still-active term,
and each `_gcv_sse` solves a fresh `np.linalg.lstsq` from scratch. Total cost
is O(M⁴) where M is the final basis count. For M=30 it's ~14k full solves
(seconds); M=100 is hours.

**Fix:** Maintain a Cholesky (or QR) of `XX[active, active]` and do rank-1
downdates when removing a column. Same algorithm, ~M× faster.

## Python scalar loop builds `bmask` each epoch

**Location:** `mars.py:263-265`.

**Issue:**
```python
for i in input_to_use:
    bmask[i] &= np.array([basic_filter(i, b) for b in basis])
    bmask[i] &= np.array([aux_filter(i, b) for b in basis])
```
Two Python list comprehensions per input per epoch. For `len(input_to_use)`
× `len(basis)` in the hundreds-to-thousands, this is the dominant cost on
the Python side.

**Fix:** Precompute once per epoch:
- `degree = np.array([len(b) for b in basis])` (M,)
- `contains[i,k] = (i in basis[k])` (p, M) -- bool sparse, O(p·M)

Then:
```python
bmask &= (degree[None, :] < max_degree) & (~contains | self_interact)
```
Vectorized in numpy. `aux_filter` is user-supplied so it has to stay scalar,
but the basic filter is the hot one.

## `argsort` redone on every `eval()` call

**Location:** `marsalgo.cc:429`.

**Issue:** Per-xcol sort order depends only on `X`, which is immutable for the
lifetime of `MarsAlgo`. Yet `argsort()` runs again every epoch for every
candidate column.

**Constraint:** A full `p × n` int32 cache is 4·p·n bytes -- 2.4 GB for the
n=6M, p=100 case. Can't just cache everything.

**Fix:** A Fast-MARS LRU sized to the current `max_inputs` cap covers the
inputs we actually re-evaluate. Eviction by age matches Friedman's Fast-MARS
recipe.

## Counter cache-line layout

**Location:** `marsalgo.h`, the `_dgks_count` atomic member.

**Issue:** `_dgks_count` lives adjacent to `_scratches`, `_tol`, etc. on the
`MarsAlgo` instance. Under heavy OMP parallelism inside `eval()`, a DGKS
fetch_add from one worker invalidates the shared cache line for every other
worker that's reading `_tol` (which they do, on every column). Probably
negligible today because DGKS fires rarely, but if benchmarks ever show OMP
scaling regressions on ill-conditioned data, separating the counter onto its
own cache line (alignas(64), or a padded wrapper) is the fix.

## Mac libomp double-link

**Issue:** On Apple Silicon with `conda activate work`, running `mars.fit()`
multi-threaded crashes:
```
OMP: Error #15: Initializing libomp.dylib, but found libomp.dylib already
initialized.
```
This is the well-known Apple libomp + conda libomp link conflict, not
specific to this codebase. Workaround:
```
KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 python ...
```
Single-threaded works fine. Real fix is to make the build link against
exactly one libomp (either system or conda) -- not urgent since the
production target is Linux.
