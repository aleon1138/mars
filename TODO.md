# TODO

Ideas and notes for future performance work. Not planned for a specific release.

## BF16 basis storage — EXPLORED AND ABANDONED (2026-06-15)

We built and benchmarked bf16 storage for the basis matrices `B`/`Bo`/`Bx`
(storage only; everything widens to f32/f64 to compute) to cut the load volume
of the bandwidth-bound forward pass. **Conclusion: not worth it on CPU — it caps
at parity with f32.** The real speedup is the CUDA port below.

**Why it caps at parity (don't retry the f32-accum shortcut):**
- A SIMD bf16 widen-load brought bf16 `orthonormalize()` from ~1.4× *slower*
  (scalar) to **parity**, but no faster: with f64 accumulation the `T = Boᵀ·Bx`
  GEMM is bound by f64 `T` traffic (load+store every FMA) + 4-wide f64 FMA, so
  halving only the `Bo`/`Bx` *load* bytes can't win.
- Getting the ~2× requires **f32 accumulation**, and that is numerically
  **rejected**: `T = Boᵀ·Bx` is full of near-zero (cancelling) projection
  entries, and f32 accumulation destroys them — measured 1e-2…4.6e-1 relative
  error vs f64's ~1e-10, and it persists even for bf16 inputs (f64-accum of bf16
  inputs is exact). The f64 accumulation is load-bearing. (This is a different
  mechanism from the 2026-06-09 `f`/`g` rejection — cancellation in the GEMM, not
  the hinge denominator.) A non-cancelling dot (e.g. the throwaway `bf16_bench.c`)
  *does* hit 2× in f32, but that regime doesn't match the GEMM.
- The only f64-preserving 2× would be a register-tiled, i-blocked GEMM (see the
  store-port-bound note below) — deferred in favour of CUDA, which wins by far.

**Where the work lives:** unmerged branches `refactor/bf16-basis-seam` (no-op
widen/store seam) and `feat/bf16-linear-only` (templated `MarsAlgo<BT>` +
`IMarsAlgo` base + runtime `basis_dtype` flag + `degeneracy_tol<BT>()`). Kept as
an archived record; `master` is bf16-free. The de-Eigen seam refactor there is
reusable if a future storage-dtype need (e.g. *memory* footprint, not speed)
ever arises.

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

## Phase 1 GEMM (`T = Boᵀ·Bx`) is store-port-bound

**Finding:** Phase 1 of `orthonormalize()` (the `axpy_m` sweep building
  `T = Boᵀ·Bx`) is limited by store-port throughput, not DRAM bandwidth or FMA.
  It writes a `T` element on ~every FMA, and x86 retires only ~1 store/cycle.
  Keeping `i` outermost gives the *optimal* DRAM pattern (`Bo`/`Bx` read exactly
  once), but then `T` (m×p f64, ~80 KB at m=p=100) can't live in registers, so
  the per-FMA store is unavoidable.

**Fix:** A register-tiled, `i`-blocked GEMM — hold a `T` tile in YMM
  accumulators across an `i`-panel, and block `i` into L2-sized chunks so the
  `Bo`/`Bx` panels are re-read from L2 (cheap) instead of re-storing `T` from
  the hot loop. That removes the store-port bottleneck.

**Why it can't just call BLAS:** `Bo`/`Bx` are f32 storage but `T` must
  accumulate in f64 (the precision contract behind narrowing the big buffers).
  `sgemm` accumulates in f32; `dgemm` needs f64 inputs. The f32-in/f64-accumulate
  combo has no library primitive — hence the hand-rolled widening kernels.

**Why deferred:** ~100 lines of intricate AVX with three levels of edge handling
  (i/k/j tails) plus hardware-specific tile tuning (i-panel size for L2, the
  register tile). The AVX-512 experiment (see CLAUDE.md, 2026-05-16) showed these
  kernels go port/setup-bound, so the win may evaporate.

**Before investing:** profile Phase 1's actual share of `eval()` on the EPYC
  server — the hinge sweep and Phase 2a are comparable O(n·p·m) costs, so Phase 1
  may not dominate. Only worth the complexity if it's a hot fraction. If you do
  it, validate the AVX path + tune tiles on the server; it can't be exercised on
  the arm64 dev box.

# MARS improvements — TODO

Context for future-me: this is a roadmap of three improvements to the
`aleon1138/mars` codebase, ordered roughly from cheapest to most invasive.
The first two sit on top of the existing hinge-basis design; the third
(BMARS) replaces the internal linear algebra wholesale. They are independent
in principle — you can ship (1)–(2) without touching (3) — but (2) and (3)
compose particularly well.

The underlying motivation across all three is the same observation: stock MARS
becomes numerically unreliable at interaction order $\geq 3$ on correlated
features. The Gram matrix loses effective rank, GCV rankings become noisy,
and the backward pass cannot be trusted to remove redundant terms. Each item
below addresses some part of that failure chain.

---

## 1. Fix `endspan` logic

Applied at the **boundaries** of the data: minimum gap between the
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

Do this **after** (1) is working — knot tuning amplifies the quality of
the discrete pick, which itself depends on a clean candidate filter. Wire
in the line search as a refinement step that runs after the discrete
forward pass and before committing the basis function.

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

Items (1)–(2) reduce the *frequency* of conditioning failures. They don't
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

**Stage 3a — univariate, linear B-splines, snap-to-grid knots.**
~A few hundred lines. Replaces just the inner-product computation in the
forward pass for univariate (non-interaction) terms. Validate on the
Friedman benchmark — should match standard MARS to machine precision on
well-conditioned cases and outperform it on ill-conditioned ones.

**Stage 3b — extend to tensor-product interactions.**
The Kronecker matvec is the only really new piece. Cost per inner product
for an order-$k$ interaction is $\mathcal{O}(k \cdot K \cdot \text{nnz})$,
which beats the hinge version's dense $\mathcal{O}(N)$ for any $K \ll N$.

**Stage 3c — integrate with knot tuning (item 2).**
Knot tuning in the B-spline basis is a 1D line search over the
representation coefficients $\beta$, which is even cheaper than in the
hinge basis because the support is bounded.

**Stage 3d (optional) — quadratic B-splines for $C^1$ continuity.**
This recovers Friedman's cubic-smoothing benefit (BMARS as published is
purely $C^0$). A hinge isn't exactly representable in quadratic B-splines
on a finite grid, but the error is a piecewise-quadratic correction that
can be absorbed into the basis. Probably not worth doing until 3a–3c are
solid and you have a concrete need for $C^1$.

### Practical complications

**Knot grid resolution vs. accuracy.** Snapping candidate knots to a grid
of $K$ points gives up Friedman's continuous knot search. In practice this
is fine — Friedman's own fast-MARS does subsampling that's morally
equivalent — and item (2) above (knot tuning) gives you sub-grid resolution
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

1. **Item 1 (`endspan` audit/fix)** — easiest first improvement; catches
   regressions in the candidate-generation code path. Do this first.
2. **Item 2 (knot tuning)** — depends on (1) being correct. Independent
   of item 3.
3. **Item 3a (univariate BMARS)** — biggest win, but most invasive.
   Validate against items 1–2 in well-conditioned regime first.
4. **Item 3b (tensor-product BMARS)** — extension of 3a.
5. **Item 3c (knot tuning in B-spline basis)** — composition step.
6. **Item 3d (quadratic B-splines)** — only if needed.

Items 1 and 2 can each ship as standalone improvements with their own
benchmarks. Item 3 is a multi-month project; stage it carefully and keep
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

## Python scalar loop builds `bmask` each epoch

**Location:** `mars.py:306-308` (the `for i in input_to_use:` loop in `fit()`).

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

**Location:** the `argsort()` call in `MarsAlgo::eval()` (`marsalgo.cc`).

**Issue:** Per-xcol sort order depends only on `X`, which is immutable for the
lifetime of `MarsAlgo`. Yet `argsort()` runs again every epoch for every
candidate column.

**Constraint:** A full `p × n` int32 cache is 4·p·n bytes -- 2.4 GB for the
n=6M, p=100 case. Can't just cache everything.

**Fix:** A Fast-MARS LRU sized to the current `max_inputs` cap covers the
inputs we actually re-evaluate. Eviction by age matches Friedman's Fast-MARS
recipe.

