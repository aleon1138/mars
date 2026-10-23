# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with
code in this repository.

## Build & Test Commands

The build is driven by `CMakeLists.txt`. The `Makefile` is a thin wrapper.

```bash
make              # configure (build/) + build; drops marslib*.so at the source root
make test         # build and run C++ unit tests (+ test_cuda if built), then pytest
make clean        # remove build/ and the built .so
make format       # format all .h/.cc files with astyle

pip install .     # alternative: build a wheel via scikit-build-core
```

For memory/sanitizer debugging, reconfigure with sanitizer flags:
```bash
rm -rf build
cmake -S . -B build -DBUILD_TESTING=ON \
      -DCMAKE_BUILD_TYPE=Debug \
      -DCMAKE_CXX_FLAGS="-O0 -g -fsanitize=address"
cmake --build build --parallel
```

To run a single GoogleTest: `./build/unittest --gtest_filter=MarsTest.DeltaSSE`

## Architecture

The library has two layers:

**C++ core (`marsalgo.h` / `marsalgo.cc`)** — implements `MarsAlgo`, a stateful
object that maintains the growing basis matrix during the forward pass. The
core is Eigen-free: all state below is plain `std::vector`/raw pointers with
explicit BLAS-style strides (see the 2026-06-09 note). Key internal state (in
`MarsData`):
- `X` — read-only column-major (Fortran-order) pointer + stride `ldX` into the caller's data
- `B` / `Bo` — the current basis and its ortho-normalized counterpart (kept in sync via Gram-Schmidt). `B` is column-major (stride `n`); `Bo` is row-major with a row stride (`bo_stride`) that holds the live basis count and grows geometrically (see the 2026-06-09 note)
- `y`, `ybo` — normalized target and its projection onto the ortho-basis

Three methods drive the algorithm:
- `eval()` — for each candidate `X` column, computes the delta-SSE improvement
  from adding a linear term or a pair of mirror hinges (`+`/`-`). Uses AVX/FMA
  intrinsics in the inner loop (`covariates()`); sweeps `JB=4` basis columns per
  gathered `Bo` row and stops a column once the cut passes below its parent's
  support (every later hinge is then in the span -- see the 2026-10-22 note).
- `append()` — commits a chosen basis (type `'l'`, `'+'`, or `'-'`), updates `Bo` and `ybo`.
- `nbasis()` / `dsse()` / `yvar()` — read-only accessors.

**pybind11 bindings (`marslib.cc`)** — exposes `MarsAlgo` as `marslib.MarsAlgo`.
The `eval()` binding parallelizes over `X` columns using OpenMP, releasing the
GIL. It accepts a boolean mask matrix `(p, m)` controlling which `(input, basis)`
pairs to evaluate (Fast-MARS cache). The module also carries the free function
`marslib.expand()`, the row-blocked OpenMP kernel behind `mars.expand()` (see
the 2026-10-23 note).

**Python orchestration (`mars.py`)** — implements the full MARS forward pass loop
on top of `marslib` (pure numpy/scipy, no numba). Key public API:
- `fit(X, y, ...)` → structured numpy array `model` of basis nodes
- `expand(X, model, order="F", threads=-1)` → basis matrix `B` for a new
  dataset, in C or Fortran order, computed in parallel by `marslib.expand()`
- `prune(B, y, ...)` → backward pass / GCV pruning (returns a coefficient vector,
  typically stored back as `model["beta"] = prune(...)`)
- `compact(model)` → drop pruned nodes (those with `model["beta"] == 0`), remap
  parent pointers; reads/writes the coefficient via the model's `beta` field
- `pprint(model)` → human-readable model formula (coefficients from `model["beta"]`)

**Performance note (2026-04-12):** We benchmarked replacing the hand-written AVX
intrinsics in `covariates()` with `#pragma omp simd`. The pragma version was ~22%
slower (0.93 vs 1.14 ms/call, pinned core, `-O3 -march=native`).
`-mprefer-vector-width=256` did not help. The hand-written intrinsics stay.

**Performance note (2026-05-16):** We benchmarked widening `covariates_impl()`
to AVX-512 (`__m512d`, 8-wide doubles). No measurable speedup on either
Skylake-AVX512 or EPYC 9845 (Zen 5, full-width 512-bit FPU). The hot loop is
not FMA-bound at typical m (~50–100) — likely limited by L1 load/store ports
and per-call setup. The AVX2 intrinsics stay.

**Performance note (2026-06-02):** We narrowed the target `_data->y` from f64 to
f32 storage and replaced the Eigen `Bx.col(j).cast<double>().dot(...)` in the
eval() linear-delta-SSE loop with a hand-written AVX2 kernel `mars::dot_widen`
(`cvtps_pd` widening + f64 FMA, two accumulators). ~10% faster on `linear_only`
fits (x86 AVX2 server); the gain is diluted in full hinge fits where the linear
dot is only ~1/m of eval(). The inputs are upcast at the load, so the dot still
accumulates in f64 — no accuracy change (DeltaSSE moves only at ~1e-13). The
speedup is consistent with Eigen scalarizing the size-changing f32→f64 cast
inside the reduction; the packed widening recovers it. (The one-shot strided
`Bo.col()` ybo dots in the constructor and `append()` were later moved off
Eigen too — see the 2026-06-09 note below.)

**Architecture note (2026-06-09):** Removed Eigen entirely from the shipped
library. `MarsData` (`X`/`B`/`Bo`/`y`/`ybo`/`s`) and the per-thread `eval()`
scratch are now `std::vector`/raw pointers with explicit strides. `B` is
preallocated column-major (stride `n`, no per-append resize); `Bo` is row-major
with a row stride (`bo_stride`) decoupled from the live basis count and grown
geometrically (double, capped at `max_terms`) via an in-place back-to-front
repack (`restride_rows`). This keeps the random Bo-row gather in the hinge sweep
near the live count while dropping the repack cost from O(n·m²) over a fit to
O(n·m) — the sweep takes the stride (`ldBo`) and the column count (`m`)
separately, so it never assumes `stride == count`.
`append()`'s Gram-Schmidt is now `mars::orthonormalize_col` (single-column GS +
DGKS; modified GS at the time, classical since 2026-10-22 -- see that note),
validated against an Eigen oracle in the tests. The `s`
column norms were widened f32→f64 (the prior Eigen `colwise().squaredNorm()`
accumulated in f32). `marsalgo.cc`/`marslib.cc`/`kernels.{h,cc}` are Eigen-free;
`find_package(Eigen3)` is scoped to the `BUILD_TESTING` block, so a library/wheel
build doesn't need Eigen. Eigen survives only as the unit-test oracle.

**Performance note (2026-06-09) — REJECTED, do not retry:** We tried narrowing
the hinge-sweep accumulators `f`/`g` in `covariates_impl()` from f64 to f32
(f32 8-wide recurrence, reductions still f64) to halve the load/store traffic on
the bandwidth-bound off-grid path. It corrupts hinge-cut selection and must not
be done. The per-cut score is `sse = uw² / den` with `den = (k0+k1) − o.ff`, a
cancellation: once `f` is f32, `o.ff = Σf[i]²` carries f32-level relative error,
and at marginal/extreme cuts where `den` is small that error is amplified into
spurious ΔSSE peaks that win the argmax → wrong knots. Keeping the reductions in
f64 does **not** save it — `f` itself being f32 is the source. On the x86/AVX
server the chosen-cut ΔSSE was off ~53% (2.36e-4 vs 1.54e-4 oracle) and `MinSpan`
localized a planted knot at −0.96 instead of 0.25. (The macOS *scalar* path
masked this — it stayed under the 1e-5 scalar `HINGE_DSSE_TOL`; only the AVX
server at 1e-6 exposed it, so validate numerics on the server.) `f`/`g` stay f64.

**Performance note (2026-06-15) — bf16 basis storage REJECTED, a performance
dead end, do not retry:** We built bf16 storage for `B`/`Bo`/`Bx` (storage only,
widen to f32/f64 to compute) to halve the basis load volume. It caps at
**parity** with f32 on CPU, so it's not in `master`. A SIMD bf16 widen-load
fixed the initial scalar slowdown (~1.4× → parity) but can't beat f32: with f64
accumulation the `T = Boᵀ·Bx` GEMM is bound by f64 `T` traffic + 4-wide f64 FMA,
not the `Bo`/`Bx` load. The ~2× needs **f32 accumulation**, which is numerically
unacceptable — `T` is full of near-zero (cancelling) projection entries and f32
accumulation gives 1e-2…4.6e-1 relative error vs f64's ~1e-10 (persists even for
bf16 inputs; f64 accumulation is load-bearing). The dead end is fundamental, not
CPU-specific: any bf16 scheme only pays off if it accumulates in ≤f32, and the
cancelling `T` forbids that. By the same argument the obvious GPU move should
fail too — bf16 tensor cores accumulate in f32 at best — but that's predicted
from the numerics, not benchmarked (only the CPU experiment and the GPU
FP64-emulation path were actually measured). That precision contract is why the
CUDA lever is blocked-**f32** GEMMs with f64 cross-chunk accumulation, not bf16
(2026-06-16 CUDA note). The CUDA port (TODO.md) is the real speedup. Archived on
the unmerged
`refactor/bf16-basis-seam` / `feat/bf16-linear-only` branches.

**Performance note (2026-06-16) — CUDA `linear_only` forward pass (branch
`feat/cuda-orthonormalize`, opt-in `-DUSE_CUDA=ON` / `make configure-cuda`):** A
GPU `mars::cuda::orthonormalize` path behind `fit(cuda=True)`; the CPU kernel
stays the oracle. ~12× over native f64 at n=500k, m=400 on the sm_120 server
(RTX PRO 6000 Blackwell, ~96 GB, *weak* f64 ≈1.44 TFLOP/s — no f64-tensor accel,
which is why f32 matters). Wins: batch the X-columns into one set of GEMMs per
block, keep scaled `X` + `Bo`/`B` resident, and compute `ybx=Bxᵀy` on-device so
the n×p `Bx` never crosses PCIe. The lever past the f64 ceiling is **blocked-f32
GEMMs**: `T=Boᵀ·Bx` as chunked `cublasSgemm` with f64 cross-chunk accumulation
(gram()-style), and the projection `Bx-=Bo·T` in f32 (`MARS_CUDA_KCHUNK=2048`); 
the full-f64 reference is `MARS_CUDA_KCHUNK=0`. The f32 path preserves fit *quality*
(R² bit-identical to f64 on well-conditioned, correlated, and 0.98-collinear
stresses) but **reorders near-tied ΔSSE terms** on correlated features, so the
GPU fit is not bit-reproducible vs the CPU there (equal quality, different
ordering).

**Performance note (2026-06-16) — cuBLAS FP64 emulation REJECTED, do not retry:**
We wired the f64 GEMMs through `cublasGemmEx` with
`CUBLAS_COMPUTE_64F_EMULATED_FIXEDPOINT` (env-gated `MARS_CUDA_FP64_EMULATE=1`)
and benchmarked it. On sm_120 it **engages** and is **accurate** (passes the
cancellation gate — DGKS + collinear/degenerate at 1e-5), but it is **~30×
slower** for our small/skinny GEMMs (`T` is ~100×100 out, K=n large): the Ozaki
fixed-point packing/scaling/per-call workspace malloc dwarfs the matmul, which
targets *large dense* f64 GEMMs. The toggle has since been removed from
`context_create()` — the blocked-f32 split-K path above is the correct lever;
emulation is the wrong tool for these shapes.

**Review pass (2026-10-22) — numerics + CPU performance:**
- **Hinge sweep had a real f32 failure mode, now gated.** Once a cut passes below
  the parent's support, `h+ = b·x − h·b` lies exactly in span(Bo, Bx), so
  `den = ‖h+‖² − ‖proj h+‖²` is 0 in exact arithmetic and the computed value is
  noise of either sign at ~eps_f32/√n·‖h+‖² (the orthonormality defect of the
  f32 `Bo`; the two sums it differences are O(‖h+‖²)). `den > _tol` (∝ n·eps_f64)
  let positive noise through, and `uw²/den` — noise over noise — scored up to
  1e-2, above every legitimate cut (seen on 2 of 100 bases in `bench_sweep` at
  n=5e5; such cuts *would have been selected*). Fix: a relative gate
  `den > HINGE_DEN_REL·‖h+‖²` (8·eps_f32 ≈ 1e-6) plus **support tracking** (each
  basis column keeps its exact nonzero count `nnz`; the sweep stops a column
  when all its nonzero rows are above the cut). Results are bit-identical at
  every non-degenerate cut. This is a plausible contributor to the "unreliable
  at order ≥ 3" symptom in TODO.md (small-support parents hit it most).
- **`_tol` floored at 8·eps_f32²** (`degeneracy_tol()`): below n≈1000 the
  n-scaled term sat under the f32 rounding noise of an exactly degenerate
  candidate (a duplicate input scored ~1/n at n=300 and `append()` accepted it
  as a basis). Nothing changes above n≈25k. Covered by
  `MarsTest.DegenerateCandidateIsZeroAtSmallN`.
- **`append()` is 13–16× faster**: classical GS (two streaming sweeps over the
  row-major `Bo`) replaces modified GS, whose per-column walk touched one cache
  line per row per column. 172→14 ms at n=5e5, m=100; 447→29 ms at m=250. It
  had become the dominant per-epoch cost on the GPU path (where eval is ~0.1 s).
  Same DGKS gate; with `Bo` orthonormal to f32 precision the two agree to that
  floor (the test cross-checks both references).
- **Sweep 3.3–3.6× faster, bit-exact** (4.6→1.3 s at n=5e5, m=100; 21.9→6.7 s
  at m=250; a 16-thread hinge fit of n=2e5, p=40, 41 terms went 19→9 s):
  JB=4 columns per gathered `Bo` row,
  full-row + gather prefetch, 64-byte-aligned `f`/`g`, forced inline of
  `covariates_impl`. Most of the gain came from discovering that **the sweep
  had no software prefetch at all**: GCC 13 deletes a loop of
  `__builtin_prefetch` calls inside a lambda (a single call or a plain loop
  survives; clang keeps every form), and the pre-review code only ever
  prefetched the first line of each row. `prefetch_line()` is inline asm on
  x86 so it cannot be dropped; the sweep now runs at the compute floor measured
  by a standalone microbench of the same loop. Bisection also showed the
  per-row gathers of `b`/`bx`/`y` are the dominant stall once the row is
  prefetched (without their prefetch the sweep is 2.5× slower), while the
  on-grid work, support bookkeeping and scalar accumulators cost <5% together.
  **Check `objdump -d marslib*.so | grep -c prefetcht0` after touching the
  sweep.**
- Smaller: `argsort` is a stable LSD radix sort (51→~5 ms at n=5e5; the
  planned LRU cache is moot); per-thread `Bx` is sized by the active columns
  `p` rather than `m` (Fast-MARS memory); `fit()` caches `max(sse1, sse2)`
  (with `linear_only` or no admissible cut, `sse2` is 0 and the Fast-MARS
  ranking was running on all-zero scores); the appended linear column is
  formed as `b·(x·s)` like the candidate, so it is bit-identical to what was
  scored; `threads` is clamped to ≥1; the MXCSR FTZ/DAZ state is restored by
  RAII; CUDA f64 GEMMs call `cublasDgemm` directly (the emulation compute-type
  plumbing is gone).
- `tests/bench_sweep.cc` (`bench_sweep` target under `BUILD_TESTING`) times
  `append()`, `eval(linear)` and `eval(hinge)` at a fit-like basis and dumps the
  outputs for bit-exact A/B. On the EPYC box, pin with `taskset` and run
  variants sequentially on the same core: concurrent runs on different cores
  disagree by ±20%.

**Performance note (2026-10-23) — `expand()` is a row-blocked C++ kernel:**
`mars.expand()` used to build `B` column by column in numpy on one core (three
temporaries per hinge column, so ~4 passes of memory traffic per column). It
now calls `marslib.expand()`, which splits the rows into blocks sized so all
`m` columns of a block fit in ~256 KB of L2 and walks the nodes in index order
inside each block. Because `basis[i] < i`, every parent block is still in
cache when its children are formed, so the parent/child dependency costs no
second pass and no synchronization: the OpenMP loop is over independent row
blocks, and the DRAM traffic is one read of the X columns the model uses plus
one write of `B`. `X` and `B` are taken with arbitrary strides, so `X` may be
C- or F-ordered and `B` is returned in whichever `order` the caller asked for.
A column-major `B` is filled in place with a contiguous, vectorized column
loop. A row-major `B` is computed by the same loop into a per-thread
column-major tile and scattered out in 16×16 sub-blocks (`transpose_tile`);
computing straight into the strided rows was ~3× slower per thread whether
walked by column (a cache-line fill per element) or by row (scalar, ~4
cycles/element), and the scalar transpose costs ~1 cycle/element -- an AVX
8×8 shuffle transpose is the obvious next step if single-thread C-order ever
matters. Arithmetic is in X's dtype (f32 or f64) with the cut rounded to that
dtype first, matching how `append()` forms the column `fit()` scored; the
pytest suite checks bit-identity against a dtype-faithful numpy reference.
Thread count follows `eval()` (all cores capped at 64, clamped to the block
count), and the main thread polls for Ctrl-C every 256 blocks.
`tests/bench_expand.py` times the old numpy loop against the kernel at several
thread counts in both orders (fresh `B` per call, so page faults are included).
On the EPYC box at n=4e6, m=100, p=16, f32 (`B` = 1.6 GB):

| threads | numpy loop | order="F" | order="C" |
|--------:|-----------:|----------:|----------:|
|       1 |    0.78 s  |   0.157 s |   0.274 s |
|       8 |            |   0.027 s |   0.036 s |
|      16 |            |   0.022 s |   0.018 s |
|      32 |            |   0.020 s |   0.014 s |

Past ~16 threads both orders sit on the memory/page-fault wall (73–114 GB/s
written); the F-order plateau is lower, presumably because each thread writes
`m` separate 2.5 KB column segments per block rather than one contiguous run.

**Data requirements:** `X` must be `float32`, **column-major** (Fortran order).
`y` and `w` must be `float32` column-major 1D arrays. The bindings assert these
layouts explicitly.

**Model node dtype** (returned by `fit`):
```
type   S1   — b'i' intercept, b'l' linear, b'+'/b'-' hinges
basis  i4   — index of parent basis node
input  i4   — column index into X
hinge  f8   — hinge cut point (NaN for linear/intercept)
beta   f4   — fitted coefficient; NaN until prune() fills it in
r2     f4
r2_cv  f4   — GCV-adjusted R²
order  i4   — polynomial degree
time   f4
```
