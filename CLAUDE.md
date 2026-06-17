# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with
code in this repository.

## Build & Test Commands

The build is driven by `CMakeLists.txt`. The `Makefile` is a thin wrapper.

```bash
make              # configure (build/) + build; drops marslib*.so at the source root
make test         # build and run C++ unit tests, then pytest
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
  intrinsics in the inner loop (`covariates()`).
- `append()` — commits a chosen basis (type `'l'`, `'+'`, or `'-'`), updates `Bo` and `ybo`.
- `nbasis()` / `dsse()` / `yvar()` — read-only accessors.

**pybind11 bindings (`marslib.cc`)** — exposes `MarsAlgo` as `marslib.MarsAlgo`.
The `eval()` binding parallelizes over `X` columns using OpenMP, releasing the
GIL. It accepts a boolean mask matrix `(p, m)` controlling which `(input, basis)`
pairs to evaluate (Fast-MARS cache).

**Python orchestration (`mars.py`)** — implements the full MARS forward pass loop
on top of `marslib`. Uses `numba` for `expand()`. Key public API:
- `fit(X, y, ...)` → structured numpy array `model` of basis nodes
- `expand(X, model)` → basis matrix `B` for a new dataset
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
`append()`'s Gram-Schmidt is now `mars::orthonormalize_col` (single-column
modified GS + DGKS), validated against an Eigen oracle in the tests. The `s`
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

**Performance note (2026-06-15) — bf16 basis storage REJECTED on CPU, do not
retry:** We built bf16 storage for `B`/`Bo`/`Bx` (storage only, widen to f32/f64
to compute) to halve the basis load volume. It caps at **parity** with f32, so
it's not in `master`. A SIMD bf16 widen-load fixed the initial scalar slowdown
(~1.4× → parity) but can't beat f32: with f64 accumulation the `T = Boᵀ·Bx` GEMM
is bound by f64 `T` traffic + 4-wide f64 FMA, not the `Bo`/`Bx` load. The ~2×
needs **f32 accumulation**, which is numerically unacceptable — `T` is full of
near-zero (cancelling) projection entries and f32 accumulation gives 1e-2…4.6e-1
relative error vs f64's ~1e-10 (persists even for bf16 inputs; f64 accumulation
is load-bearing). The CUDA port (TODO.md) is the real speedup. Archived on the
unmerged `refactor/bf16-basis-seam` / `feat/bf16-linear-only` branches.

**Performance note (2026-06-16) — CUDA `linear_only` forward pass (branch
`feat/cuda-orthonormalize`, opt-in `-DUSE_CUDA=ON` / `make configure-cuda`):** A
GPU `mars::cuda::orthonormalize` path behind `fit(cuda=True)`; the CPU kernel
stays the oracle. ~12× over native f64 at n=500k, m=400 on the sm_120 server
(RTX PRO 6000 Blackwell, ~96 GB, *weak* f64 ≈1.44 TFLOP/s — no f64-tensor accel,
which is why f32 matters). Wins: batch the X-columns into one set of GEMMs per
block, keep scaled `X` + `Bo`/`B` resident, and compute `ybx=Bxᵀy` on-device so
the n×p `Bx` never crosses PCIe. The lever past the f64 ceiling is **blocked-f32
GEMMs**: `T=Boᵀ·Bx` as chunked `cublasSgemm` with f64 cross-chunk accumulation
(gram()-style), and the projection `Bx-=Bo·T` in f32 — **ON by default**
(`MARS_CUDA_KCHUNK=2048`, `MARS_CUDA_PROJ_F32=1`); the full-f64 reference is
`MARS_CUDA_KCHUNK=0 MARS_CUDA_PROJ_F32=0`. The f32 path preserves fit *quality*
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
