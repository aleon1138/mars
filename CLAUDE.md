# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with
code in this repository.

## Build and test commands

The build is driven by `CMakeLists.txt`. The `Makefile` is a thin wrapper.

```bash
make              # configure (build/) + build; drops marslib*.so at the source root.
                  # GPU kernels are built in when nvcc + an NVIDIA GPU are detected
NOCUDA=1 make     # CPU-only build (skip the probe)
make cuda         # GPU kernels required (configure error without nvcc)
make test         # build and run C++ unit tests (+ test_cuda if built), then pytest
make clean        # remove build/ and the built .so (also needed to flip CPU <-> GPU)
make format       # format all .h/.cc files with astyle

pip install .     # alternative: build a wheel via scikit-build-core; same CUDA
                  # auto-detection, NOCUDA=1 for CPU-only, -C cmake.define.USE_CUDA=ON to require
```

To run a single GoogleTest: `./build/unittest --gtest_filter=MarsTest.DeltaSSE`

For memory and sanitizer debugging, reconfigure with sanitizer flags:
```bash
rm -rf build
cmake -S . -B build -DBUILD_TESTING=ON \
      -DCMAKE_BUILD_TYPE=Debug \
      -DCMAKE_CXX_FLAGS="-O0 -g -fsanitize=address"
cmake --build build --parallel
```

Build details:
- CMake's `USE_CUDA` is a tri-state cache string: `AUTO` (default), `ON` or
  `OFF`. `AUTO` runs `check_language(CUDA)` and `nvidia-smi -L`, and it builds
  the GPU kernels only if both succeed. `NOCUDA=1` in the environment makes
  `AUTO` CPU-only.
- The resolved boolean is the plain CMake variable `MARS_CUDA`. Test that,
  never `if(USE_CUDA)`, because `"AUTO"` is truthy.
- The default `CMAKE_CUDA_ARCHITECTURES` (`75;120`) is trimmed to what
  `nvcc --list-gpu-code` reports, so an older toolkit still builds. An explicit
  `-DCMAKE_CUDA_ARCHITECTURES` is never trimmed.
- CMake can find a different Python than the active environment. Then pytest
  imports a site-packages `marslib`, and `tests/conftest.py` stops the run.
  Configure with `-DPython_EXECUTABLE=$(which python3)` to fix it.
- Eigen is needed only for the tests: `find_package(Eigen3)` sits in the
  `BUILD_TESTING` block, so the library and the wheel build without it.

## Architecture

The library has a C++ core and a Python driver.

**C++ core (`marsalgo.h` / `marsalgo.cc`).** It implements `MarsAlgo`, a
stateful object that keeps the growing basis matrix during the forward pass.
The core is Eigen-free. All state is plain `std::vector` or raw pointers with
explicit BLAS-style strides. The state in `MarsData`:
- `X`: a read-only column-major (Fortran-order) pointer into the caller's
  data, with stride `ldX`.
- `B` / `Bo`: the current basis and its orthonormalized copy, kept in sync by
  Gram-Schmidt. `B` is column-major (stride `n`). `Bo` is row-major, and its
  row stride (`bo_stride`) grows geometrically ahead of the live basis count.
- `y`, `ybo`: the normalized target (f32 storage) and its projection onto
  `Bo`.
- `yres`: the residual target `y − Bo·ybo` (an f64 master plus an f32 copy,
  extended per `append()`). The linear ΔSSE is scored against it.
- `E`, `Eybo`: `BoᵀBo − I` (m×m f64, one row and column per `append()`) and
  `E·ybo`. They are the first-order Gram correction of the closed-form linear
  ΔSSE.
- `lin`: the CPU `linear_only` T cache, one `LinCache` per X column.
  `lin_cache` (the `MARS_LINEAR_CACHE_GB` budget check) enables it and,
  independently, the device cache that `eval_batch()` keeps in the CUDA
  context.
- `lin_x` / `lin_b`: which basis columns are linear terms. Both caches mark
  those candidates dead.
- `nnz`: the exact nonzero count of each `B` column (the parent's support for
  the hinge sweep).

These methods drive the algorithm:
- `eval()`: for each candidate `X` column, computes the ΔSSE of adding a
  linear term or a pair of mirror hinges (`+`/`-`). On the CPU, `linear_only`
  scores in closed form (`mars::linear_dsse`), from the T cache when it is
  enabled. The hinge sweep uses hand-written AVX2/FMA intrinsics
  (`covariates_impl()`). It sweeps `JB=4` basis columns per gathered `Bo` row
  and stops a column once the cut passes below its parent's support.
- `eval_batch()`: the GPU `linear_only` path, with every X column in one call.
- `append()`: commits a chosen basis (type `'l'`, `'+'` or `'-'`) and updates
  `Bo`, `ybo`, `yres` and `E`. Its row passes run over fixed 4096-row blocks,
  so the results are bit-identical for any thread count.
- `nbasis()` / `dsse()` / `yvar()`: read-only accessors.

`kernels.h` / `kernels.cc` hold the BLAS-style kernels behind `eval()` and
`append()`: `orthonormalize()`, the register-tiled Phase 1 (`T = Boᵀ·Bx`),
`linear_dsse()`, `orthonormalize_col()`, `store_col()` and `dot_widen()`.

**CPU and GPU paths stay separate.** The library must build and run on
machines without a GPU. GPU code lives in `cuda/` behind `MARS_USE_CUDA`.
Only the `cuda` branches of `eval()` and `eval_batch()` reach it. CPU
routines (`append()`, the CPU eval and its caches) carry no GPU work or
GPU-only state. Shared inputs live in `MarsData` (for example `lin_x` /
`lin_b`), and each path consumes them on its own. The GPU files:
- `cuda/mars_cuda.h`: the host API, free of CUDA types.
- `cuda/context.cuh`: the `Context` definition and shared reductions.
- `cuda/orthonormalize.cu`: the orthonormalize, with the blocked-f32 `T` GEMM
  and the f64 DGKS retry.
- `cuda/hinge_sweep.cu`: the hinge sweep as a two-pass chunked scan.
- `cuda/linear_cache.cu`: the device T cache behind `eval_batch()`.

**pybind11 bindings (`marslib.cc`).** They expose `MarsAlgo` as
`marslib.MarsAlgo`. The `eval()` binding parallelizes over `X` columns with
OpenMP and releases the GIL. It takes a boolean mask `(p, m)` that selects
which `(input, basis)` pairs to evaluate (the Fast-MARS cache). With
`cuda=True` it runs on one host thread, and with `linear_only` it calls
`eval_batch()`. The module also has the free function `marslib.expand()`, the
row-blocked OpenMP kernel behind `mars.expand()`.

**Python orchestration (`mars.py`).** It implements the full MARS forward-pass
loop on top of `marslib` (pure numpy/scipy, no numba). The public API:
- `fit(X, y, ...)` → structured numpy array `model` of basis nodes. The
  `fit()` docstring lists the fields.
- `expand(X, model, order="F", threads=-1)` → basis matrix `B` for a new
  dataset, in C or Fortran order, computed in parallel by `marslib.expand()`.
- `prune(B, y, ...)` → backward pass / GCV pruning. It returns a coefficient
  vector, usually stored back as `model["beta"] = prune(...)`.
- `compact(model)` → drop pruned nodes (`model["beta"] == 0`) and remap the
  parent pointers.
- `pprint(model)` → human-readable model formula, with coefficients from
  `model["beta"]`.

**Data requirements:** `X` must be `float32`, **column-major** (Fortran order).
`y` and `w` must be `float32` column-major 1D arrays. The bindings assert these
layouts explicitly.

## Rules from past work

`docs/notes.md` holds the dated notes: the measurements, the reasoning and the
rejected experiments. Before you change the hinge sweep, the T caches, the
CUDA kernels, `append()` or `expand()`, read the note for that area. Open work
is in `TODO.md`.

Numerics:
- f64 accumulation is load-bearing. Never narrow the sweep accumulators
  `f`/`g` to f32, and never accumulate `T = Boᵀ·Bx` in f32. Both corrupt the
  results through cancellation.
- The closed-form linear ΔSSE takes `T` errors at first order. So it needs an
  f64 `T`, and the GPU's blocked-f32 `T` must not feed it.
- On the explicit path (projection, then a dot), score the linear ΔSSE
  against `yres`, not `y`. `Bo` is f32, so `BoᵀBo = I + E`, and scoring
  against `y` lets `E` in at first order. The closed form handles `E`
  explicitly.
- Accuracy oracles project onto span(`Bo`), read through
  `MarsAlgo::ortho_basis()`, not onto span(`B`). The two differ by `Bo`'s f32
  rounding, which swamps the effects under test.
- Validate numerics on the x86/AVX server. The scalar macOS path has looser
  tolerances and once hid a real bug.
- Validate sweep numerics on continuous data, not only on quantized columns
  with ties. A/B the GPU sweep against the CPU sweep on the same device `Bx`
  (`MARS_CUDA_SWEEP=0`). The f32 projection in the GPU orthonormalize alone
  moves near-gate scores by up to ~5e-4 relative.
- GPU results vary at the ULP between runs (f64 atomics in the orthonormalize
  reductions). Anything downstream of them is tested with tolerances, not
  bit equality.
- The GPU fit is not bit-identical to the CPU fit. With the blocked-f32 `T`,
  near-tied terms on correlated features can come out in a different order.

Rejected approaches. Do not retry them:
- `#pragma omp simd` instead of the AVX2 intrinsics in `covariates()`: ~22%
  slower.
- AVX-512: no gain in `covariates_impl()`, and 15–30% slower for the Phase 1
  tile.
- f32 `f`/`g` in the hinge sweep: it selects wrong knots.
- bf16 storage of `B`/`Bo`/`Bx`: parity with f32 at best. The ~2× would need
  f32 accumulation, which `T` forbids.
- cuBLAS FP64 emulation: accurate, but ~30× slower for these skinny GEMMs.

Performance checks:
- After you touch the hinge sweep, check
  `objdump -d marslib*.so | grep -c prefetcht0`. GCC 13 deletes a loop of
  `__builtin_prefetch` calls inside a lambda, which is why `prefetch_line()`
  is inline asm.
- Benches: `tests/bench_sweep.cc` (the `bench_sweep` target) times `append()`,
  `eval(linear)` and `eval(hinge)`, and it can dump the outputs for a
  bit-exact A/B. `tests/bench_eval.py` is the whole-fit bench (`--cuda`,
  `--linear-only`). `tests/bench_linear.py` A/Bs the T cache, and
  `tests/bench_expand.py` times `expand()`.

Benchmarking on the shared EPYC server:
- Timings vary by ±20% between cores and between concurrent runs. That is
  enough to invert small A/B results.
- Build each variant as its own binary, pin it with `taskset -c <core>`, and
  run the variants back to back on that core. Diff the `bench_sweep` dump
  files for bit-exactness.
- Keep the bench basis well conditioned. Random hinges on random parents make
  the ΔSSEs noise-dominated.
- Make sure the bench fires the DGKS retry, or it measures a path that
  production never takes. Random-`Bo` benches never fire it. Uncentered
  features (mean/std = 3) fire it on 66–89% of candidates.

## Environment variables

| Variable | Effect |
|---|---|
| `MARS_LINEAR_CACHE_GB` | Budget for the `linear_only` T caches, CPU and GPU (default 8, 0 disables them). |
| `MARS_CUDA_KCHUNK` | Rows per K-chunk of the blocked-f32 `T` GEMM (default 2048). 0 selects the native-f64 reference. |
| `MARS_CUDA_BLOCK_GB` | Scratch budget of the batched GPU path (default 8). |
| `MARS_CUDA_SWEEP` | 0 keeps the GPU orthonormalize but runs the sweep on the CPU. |
| `MARS_CUDA_SWEEP_CHUNK`, `_TILE`, `_BLOCK`, `_JB`, `_BATCH` | Shape of the GPU sweep: rows per chunk, rows per shared-memory tile, block size, columns per block (4 or 8), grid slots per reduction batch. The tests set `_BLOCK=32` to force several lanes per thread. |
| `MARS_CUDA_PROFILE=1` | Per-phase GPU times at context destroy, and the eval/orchestration split of `fit()`. |
| `MARS_CUDA_VERBOSE=1` | Diagnostics on stderr, such as the chosen sweep shape and the T cache allocation. |
