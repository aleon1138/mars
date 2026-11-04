# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with
code in this repository.

## Build & Test Commands

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

**Build note (2026-10-23):** CMake's `USE_CUDA` is a tri-state cache string,
`AUTO` (default) / `ON` / `OFF`. `AUTO` runs `check_language(CUDA)` and
`nvidia-smi -L` and builds the GPU kernels only if both succeed; `NOCUDA=1` in
the environment short-circuits it to CPU-only. The resolved boolean is the
plain CMake variable `MARS_CUDA`; test that, never `if(USE_CUDA)` (`"AUTO"` is
truthy). The default `CMAKE_CUDA_ARCHITECTURES` (`75;120`) is trimmed to what
`nvcc --list-gpu-code` reports so an older toolkit still builds; an explicit
`-DCMAKE_CUDA_ARCHITECTURES` is never trimmed.

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
- `yres` — the residual target `y − Bo·ybo` (f64 master + f32 copy, extended per
  `append()`): the linear ΔSSE dot target (see the 2026-11-04 note)
- `E`, `Eybo` — `BoᵀBo − I` (m×m f64, one row/column per `append()` via
  `mars::store_col`) and `E·ybo`: the first-order Gram correction of the
  closed-form linear ΔSSE (2026-11-04 performance note)
- `lin` — the CPU `linear_only` T cache, one `LinCache` per X column
  (2026-11-04 note). `lin_cache` (the `MARS_LINEAR_CACHE_GB` budget check)
  enables it and, independently, the device cache `eval_batch()` keeps in the
  CUDA context (`cuda/linear_cache.cu`); `lin_x` / `lin_b` record which basis
  columns are linear terms (both caches mark those candidates dead)

Three methods drive the algorithm:
- `eval()` — for each candidate `X` column, computes the delta-SSE improvement
  from adding a linear term or a pair of mirror hinges (`+`/`-`). Uses AVX/FMA
  intrinsics in the inner loop (`covariates()`); sweeps `JB=4` basis columns per
  gathered `Bo` row and stops a column once the cut passes below its parent's
  support (every later hinge is then in the span -- see the 2026-10-22 note).
- `append()` — commits a chosen basis (type `'l'`, `'+'`, or `'-'`), updates `Bo`, `ybo`,
  `y_res` and `E` (row passes threaded over fixed blocks; 2026-11-04 note).
- `nbasis()` / `dsse()` / `yvar()` — read-only accessors.

**CPU and GPU paths stay separate.** The library must build and run on
machines without a GPU. GPU code lives in `cuda/` behind `MARS_USE_CUDA` and
is reached only from the `cuda` branches of `eval()` and from `eval_batch()`;
CPU routines (`append()`, the CPU eval and its caches) carry no GPU work or
GPU-only state. Shared inputs live in `MarsData` (e.g. `lin_x` / `lin_b`), and
each path consumes them on its own.

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
  Replacing the gathered `Bo` row by an L1-resident one cut the sweep from
  2.8 to 0.93 s, yet no prefetch variant moved it until the binary was
  checked: it contained no prefetch instructions. Tuning then confirmed
  prefetch distance 4, JB=4 columns per row and the forced inline; the loop's
  compute floor is ~0.9 s at n=5e5, m=100 (1.3 s measured). **Check `objdump -d marslib*.so | grep -c prefetcht0` after touching the
  sweep.**
- Smaller: `argsort` is a stable LSD radix sort (51→~5 ms at n=5e5; the
  planned LRU cache is moot); per-thread `Bx` is sized by the active columns
  `p` rather than `m` (Fast-MARS memory); `fit()` caches `max(sse1, sse2)`
  (with `linear_only` or no admissible cut, `sse2` is 0 and the Fast-MARS
  ranking was running on all-zero scores); the appended linear column is
  formed as `b·(x·s)` like the candidate, so it is bit-identical to what was
  scored; `threads` is clamped to ≥1; the MXCSR FTZ/DAZ state is restored by
  RAII; CUDA f64 GEMMs call `cublasDgemm` directly (the emulation compute-type
  plumbing is gone). Dropped with it: the row-order sensitivity of the
  constructor reductions is not a fit-variation source (f64 sums, ~√n·eps_f64;
  `tests/repro_shuffle.py`'s row-order dependence is the sort's tie-breaking
  by row index -- see the comment in the `MarsAlgo` constructor).
- **`fit()` candidate filter vectorized (2026-10-23):** `fit()` keeps
  `degree[k]` and a `(p, M)` boolean `contains[i, k]` alongside the `basis`
  list and applies the degree / self-interaction filter as array operations
  over the whole candidate block; a user `xfilter` is called only on the
  survivors (never when not supplied). Bit-identical models; the Python side
  of an epoch went 3.3 → 1.0 ms at p=200, M=150. The ~130 ms/epoch once
  blamed on this loop was mostly the strided modified-GS `append()`.
- `tests/bench_sweep.cc` (`bench_sweep` target under `BUILD_TESTING`) times
  `append()`, `eval(linear)` and `eval(hinge)` at a fit-like basis and dumps the
  outputs for bit-exact A/B. On the EPYC box, pin with `taskset` and run
  variants sequentially on the same core: concurrent runs on different cores
  disagree by ±20%.

**CUDA note (2026-11-03) — the GPU hinge sweep (`cuda/hinge_sweep.cu`),
first pass, validated on the sm_120 box:** with `cuda=True` and
`linear_only=False`, `eval()` keeps the normalized `Bx` on the device
(`orthonormalize(..., normalize_resident=true)`) and calls
`mars::cuda::hinge_sweep()`, which sorts the raw X column (CUB radix sort,
`-0.0` folded, stable descending like `argsort`) and runs the CPU sweep's f/g
recurrence as a **two-pass chunked scan** over the sorted rows: pass 1
computes each chunk's local end state from zero, a propagate kernel turns
those into carry-ins (`g_b = g_a + g_loc`, `f_b = f_a + D(a,b)·g_a + f_loc`,
`h2_b = h2_a + D(a,b)(2bd_a + D(a,b)b2_a) + h2_loc`, with `D(a,b)` = the f32
deltas summed over `[a, b)` -- `d_a` in, `d_b` out, exactly the steps the
chunk applies, so every row's weight matches the CPU's), and pass 2 reruns
each chunk from its carry-in scoring the min_span grid with the same
`HINGE_DEN_REL` gate (now in `kernels.h`) and support clipping (`i_last[j]`,
the last sorted position with a nonzero parent, clamped to `tail-1`). One
block per (chunk, JB=4 columns); thread lanes are the m Bo columns, the Bx
column, one parent-b lane per column (b2/bd/‖h+‖², in consecutive threads so
their k0/k1 FMAs share two warp instructions) and one y lane per column
(vb/w); Bo is kept row-major on the device (`dBo_rm`) so the gathered row is
contiguous, staged 16 rows at a time in shared memory with the block-uniform
per-row values (d, y, b, Bx) staged as f64. Grid scoring writes per-thread
`ff`/`fy` partials to shared memory and reduces them cooperatively per grid
slot (each thread sums a strided run, then a short shuffle). Everything
accumulates in f64 (the rejected f32 f/g mode applies here too).
- **Measured** (`bench_sweep 500000 {100,250} 20 7 2 - 1`): eval(hinge) 76 ms
  at m=100 (CPU one core: 2.1 s, 28×) and 386 ms at m=250 (12.4 s, 32×);
  identical cuts at m=100, 1/250 near-tie at m=250; vs the CPU sweep on the
  *same* device `Bx` the scores agree to ≤ 7e-9 relative with identical cuts
  (`CudaSweepTest.*`, up to m=260). Per eval: sort+prep 1.5 ms, pass 1 27 / 165
  ms, pass 2 41 / 196 ms (m=100 / 250). Pass 1 sits at ~1.9× the f64 FMA floor
  (14 / 88 ms); the grid scoring is the pass 2 − pass 1 difference.
- **What the tuning found (2026-11-03, all measured on the box):** (1) f32→f64
  conversion runs at the f64 rate on this card -- staging the uniform values as
  f64 took pass 1 from 67 to 38 ms. (2) The per-grid-point warp-shuffle
  reduction (5 f64-add rounds per warp per value) was ~70% extra f64 work at
  min_span=7; the shared-memory batched reduction cut the grid cost from 27 to
  ~12 ms at m=100. (3) The single parent-b lane issuing 8 k0/k1 FMAs alone in
  its warp cost ~25% of pass 1; spreading the b/y lanes one column per thread
  took pass 1 from 38 to 27 ms. (4) Residency matters for pass 2 once the f64
  reduction work is gone: a block past ~34 KB of shared memory (one resident
  block) costs ~2×, so the partials buffer is one grid slot wide and only as
  wide as the lanes that contribute, and the pass 2 tile shrinks to fit;
  larger batches (`MARS_CUDA_SWEEP_BATCH`) measured no better. (5) Idle padding
  lanes cost real FMAs: with 259 lanes a 256-thread block at two lanes per
  thread wasted half the work (pass 1 +34% at m=250); the block is sized to
  the lane count (160 threads for 259 lanes). (6) 8 columns per block
  (`MARS_CUDA_SWEEP_JB=8`) ties at m=100 and loses at m=250 (registers/tile),
  so 4 is the default. No register spills in the default variants (pass 2:
  117 regs, pass 1: 62 at one lane per thread; the JB=8 four-lanes-per-thread
  variant spills).
- **Validation lesson:** a CPU emulation of the chunked scan matched the
  sequential sweep to 1e-13 on a heavily quantized column while the GPU tests
  failed by up to 3% on continuous ones -- the carry had been defined as the
  state *after* step `a` (an extra f-step with `d_b`, deltas over `(a, b]`),
  so pass 2 applied `d_a·g_a` twice, invisible wherever `d_a` is a tie (0).
  Validate sweep numerics on continuous data, and A/B the GPU sweep against
  the CPU sweep on the same device `Bx` (`MARS_CUDA_SWEEP=0`): the f32
  projection in the GPU orthonormalize alone moves near-gate scores by up to
  ~5e-4 relative, which would mask a sweep bug. `MARS_CUDA_SWEEP_CHUNK=65536`
  (one chunk) isolates the carry logic. The orthonormalize's f64 atomics make
  the linear term vary at the ULP between runs (TODO.md), so no bit-equality
  on anything downstream of it.
- Knobs: `MARS_CUDA_SWEEP=0` (GPU orthonormalize + CPU sweep),
  `MARS_CUDA_SWEEP_CHUNK` (rows per chunk; default sized for ~6k blocks),
  `MARS_CUDA_SWEEP_TILE` (rows per shared tile, both passes),
  `MARS_CUDA_SWEEP_BLOCK` (block size; the tests use 32 to force 2 and 4 lanes
  per thread), `MARS_CUDA_SWEEP_JB` (4 or 8 columns per block),
  `MARS_CUDA_SWEEP_BATCH` (grid slots per reduction batch),
  `MARS_CUDA_PROFILE=1` (per-phase times at context destroy),
  `MARS_CUDA_VERBOSE=1` (the chosen shape). `m + 1 + 2·JB > 1024` lanes throws.
  Tests: `CudaSweepTest.*` in `tests/test_cuda.cc` (three-way: CPU, hybrid,
  GPU), `tests/test_cuda.py` (whole fits: identical basis for the hinge fit),
  `bench_sweep ... 1` (timing + deviation vs the CPU eval).

**Numerics note (2026-11-04) — linear ΔSSE scored against the residual
target:** `eval()` now computes `ybx = <q, y_res>` with `y_res = y − Bo·ybo`
(`MarsData::yres`; on the GPU via `context_set_residual()`) instead of
`<q, y>`. The two agree in exact arithmetic, but `Bo` is f32, so
`BoᵀBo = I + E`, and one projection pass leaves the unit candidate `q` with an
in-span error of O(E·T). Against `y`, whose in-span part `Bo·ybo` is O(1), that
enters `ybx` at first order; `y_res`'s in-span part is O(E·ybo), so it becomes
second order. `E` is dominated by the intercept: a constant column rounds
coherently, so `‖bo₀‖² − 1` is ~eps_f32 (6.6e-8 at n=2e4), not damped by √n;
the error therefore scales with the intercept's share of `y`, i.e. with the
target's mean. `MarsTest.LinearDsseMatchesOracleOnUncenteredInputs` (n=2e4,
m=120, inputs with mean/std 3 and 0) scores every candidate against an exact
f64 QR projection onto span(`Bo`): ΔSSE relative error went from median
1.7–3.9e-7, max 5.2–5.9e-5 to median 6.6–7.5e-8, max 1.1e-5; a numpy model put
the old tail at 1.5e-3 on columns the DGKS retry skips. The oracle must use
span(`Bo`) (read through `MarsAlgo::ortho_basis()`), not span(`B`): the two
differ by `Bo`'s f32 rounding, ~eps_f32·κ on the residual, which swamps the
effect. The hinge sweep uses `ybx` as `q`'s coefficient, so its scores move at
the same level (the CPU sweep tests pass unchanged; `CudaSweepTest.*` uses
the same target on both sides). Same review: the CPU DGKS retry
now re-runs the flagged columns through the batched Phase 1/2a (`bo_t_bx` /
`project_out` over an index list) instead of two `Bo` sweeps per column --
**no measurable speedup** (7.6 → 7.3 s, n=2e5, m=80, p=16, uncentered, one
core): the per-column passes already ran at the batched rate, and on
uncentered inputs the retry fires on nearly every candidate, so its cost is a
full second pass of arithmetic. Only avoiding the retry helps (TODO.md 1.B).
**GPU half (same date):** the batched/per-eval DGKS retry used to restart
from the residual left by the f32 projection sgemm, which carries out-of-span
noise ~eps_f32·√m·‖T‖ -- eps_f32·√m·κ relative on a cancelling column -- that
no further projection removes. It now recomputes each flagged column's f64
`B·x` (`refill_widen_kernel`, the same f32 product as the fill) and runs both
passes in f64 against the gathered first-pass `T` (three Dgemms at width
`fired`, was two). `CudaKernelsTest.LinearDsseMatchesOracleCollinearLargeBasis`
(n=6000, m=400, uncentered, default blocked-f32 `T`) on the sm_120 box: GPU
eval / eval_batch median 2.6e-7, max 7.5e-6 relative vs the span(`Bo`) oracle
(CPU: 2.8e-7 / 1.1e-5); the old retry gave median 6.7e-4 / 1.1e-3, max
0.12 / 0.24. The hinge sweep's loose GPU-vs-CPU A/B is unchanged by both
halves (`CudaSweepTest.MatchesCpuEvalLargeBasis`, m=260: max rel 6.1e-2 before
and after; pairs > 1e-3 34 → 32, differing cuts 9 → 6) -- that spread is the
f32 projection on non-retried columns near `HINGE_DEN_REL`, not this change.

**Performance note (2026-11-04) — closed-form linear ΔSSE (CPU
`linear_only`):** `eval(linear_only)` on the CPU calls `mars::linear_dsse`
instead of `orthonormalize()` + a dot. With `T = BoᵀBx` (Phase 1),
`‖Bx‖²` and `<Bx,y>` (fused into the fill), it scores
`den = ‖Bx‖² − TᵀT + TᵀET`, `u = <Bx,y> − Tᵀybo + TᵀE·ybo`,
`ΔSSE = u²/(den+tol)`: the projection onto span(`Bo`) with the exact Gram
`I + E` to first order (the dropped term is ~‖E‖²κ², ~5e-9 at the gate). No
projection, DGKS retry, normalize or f32 rounding of the residual. Columns
with κ² = ‖Bx‖²/den > `LINEAR_KAPPA2_MAX` (1e6) fall back to the explicit
path; the f64 cancellation in `den` costs ~eps_f64·√n·κ² relative, so the
gate holds that under ~1e-7 at n=5e5 (observed 3e-10 at κ² ~ 7e4, n=4096). The
full `E` is required: its diagonal alone (the coherent intercept defect) left
errors up to 1.7e-3 in a numpy model. Accuracy vs the span(`Bo`) oracle
(n=2e4, m=120): median 4.5e-9 / 1.1e-12, max 8.5e-7 / 1.5e-10 for inputs with
mean/std 3 / 0 (explicit path: median ~7e-8, max ~1.1e-5). Speed, one core,
n=2e5, m=80, p=16, two noisy back-to-back runs each: uncentered 8.1 / 10.8 →
3.7 / 4.1 s (2.2–2.6×, the DGKS retry is gone), centered 3.9 / 5.7 → 3.6 /
4.0 s (~1.1–1.4×): Phase 2a was the cheap half, so Phase 1 (then store-bound;
register-tiled since, next note) was nearly the whole cost. `T`'s accuracy is
load-bearing: `den` takes a `T` error at first order, so the GPU's blocked-f32
`T` (~2e-7·κ² relative at n=5e5) cannot feed the closed form; the GPU scores
it only from its own f64 T cache (the GPU T cache note below).
Tests: `KernelsTest.LinearDsseMatchesExactProjection` (closed form, high κ,
fallback with a residual, in-span, zero column), and the tightened
`MarsTest.LinearDsseMatchesOracleOnUncenteredInputs`.

**Performance note (2026-11-04) — register-tiled Phase 1:** `bo_t_bx`
(`T = BoᵀBx`, f32 in / f64 accumulate) walks the rows in 128-row panels,
packs the panel's `Bo` rows as f64 in 12-row k-tiles and widens the `Bx`
columns, and accumulates a 12×4 block of `T` in YMM registers across the
panel (12 accumulators + 3 `Bo` vectors + 1 broadcast = all 16 registers),
adding it to `T` once per panel. The old loop stored a `T` element on every
FMA. Fewer than 4 columns (small DGKS retries) keep the old loop. Sums become
two-level, so `T` is not bit-identical to before (slightly more accurate);
all tests unchanged. `bench_sweep 500000 {100,250}`, one core, interleaved
A/B under heavy shared load: `eval(linear)` 0.63 → 0.30 s at m=100 and 4.8 →
1.7–2.2 s at m=250 (2.0–2.2×, on top of the closed form); `eval(hinge)` gains
~10% (it shares Phase 1). ~15 GFMA/s at m=250, ~60–80% of one core's AVX2
peak. Tried and rejected: an AVX-512 16×4 tile (15–30% slower; same story as
the 2026-05-16 note); panel rows 64/256/512 (no difference within the noise);
the 8×4 tile is ~10–15% slower than 12×4 -- re-check on a quiet box.

**Performance note (2026-11-04) — `linear_only` T cache (CPU):** `Bo`, `B`
and the scaled X never change, so `T[r, (k,c)] = <bo_r, fl(b_k·x_c)>` never
changes once computed. `MarsAlgo::eval_linear_cached` keeps, per X column c
(`LinCache`), the `T` column of every parent the mask has requested plus
`‖Bx‖²`, `<Bx,y>`, `TᵀT`, `TᵀET`, `Tᵀybo`; each eval extends the live
parents by the rows appended since c's last eval (`mars::cross_rows`, one
pass over their `B` columns, the f32 product formed as the fill forms it --
an unrounded `B·(bo⊙x)` GEMM would read `B` once for all c but perturb `den`
at first order), adds a full Phase 1 column for newly requested parents, and
scores in closed form at O(m) per candidate (only `TᵀE·ybo` is recomputed).
Per X column per epoch that is O(n·m) instead of O(n·m·p). A candidate the
explicit fallback finds in span(`Bo`) (residual ≤ tol) is marked dead: the
span only grows, so it scores 0 for good and is no longer extended. Without
that, every appended (c, k) pair was re-proven degenerate (projection + DGKS
retry) every later epoch, which cut the gain from ~5× to 2.5×. Memory: up to
`max_terms²` f64 per X column, enabled when that worst case fits
`MARS_LINEAR_CACHE_GB` (default 8; 0 disables). Whole `fit(linear_only=True)`,
n=2e5, p=16 uncentered inputs, one core, under heavy shared load: 60 terms
13.5 → 2.8 s (eval 13.0 → 2.3 s), 120 max terms (stops at 72) 23.2 → 4.0 s
(eval 22.6 → 3.4 s); identical models. What remains per column per epoch is
the new parent's Phase 1 column (a single column takes the store-bound axpy
loop) and the row extension. `MarsTest.LinearCacheMatchesUncached` drives
both paths through 79 lagged, randomly masked epochs: max 3e-8 relative
apart, final scores within median 4.5e-10 / max 2e-8 of the oracle. The GPU
half is the next note.

**Performance note (2026-11-04) — `linear_only` T cache (GPU,
`cuda/linear_cache.cu`):** `eval_batch()` keeps the same cache on the device
when `MARS_LINEAR_CACHE_GB` allows (one dense slot per (X column, parent):
f64 `T` column of `max_terms` rows plus the five scalars; it falls back to
`orthonormalize_batch()` if the allocation fails). Per call: one f64 kernel
extends every live slot by the Bo rows appended since the last call (one
thread per slot walking a row chunk, the f32 product formed as the fill forms
it, f64 atomics across chunks; up to 4 rows per launch) and a warp per slot
updates TᵀT / TᵀET / Tᵀybo row by row (only `E`'s upper triangle is read, so
only its new columns are uploaded); newly requested slots get a fill + f64
`cublasDgemm`; a warp per candidate scores the closed form; κ² >
`LINEAR_KAPPA2_MAX` candidates go through `f64_reproject` (the DGKS retry's
two f64 passes, factored out of `project_reduce` and shared) and score
`<r,y_res>²/(s+tol)` in f64, dying when `s <= tol`. The blocked-f32 `T` is
never used here: `den` takes `T` errors at first order. Whole
`fit(linear_only=True, cuda=True)` on the sm_120 box, uncentered inputs:
n=2e5, p=16, 72 terms: 1.91 → 0.90 s (eval 1.07 → 0.15 s); n=5e5, p=50, 53
terms: 5.12 → 1.78 s (eval 3.66 → 0.33 s; extension 52 ms, fresh slots 74
ms, scoring 2 ms, fallback 106 ms for 79 candidates, all dead). Identical
models with and without the cache. The uncached path's cost was mostly the
DGKS retry (2.8 of 3.6 s: it fires on ~2/3 of uncentered candidates). The
fit was then bound by the CPU `append()` (~0.84 s of the 1.78 s), threaded
next (note below).
`CudaKernelsTest.LinearCacheMatchesCpuOverEpochs` drives the device cache
against the CPU cache on one MarsAlgo over a 90-term fit with random masks,
inactive X columns, lagged calls (multi-row extensions) and double appends:
max 4.2e-8 relative apart over 6305 comparisons (931 candidates 0 on both),
final scores median 7.1e-10 / max 1.9e-7 from the span(`Bo`) oracle. Same change: the batch capacity `p_cap` is clamped to 65535 (`grid.y`).

**Performance note (2026-11-04) — threaded `append()`, dead-on-append:**
with the T caches, `append()` became the largest per-epoch cost of a
`linear_only` fit (12–20 ms per call at n=5e5, single-threaded: ~half of a
GPU fit). Its row passes now run over fixed 4096-row blocks
(`mars::row_blocks`, `ROW_BLOCK`) on the `threads` OpenMP threads `fit()`
passes (`append(..., threads)`; the binding resolves ≤ 0 like `eval()`).
Every reduction is formed per block and combined in block order, so B, Bo,
the MSE and all downstream scores are bit-identical for any thread count
(`MarsTest.AppendIsThreadCountIndependent`). The new Bo column, its Gram
column for `E` and the new `ybo` entry come out of one sweep
(`mars::store_col`, which replaced `gram_col`), and the `y_res` update counts
the support in the same pass. n=5e5, 53 appends: 634 ms total at 1 thread
(839 before the fusion), 260 / 131 / 79 / 65 ms at 4 / 8 / 16 / 32 (the 16-
and 32-thread runs unpinned; pinned to 16 busy cores of the shared box it
collapsed to 50 ms per call, so oversubscription hurts here as in `eval()`).
Whole CPU fit, n=2e5, p=16, 72 terms, 16 threads: 1.37 → 0.6 s, identical
model. **Dead-on-append:** the linear candidate (c, k) just appended is the
new B column bit for bit, so it is in span by construction. `append()` only
records the term (`MarsData::lin_x` / `lin_b`, -1 for the intercept and
hinges); each cache marks the slot dead itself when it catches up with that
row (the CPU one per X column in `eval_linear_cached`, the device one in
`linear_cache_eval`), instead of proving it through the explicit fallback;
dead slots are never filled. `append()` holds no cache or GPU logic: the CPU
and GPU paths share only `MarsData`, and `MarsData::lin_cache` (the
`MARS_LINEAR_CACHE_GB` budget check) switches each cache on independently. Candidates in span by other routes (b_j·x_c·x_c' reached in
both orders) still take the fallback. GPU `fit(linear_only=True,
cuda=True)`, n=5e5, p=50, 53 terms: 1.78 → 0.92 s with the cache
(orchestration 1.37 → 0.49 s; fallbacks 79 → 28, 106 → 44 ms; 52 slots dead
on append), identical model. Of the 0.32 s eval, `context_sync_basis` is now
the largest piece (~130 ms: a single-threaded strided gather of each new
`Bo` column from the row-major host copy; TODO.md 1). Keep that fix inside
the CUDA path: the CPU path must not carry work for the GPU.

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

**Benchmarking note:** on the shared EPYC server, timings differ by ±20%
between cores and between concurrent runs -- enough to invert small A/B
results. Build each variant as its own binary, `taskset -c <one core>`, run
variants back to back on that core, and diff `bench_sweep`'s dump files for
bit-exactness. The bench's basis must stay well conditioned (random hinges on
random parents make the summed delta-SSEs noise-dominated and useless for
A/B). A perf benchmark must also exercise the DGKS retry or it measures a path
production never takes: the random-`Bo` benches never fire it, while a
fit-like basis on uncentered features (mean/std = 3) fired it on 66–89% of
candidates in a 2026-11-04 numpy emulation, and the same offset doubled CPU
`eval(linear)` (3.8 → 7.6 s, n=2e5, m=80, p=16, one core). The 2026-06-17 GPU
profile had DGKS at 19%; `CudaKernelsTest.BatchDgksRetryMatchesPerColumn` covers the
batched retry). For whole `linear_only` fits, `tests/bench_linear.py` (`--cuda`
for the GPU) A/Bs the T cache against `MARS_LINEAR_CACHE_GB=0` in one process
and checks the models match; `tests/bench_eval.py` is the whole-fit bench for
hinge fits (`--cuda`, `--linear-only`), with `MARS_CUDA_PROFILE=1` for the
per-phase GPU breakdown.

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
