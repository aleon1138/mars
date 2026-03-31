# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Test Commands

```bash
make              # build the pybind11 extension (marslib*.so)
make test         # build and run C++ unit tests
make clean        # remove build artifacts
make format       # format all .h/.cc files with astyle

pip install .     # alternative: install via pip (uses pyproject.toml)
```

For memory/sanitizer debugging, replace `-O3` with `-O0 -g -fsanitize=address` in `CXXFLAGS`.

To run a single GoogleTest: `./unittest --gtest_filter=MarsTest.DeltaSSE`

## Architecture

The library has two layers:

**C++ core (`marsalgo.h` / `marsalgo.cc`)** — implements `MarsAlgo`, a stateful object that maintains the growing basis matrix during the forward pass. Key internal state (in `MarsData`):
- `X` — read-only Fortran-order (column-major) view of training data
- `B` / `Bo` — the current basis and its ortho-normalized counterpart (kept in sync via Gram-Schmidt)
- `y`, `ybo` — normalized target and its projection onto the ortho-basis

Three methods drive the algorithm:
- `eval()` — for each candidate `X` column, computes the delta-SSE improvement from adding a linear term or a pair of mirror hinges (`+`/`-`). Uses AVX2/FMA intrinsics in the inner loop (`covariates()`).
- `append()` — commits a chosen basis (type `'l'`, `'+'`, or `'-'`), updates `Bo` and `ybo`.
- `nbasis()` / `dsse()` / `yvar()` — read-only accessors.

**pybind11 bindings (`marslib.cc`)** — exposes `MarsAlgo` as `marslib.MarsAlgo`. The `eval()` binding parallelizes over `X` columns using OpenMP, releasing the GIL. It accepts a boolean mask matrix `(p, m)` controlling which `(input, basis)` pairs to evaluate (Fast-MARS cache).

**Python orchestration (`mars.py`)** — implements the full MARS forward pass loop on top of `marslib`. Uses `numba` for `expand()`. Key public API:
- `fit(X, y, ...)` → structured numpy array `model` of basis nodes
- `expand(X, model)` → basis matrix `B` for a new dataset
- `prune(XX, XY, YY, ...)` → backward pass / GCV pruning
- `pprint(model, beta)` → human-readable model formula

**Data requirements:** `X` must be `float32`, **column-major** (Fortran order). `y` and `w` must be `float32` column-major 1D arrays. The bindings assert these layouts explicitly.

**Model node dtype** (returned by `fit`):
```
type   S1   — b'i' intercept, b'l' linear, b'+'/b'-' hinges
basis  i4   — index of parent basis node
input  i4   — column index into X
hinge  f8   — hinge cut point (NaN for linear/intercept)
r2     f4
r2_cv  f4   — GCV-adjusted R²
order  i4   — polynomial degree
time   f4
```
