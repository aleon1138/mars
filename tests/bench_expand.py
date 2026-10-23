#!/usr/bin/env python3
"""Time mars.expand() against the pre-kernel numpy loop.

    python3 tests/bench_expand.py [n] [m] [p]

Builds a random model of `m` nodes over `p` inputs, expands an F-ordered f32 X
of `n` rows, and reports the numpy loop once and the kernel at several thread
counts in both output orders. Each call allocates a fresh B, so the timings
include first-touch page faults, as a real call does. The kernel is memory
bound; run on an idle box and expect the usual cross-core noise (see
CLAUDE.md's bench notes).
"""

import os
import sys
import time

import numpy as np

# Import the checkout's mars.py + marslib, not a pip-installed copy: run as
# `python3 tests/bench_expand.py`, sys.path[0] is tests/, not the repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mars  # noqa: E402


def expand_numpy(X, model):
    """mars.expand() as it was before the kernel (single core, numpy)."""
    B = np.empty((len(X), len(model)), dtype=X.dtype, order="F")
    for i, node in enumerate(model):
        t, b, x, h = node["type"], node["basis"], node["input"], node["hinge"]
        if t == b"i":
            B[:, i] = 1.0
        elif t == b"l":
            B[:, i] = B[:, b] * X[:, x]
        elif t == b"+":
            B[:, i] = B[:, b] * np.maximum(X[:, x] - h, 0)
        else:
            B[:, i] = B[:, b] * np.maximum(h - X[:, x], 0)
    return B


def random_model(rng, X, m, max_degree=3):
    dtype = [("type", "S1"), ("basis", "i4"), ("input", "i4"), ("hinge", "f8")]
    model = np.zeros(m, dtype=dtype)
    model[0] = ("i", 0, 0, np.nan)
    degree = [0]
    for i in range(1, m):
        parents = [k for k in range(i) if degree[k] < max_degree]
        b = int(rng.choice(parents))
        x = int(rng.integers(X.shape[1]))
        t = str(rng.choice(["l", "+", "-"]))
        h = float(X[rng.integers(len(X)), x]) if t != "l" else np.nan
        model[i] = (t, b, x, h)
        degree.append(degree[b] + 1)
    return model


def main():
    n = int(float(sys.argv[1])) if len(sys.argv) > 1 else 4_000_000
    m = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    p = int(sys.argv[3]) if len(sys.argv) > 3 else 16

    rng = np.random.default_rng(0)
    X = np.asfortranarray(rng.standard_normal((n, p)).astype("f"))
    model = random_model(rng, X, m)
    gb = n * m * 4 / 1e9
    print(f"n={n:,}  m={m}  p={p}  B = {gb:.2f} GB  ({mars.__file__})")

    t0 = time.perf_counter()
    B0 = expand_numpy(X, model)
    t_np = time.perf_counter() - t0
    print(f"numpy loop                   {t_np:8.3f} s  {gb / t_np:6.1f} GB/s")

    for order in "FC":
        for threads in (1, 2, 4, 8, 16, 32, -1):
            best = float("inf")
            for _ in range(3):
                t0 = time.perf_counter()
                B = mars.expand(X, model, order=order, threads=threads)
                best = min(best, time.perf_counter() - t0)
            # Not bit-exact: the old loop subtracts the f64 cut from f32 data
            # (NumPy 2 promotes that to f64 and rounds once at the store), the
            # kernel rounds the cut to f32 first as append() does. ~1 ulp.
            np.testing.assert_allclose(B, B0, rtol=1e-6, atol=1e-7)
            del B
            print(
                f"kernel order={order} threads={threads:3d}  {best:8.3f} s"
                f"  {gb / best:6.1f} GB/s  {t_np / best:6.1f}x"
            )


if __name__ == "__main__":
    main()
