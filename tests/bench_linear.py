#!/usr/bin/env python3
"""
A/B a whole fit(linear_only=True) with and without the T cache.

The cache switch (MARS_LINEAR_CACHE_GB, read by the MarsAlgo constructor) is
flipped between fits in this one process, alternating, after a small warm-up
fit (CUDA context / cuBLAS init with --cuda). Prints wall times and whether
the models match term for term. Inputs are offset (mean `--mu`, default 3),
the regime where the DGKS retry fires on most candidates without the cache.

    python tests/bench_linear.py --n 200000 --p 16 --max-terms 120
    MARS_CUDA_PROFILE=1 python tests/bench_linear.py --cuda --n 500000 --p 50 --max-terms 300

Without the cache the GPU path uses the blocked-f32 T, so a GPU A/B may
legitimately differ on near-tied terms (it has matched in practice).
"""

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mars  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200_000)
    ap.add_argument("--p", type=int, default=16)
    ap.add_argument("--max-terms", type=int, default=120)
    ap.add_argument("--mu", type=float, default=3.0)
    ap.add_argument("--threads", type=int, default=-1)
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--cuda", action="store_true")
    a = ap.parse_args()

    rng = np.random.default_rng(0)
    Z = rng.standard_normal((a.n, a.p)).astype(np.float32)
    y = (np.sin(Z[:, 0]) * Z[:, 1] + Z[:, 2] * Z[:, 3]
         + 0.3 * rng.standard_normal(a.n)).astype(np.float32)
    X = np.asfortranarray(Z + a.mu)

    def run(cache, Xf, yf, terms):
        if cache:
            os.environ.pop("MARS_LINEAR_CACHE_GB", None)
        else:
            os.environ["MARS_LINEAR_CACHE_GB"] = "0"
        t = time.perf_counter()
        model = mars.fit(Xf, yf, max_terms=terms, linear_only=True, cuda=a.cuda,
                         threads=a.threads, r2_thresh=0.0)
        return model, time.perf_counter() - t

    m0 = min(a.n, 20_000)
    run(True, np.asfortranarray(X[:m0]), y[:m0], 8)    # warm-up
    models = {}
    for _ in range(a.repeat):
        for cache in (False, True):
            model, dt = run(cache, X, y, a.max_terms)
            models[cache] = model
            print(f"{'cache   ' if cache else 'no cache'} {'gpu' if a.cuda else 'cpu'} "
                  f"n={a.n} p={a.p} terms={len(model)} r2={model['r2'][-1]:.6f}: "
                  f"{dt:.2f} s", flush=True)

    u, c = models[False], models[True]
    k = min(len(u), len(c))
    same = (u["input"][:k] == c["input"][:k]) & (u["basis"][:k] == c["basis"][:k])
    if same.all() and len(u) == len(c):
        verdict = "identical"
    else:
        verdict = f"first difference at term {int(np.argmin(same)) if not same.all() else k}"
    print(f"models: {len(u)} vs {len(c)} terms, {verdict}; "
          f"r2 {u['r2'][-1]:.7f} vs {c['r2'][-1]:.7f}")


if __name__ == "__main__":
    main()
