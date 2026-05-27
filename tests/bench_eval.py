#!/usr/bin/env python3
"""
Short MARS fit benchmark for allocation/perf profiling.

Default parameters target ~10s on this box and exercise eval() over
many columns × many threads, which is the regime where glibc arena
churn shows up as state-D threads in mmap/munmap.

Typical wrappers
----------------
    # 1. baseline wall time + peak RSS
    python tests/bench_eval.py

    # 2. confirm arena theory: collapse to one arena
    MALLOC_ARENA_MAX=1 python tests/bench_eval.py

    # 3. jemalloc comparison
    LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so.2 \
        python tests/bench_eval.py

    # 4. attribute mmap calls to C++ stacks
    sudo perf record -e syscalls:sys_enter_mmap -g \
        -p $(pgrep -f bench_eval.py)  # in another shell while it runs

    # 5. per-callsite allocation tally
    heaptrack python tests/bench_eval.py
"""
import argparse
import os
import resource
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mars  # noqa: E402


def make_data(n, p, seed):
    rng = np.random.default_rng(seed)
    X = np.asfortranarray(rng.standard_normal((n, p)).astype("f"))
    y = (
        np.maximum(X[:, 0] - 0.3, 0)
        - 0.8 * np.maximum(0.5 - X[:, 1], 0)
        + 0.6 * X[:, 2] * np.maximum(X[:, 3], 0)
        + 0.4 * X[:, 4]
        + 0.3 * rng.standard_normal(n).astype("f")
    ).astype("f")
    return X, y


def rss_kb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100_000)
    ap.add_argument("--p", type=int, default=1200)
    ap.add_argument("--max-terms", type=int, default=13)
    ap.add_argument("--threads", type=int, default=-1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--warmup", action="store_true",
                    help="run a tiny fit first to JIT numba / warm allocators")
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--linear-only", action="store_true",
                    help="pass linear_only=True to mars.fit (production regime)")
    args = ap.parse_args()

    X, y = make_data(args.n, args.p, args.seed)
    print(f"# n={args.n} p={args.p} max_terms={args.max_terms} threads={args.threads}"
          f"  linear_only={args.linear_only}"
          f"  MALLOC_ARENA_MAX={os.environ.get('MALLOC_ARENA_MAX', 'default')}"
          f"  LD_PRELOAD={os.environ.get('LD_PRELOAD', '')}")

    if args.warmup:
        mars.fit(X[:1000], y[:1000], max_terms=5, threads=args.threads,
                 linear_only=args.linear_only)

    rss_before = rss_kb()
    for i in range(args.repeat):
        t0 = time.perf_counter()
        model = mars.fit(X, y, max_terms=args.max_terms, threads=args.threads,
                         linear_only=args.linear_only)
        dt = time.perf_counter() - t0
        print(f"run {i+1}/{args.repeat}: {dt:.2f}s  M={len(model)}"
              f"  ms/term={1000*dt/max(len(model),1):.0f}"
              f"  peak_rss={rss_kb()/1024:.0f} MB"
              f"  Δrss={(rss_kb()-rss_before)/1024:+.0f} MB")


if __name__ == "__main__":
    main()
