#!/usr/bin/env python3
"""
Quantify how much shuffling input rows changes MARS's GCV R² curves,
and decompose the contribution by source of numerical noise.

We generate a MARS-friendly synthetic target, then run fit() once with
the data sorted (worst case for row-order summation) and K times with
random shuffles. Two metrics:

  roughness  - within-curve sum of |Δ² r2_cv|. The user's "smoother"
               observation: smaller = smoother.
  spread     - across-shuffle std of r2_cv at each epoch, summed. A
               proxy for "how much does this fix make results
               row-order-invariant?"
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mars  # noqa: E402


def make_data(n=200000, p=12, noise=0.3, seed=0, ties=True):
    """Stress the numerics: large n, low-cardinality features (ties),
    correlated features, heavy-tailed noise."""
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((n, p)).astype("f")
    # Correlate some features
    Z[:, 1] = 0.7 * Z[:, 0] + 0.3 * Z[:, 1]
    Z[:, 5] = 0.5 * Z[:, 2] + 0.5 * Z[:, 5]
    if ties:
        # Bin a few features to create ties — this exercises the unstable sort
        # and also concentrates floating-point mass at repeated values.
        Z[:, 6] = np.round(Z[:, 6] * 4) / 4
        Z[:, 7] = np.round(Z[:, 7] * 2) / 2
        Z[:, 8] = (Z[:, 8] > 0).astype("f")
    X = np.asfortranarray(Z)
    y = (
        np.maximum(X[:, 0] - 0.3, 0)
        - 0.8 * np.maximum(0.5 - X[:, 1], 0)
        + 0.6 * X[:, 2] * np.maximum(X[:, 3], 0)
        + 0.4 * X[:, 4]
        + 0.3 * X[:, 6] * X[:, 7]
    )
    # Heavy-tailed noise grows the magnitude of intermediate sums.
    y = y + noise * rng.standard_t(df=3, size=n).astype("f")
    return X, y.astype("f")


def fit_curve(X, y, max_epochs=20, seed=None):
    if seed is not None:
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(X))
        X, y = np.asfortranarray(X[perm]), y[perm]
    model = mars.fit(
        X, y,
        max_epochs=max_epochs,
        threads=1,        # avoid OpenMP nondeterminism
        r2_window=1_000,  # disable patience early-stop for clean curves
        r2_thresh=0.0,
    )
    return model["r2_cv"].astype(np.float64)


def roughness(curve):
    """Sum of |2nd differences|. Larger = bumpier."""
    if len(curve) < 3:
        return 0.0
    d2 = np.diff(curve, n=2)
    return float(np.sum(np.abs(d2)))


def measure(label, X, y, k_shuffles=10, max_epochs=20):
    sorted_idx = np.argsort(y)  # induce strong row-order structure
    Xs, ys = np.asfortranarray(X[sorted_idx]), y[sorted_idx]
    sorted_curve = fit_curve(Xs, ys, max_epochs=max_epochs)

    shuffled_curves = []
    for s in range(k_shuffles):
        c = fit_curve(X, y, max_epochs=max_epochs, seed=1000 + s)
        shuffled_curves.append(c)
    L = min(len(sorted_curve), *(len(c) for c in shuffled_curves))
    sorted_curve = sorted_curve[:L]
    shuffled_curves = np.array([c[:L] for c in shuffled_curves])

    rough_sorted = roughness(sorted_curve)
    rough_shuffled = np.mean([roughness(c) for c in shuffled_curves])
    spread = float(np.sum(shuffled_curves.std(axis=0)))

    print(f"\n=== {label} ===")
    print(f"  curve length:        {L}")
    print(f"  sorted   roughness:  {rough_sorted:.5e}")
    print(f"  shuffled roughness:  {rough_shuffled:.5e}  (mean over {k_shuffles})")
    print(f"  ratio sorted/shuf:   {rough_sorted / max(rough_shuffled, 1e-30):.2f}x")
    print(f"  shuffled spread:     {spread:.5e}  (sum of per-epoch std)")
    print(f"  final r2_cv sorted:  {sorted_curve[-1]:.6f}")
    print(f"  final r2_cv shuf:    {shuffled_curves[:, -1].mean():.6f}"
          f" ± {shuffled_curves[:, -1].std():.2e}")
    return {
        "rough_sorted": rough_sorted,
        "rough_shuffled": rough_shuffled,
        "ratio": rough_sorted / max(rough_shuffled, 1e-30),
        "spread": spread,
        "curves": {"sorted": sorted_curve, "shuffled": shuffled_curves},
    }


if __name__ == "__main__":
    X, y = make_data()
    label = sys.argv[1] if len(sys.argv) > 1 else "baseline"
    out = measure(label, X, y, k_shuffles=8, max_epochs=25)
    np.savez(f"/tmp/repro_{label}.npz", **out["curves"])
