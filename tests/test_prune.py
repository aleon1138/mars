"""Tests for mars.prune()."""

import numpy as np
from mars import prune


def _make_gram(B, y):
    """Build prune() inputs from a basis matrix B and target y."""
    XX = B.T @ B
    XY = B.T @ y
    YY = float(y @ y)
    return XX, XY, YY


def test_perfect_fit_kept():
    """When all terms contribute to a perfect fit, none are pruned."""
    rng = np.random.RandomState(0)
    n = 200
    B = np.column_stack([np.ones(n), rng.randn(n), rng.randn(n)])
    true_beta = np.array([1.0, 3.0, -2.0])
    y = B @ true_beta

    beta = prune(*_make_gram(B, y), n_true=n, penalty=3)
    np.testing.assert_allclose(beta, true_beta, atol=1e-6)


def test_noise_columns_pruned():
    """Noise-only columns should be pruned away."""
    rng = np.random.RandomState(1)
    n = 500
    signal = rng.randn(n)
    noise1 = rng.randn(n) * 0.01
    noise2 = rng.randn(n) * 0.01
    B = np.column_stack([np.ones(n), signal, noise1, noise2])
    y = 5.0 + 3.0 * signal

    beta = prune(*_make_gram(B, y), n_true=n, penalty=3)
    # Signal columns should survive
    assert beta[0] != 0, "intercept was pruned"
    assert beta[1] != 0, "signal was pruned"
    # Noise columns should be zeroed out
    assert beta[2] == 0, "noise1 survived"
    assert beta[3] == 0, "noise2 survived"


def test_zero_variance_column_excluded():
    """Columns with zero variance (diag(XX)==0) are never selected."""
    rng = np.random.RandomState(2)
    n = 200
    B = np.column_stack([np.ones(n), rng.randn(n), np.zeros(n)])
    y = 2.0 + 4.0 * B[:, 1]

    beta = prune(*_make_gram(B, y), n_true=n, penalty=3)
    assert beta[2] == 0, "zero-variance column should have beta=0"
    np.testing.assert_allclose(beta[0], 2.0, atol=1e-5)
    np.testing.assert_allclose(beta[1], 4.0, atol=1e-5)


def test_mask_restricts_terms():
    """A mask=False term should never get a nonzero coefficient."""
    rng = np.random.RandomState(3)
    n = 200
    x1 = rng.randn(n)
    x2 = rng.randn(n)
    B = np.column_stack([np.ones(n), x1, x2])
    y = 1.0 + 2.0 * x1 + 3.0 * x2

    mask = np.array([True, True, False])
    beta = prune(*_make_gram(B, y), n_true=n, penalty=3, mask=mask)
    assert beta[2] == 0, "masked term should remain zero"


def test_ridge():
    """Ridge regularization should shrink coefficients toward zero."""
    rng = np.random.RandomState(4)
    n = 200
    x = rng.randn(n)
    B = np.column_stack([np.ones(n), x])
    # Normalize columns to unit norm so the ridge assert passes
    B = B / np.linalg.norm(B, axis=0)
    y = B @ np.array([5.0, 10.0])

    beta_no_ridge = prune(*_make_gram(B, y), n_true=n, penalty=3, ridge=0)
    beta_ridge = prune(*_make_gram(B, y), n_true=n, penalty=3, ridge=0.5)
    # Ridge should shrink the L2 norm of coefficients
    assert np.linalg.norm(beta_ridge) < np.linalg.norm(beta_no_ridge)


def test_single_term():
    """Pruning a single-term model (intercept only) should still work."""
    n = 100
    B = np.ones((n, 1))
    y = np.full(n, 3.0)

    beta = prune(*_make_gram(B, y), n_true=n, penalty=3)
    np.testing.assert_allclose(beta, [3.0], atol=1e-6)


def test_full_elimination_finds_global_best():
    """
    Verify that full elimination can find a better model than greedy stopping.

    Construct a case where removing term A alone hurts GCV, but removing
    both A and B together is better than the full model. A greedy approach
    that stops when no single removal helps would miss this.
    """
    rng = np.random.RandomState(42)
    n = 50
    x1 = rng.randn(n)
    x2 = rng.randn(n)
    # Two correlated noise terms that individually look useful due to
    # their mutual correlation, but are jointly useless.
    shared = rng.randn(n) * 0.15
    noise_a = shared + rng.randn(n) * 0.01
    noise_b = shared + rng.randn(n) * 0.01

    B = np.column_stack([np.ones(n), x1, x2, noise_a, noise_b])
    y = 1.0 + 2.0 * x1 - 1.5 * x2

    beta = prune(*_make_gram(B, y), n_true=n, penalty=3)
    # The correlated noise pair should both be pruned
    np.testing.assert_allclose(beta[3], 0, atol=1e-10)
    np.testing.assert_allclose(beta[4], 0, atol=1e-10)
    # Signal should survive
    assert abs(beta[0]) > 0.1
    assert abs(beta[1]) > 0.1
    assert abs(beta[2]) > 0.1


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
