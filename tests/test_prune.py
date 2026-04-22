"""Tests for mars.prune() and mars.gram()."""

import numpy as np
from mars import prune, gram


def test_gram_square():
    """gram(A) should match A.T @ A."""
    rng = np.random.RandomState(0)
    A = rng.randn(500, 7).astype("f")
    np.testing.assert_allclose(gram(A), A.T.astype("d") @ A.astype("d"), atol=1e-5)


def test_gram_cross():
    """gram(A, B) should match A.T @ B."""
    rng = np.random.RandomState(1)
    A = rng.randn(500, 7).astype("f")
    B = rng.randn(500, 3).astype("f")
    np.testing.assert_allclose(gram(A, B), A.T.astype("d") @ B.astype("d"), atol=1e-5)


def test_gram_1d():
    """gram(A, y) with 1D y should return a 1D vector."""
    rng = np.random.RandomState(2)
    A = rng.randn(300, 4).astype("f")
    y = rng.randn(300).astype("f")
    out = gram(A, y)
    assert out.shape == (4,)
    np.testing.assert_allclose(out, A.T.astype("d") @ y.astype("d"), atol=1e-5)


def test_gram_scalar():
    """gram(y, y) with 1D y should return a scalar."""
    rng = np.random.RandomState(3)
    y = rng.randn(300).astype("f")
    out = gram(y, y)
    assert isinstance(out, float)
    np.testing.assert_allclose(out, float(y @ y), atol=1e-3)


def test_gram_precision():
    """Large n should accumulate in f64, beating naive f32 sum."""
    # Use an exactly-representable f32 value to isolate accumulation error
    n = 100_000
    A = np.full((n, 1), 0.5, dtype="f")  # 0.5 * 0.5 = 0.25, both exact in f32
    expected = n * 0.25
    out = gram(A)
    np.testing.assert_allclose(out[0, 0], expected, rtol=1e-14)
    # Naive f32 reduction loses precision at this n; f64 accumulation preserves it
    naive = float((A.T @ A)[0, 0])
    assert abs(out[0, 0] - expected) <= abs(naive - expected)


def test_perfect_fit_kept():
    """When all terms contribute to a perfect fit, none are pruned."""
    rng = np.random.RandomState(0)
    n = 200
    B = np.column_stack([np.ones(n), rng.randn(n), rng.randn(n)])
    true_beta = np.array([1.0, 3.0, -2.0])
    y = B @ true_beta

    beta = prune(B, y, n_true=n, penalty=3)
    np.testing.assert_allclose(beta, true_beta, atol=1e-5)


def test_noise_columns_pruned():
    """Noise-only columns should be pruned away."""
    rng = np.random.RandomState(1)
    n = 500
    signal = rng.randn(n)
    noise1 = rng.randn(n) * 0.01
    noise2 = rng.randn(n) * 0.01
    B = np.column_stack([np.ones(n), signal, noise1, noise2])
    y = 5.0 + 3.0 * signal

    beta = prune(B, y, n_true=n, penalty=3)
    assert abs(beta[0]) > 0.1, "intercept was pruned"
    assert abs(beta[1]) > 0.1, "signal was pruned"
    np.testing.assert_allclose(beta[2], 0, atol=1e-10)
    np.testing.assert_allclose(beta[3], 0, atol=1e-10)


def test_zero_variance_column_excluded():
    """Columns with zero variance are never selected."""
    rng = np.random.RandomState(2)
    n = 200
    B = np.column_stack([np.ones(n), rng.randn(n), np.zeros(n)])
    y = 2.0 + 4.0 * B[:, 1]

    beta = prune(B, y, n_true=n, penalty=3)
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
    beta = prune(B, y, n_true=n, penalty=3, mask=mask)
    assert beta[2] == 0, "masked term should remain zero"


def test_ridge():
    """Ridge regularization should shrink coefficients toward zero."""
    rng = np.random.RandomState(4)
    n = 200
    x = rng.randn(n)
    B = np.column_stack([np.ones(n), x])
    y = B @ np.array([5.0, 10.0])

    beta_no_ridge = prune(B, y, n_true=n, penalty=3, ridge=0)
    beta_ridge = prune(B, y, n_true=n, penalty=3, ridge=0.5)
    assert np.linalg.norm(beta_ridge) < np.linalg.norm(beta_no_ridge)


def test_single_term():
    """Pruning a single-term model (intercept only) should still work."""
    n = 100
    B = np.ones((n, 1))
    y = np.full(n, 3.0)

    beta = prune(B, y, n_true=n, penalty=3)
    np.testing.assert_allclose(beta, [3.0], atol=1e-5)


def test_weights():
    """Passing w should be equivalent to pre-scaling B and y by sqrt(w)."""
    rng = np.random.RandomState(5)
    n = 300
    B = np.column_stack([np.ones(n), rng.randn(n), rng.randn(n)])
    y = 1.0 + 2.0 * B[:, 1] - 0.5 * B[:, 2]
    w = rng.uniform(0.1, 2.0, n)

    beta_w = prune(B, y, w=w, n_true=n, penalty=3)
    sw = np.sqrt(w)
    beta_manual = prune(B * sw[:, None], y * sw, n_true=n, penalty=3)
    np.testing.assert_allclose(beta_w, beta_manual, atol=1e-6)


def test_full_elimination_finds_global_best():
    """
    Verify that full elimination can find a better model than greedy stopping.
    """
    rng = np.random.RandomState(42)
    n = 50
    x1 = rng.randn(n)
    x2 = rng.randn(n)
    shared = rng.randn(n) * 0.15
    noise_a = shared + rng.randn(n) * 0.01
    noise_b = shared + rng.randn(n) * 0.01

    B = np.column_stack([np.ones(n), x1, x2, noise_a, noise_b])
    y = 1.0 + 2.0 * x1 - 1.5 * x2

    beta = prune(B, y, n_true=n, penalty=3)
    np.testing.assert_allclose(beta[3], 0, atol=1e-10)
    np.testing.assert_allclose(beta[4], 0, atol=1e-10)
    assert abs(beta[0]) > 0.1
    assert abs(beta[1]) > 0.1
    assert abs(beta[2]) > 0.1


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
