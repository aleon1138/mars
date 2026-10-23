"""Tests for mars.expand() (the marslib.expand kernel)."""

import numpy as np
import pytest

import mars
from mars import expand

MODEL_DTYPE = [
    ("type", "S1"),
    ("basis", "i4"),
    ("input", "i4"),
    ("hinge", "f8"),
    ("beta", "f4"),
    ("r2", "f4"),
    ("r2_cv", "f4"),
    ("order", "i4"),
    ("time", "f4"),
]


def make_model(nodes):
    """Build a model array from (type, basis, input, hinge) tuples."""
    model = np.zeros(len(nodes), dtype=MODEL_DTYPE)
    for i, (t, b, x, h) in enumerate(nodes):
        model[i] = (t, b, x, h, np.nan, 0, 0, 0, 0)
    return model


def random_model(rng, X, m, max_degree=3):
    """A random tree of m nodes over the columns of X with cuts drawn from X."""
    p = X.shape[1]
    nodes = [("i", 0, 0, np.nan)]
    degree = [0]
    for i in range(1, m):
        parents = [k for k in range(i) if degree[k] < max_degree]
        b = int(rng.choice(parents))
        x = int(rng.integers(p))
        t = str(rng.choice(["l", "+", "-"]))
        h = float(X[rng.integers(len(X)), x]) if t != "l" else np.nan
        nodes.append((t, b, x, h))
        degree.append(degree[b] + 1)
    return make_model(nodes)


def expand_reference(X, model):
    """The pre-kernel numpy implementation, kept dtype-faithful.

    Each factor is computed in X's dtype with the cut rounded to that dtype
    first, which is also what the kernel does, so the two agree bit for bit.
    """
    X = np.asarray(X)
    B = np.empty((len(X), len(model)), dtype=X.dtype, order="F")
    for i, node in enumerate(model):
        t, b, x = node["type"], node["basis"], node["input"]
        h = X.dtype.type(node["hinge"])
        if t == b"i":
            B[:, i] = 1
        elif t == b"l":
            B[:, i] = B[:, b] * X[:, x]
        elif t == b"+":
            B[:, i] = B[:, b] * np.maximum(X[:, x] - h, 0)
        elif t == b"-":
            B[:, i] = B[:, b] * np.maximum(h - X[:, x], 0)
        else:
            raise AssertionError(t)
    return B


@pytest.mark.parametrize("order", ["F", "C"])
@pytest.mark.parametrize("dtype", ["f", "d"])
@pytest.mark.parametrize("xorder", ["F", "C"])
def test_matches_reference(order, dtype, xorder):
    """Every (B order, dtype, X order) combination is bit-identical to numpy."""
    rng = np.random.default_rng(0)
    X = np.array(rng.standard_normal((3001, 6)), dtype=dtype, order=xorder)
    model = random_model(rng, X, 40)

    B = expand(X, model, order=order)
    assert B.shape == (3001, 40)
    assert B.dtype == X.dtype
    assert B.flags["F_CONTIGUOUS" if order == "F" else "C_CONTIGUOUS"]
    np.testing.assert_array_equal(B, expand_reference(X, model))


def test_default_order_is_fortran():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((100, 3)).astype("f", order="F")
    model = random_model(rng, X, 5)
    assert expand(X, model).flags.f_contiguous


def test_threads_and_partial_blocks_agree():
    """Many blocks per thread, a ragged last block, 1 vs many threads."""
    rng = np.random.default_rng(2)
    X = np.asfortranarray(rng.standard_normal((20_011, 4)).astype("f"))
    model = random_model(rng, X, 70)  # ~900-row blocks -> 23 blocks
    ref = expand_reference(X, model)
    for order in ("F", "C"):
        for threads in (1, 2, 7, -1):
            B = expand(X, model, order=order, threads=threads)
            np.testing.assert_array_equal(B, ref)


def test_strided_x_view():
    """X need not be contiguous in either axis."""
    rng = np.random.default_rng(3)
    X_full = np.asfortranarray(rng.standard_normal((2000, 8)).astype("f"))
    X = X_full[::2, 1::2]  # (1000, 4), strides (8, 2*2000*4)
    model = random_model(rng, X, 12)
    np.testing.assert_array_equal(expand(X, model), expand_reference(X, model))
    np.testing.assert_array_equal(
        expand(X, model, order="C"), expand_reference(X, model)
    )


def test_hinge_outside_support_is_zero():
    """A hinge entirely below the data is identically zero (not garbage)."""
    X = np.asfortranarray(np.linspace(0, 1, 50, dtype="f")[:, None])
    model = make_model([("i", 0, 0, np.nan), ("-", 0, 0, -1.0), ("+", 1, 0, 0.5)])
    B = expand(X, model)
    assert not B[:, 1].any()
    assert not B[:, 2].any()


def test_non_float_input_is_promoted():
    X = np.arange(12).reshape(4, 3)
    model = make_model([("i", 0, 0, np.nan), ("l", 0, 2, np.nan)])
    B = expand(X, model)
    assert B.dtype == np.float64
    np.testing.assert_array_equal(B[:, 1], X[:, 2])


def test_empty_rows():
    X = np.empty((0, 3), dtype="f")
    model = make_model([("i", 0, 0, np.nan), ("l", 0, 1, np.nan)])
    for order in ("F", "C"):
        B = expand(X, model, order=order)
        assert B.shape == (0, 2)


def test_invalid_order():
    X = np.ones((4, 2), dtype="f")
    model = make_model([("i", 0, 0, np.nan)])
    with pytest.raises(ValueError):
        expand(X, model, order="K")


@pytest.mark.parametrize(
    "nodes",
    [
        [("i", 0, 0, np.nan), ("l", 1, 0, np.nan)],  # parent is itself
        [("i", 0, 0, np.nan), ("l", 0, 0, np.nan), ("+", 2, 0, 0.0)],  # parent later
        [("i", 0, 0, np.nan), ("l", 0, 2, np.nan)],  # input out of range
        [("i", 0, 0, np.nan), ("x", 0, 0, np.nan)],  # unknown type
    ],
)
def test_bad_model_raises(nodes):
    X = np.ones((4, 2), dtype="f")
    with pytest.raises(RuntimeError):
        expand(X, make_model(nodes))


def test_fit_roundtrip():
    """expand() of a fitted model agrees with the reference in both orders,
    and prune() gives the same coefficients on either layout."""
    rng = np.random.default_rng(4)
    n = 4000
    X = np.asfortranarray(rng.standard_normal((n, 3)).astype("f"))
    y = (np.maximum(X[:, 0] - 0.3, 0) * X[:, 1] + X[:, 2]).astype("f")
    model = mars.fit(X, y, max_terms=11, tail_span=0)
    assert len(model) > 1

    ref = expand_reference(X, model)
    BF = expand(X, model, order="F")
    BC = expand(X, model, order="C")
    np.testing.assert_array_equal(BF, ref)
    np.testing.assert_array_equal(BC, ref)

    # prune() is not layout-invariant bit for bit: gram()'s per-chunk f32
    # sgemm rounds differently for a transposed operand, and the hinge pairs
    # make the Gram matrix ill-conditioned enough to turn that into ~1e-3 in
    # the coefficients. The selected terms and the fit are the same.
    beta_f = mars.prune(BF, y)
    beta_c = mars.prune(BC, y)
    np.testing.assert_array_equal(beta_f != 0, beta_c != 0)
    np.testing.assert_allclose(BF @ beta_f, BC @ beta_c, atol=1e-2 * np.std(y))
    for beta in (beta_f, beta_c):
        r2 = 1 - np.var(y - BF @ beta) / np.var(y)
        assert r2 > 0.99


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
