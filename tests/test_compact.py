"""Tests for mars.compact()."""

import numpy as np
from mars import compact, expand

# Model node dtype (same as fit() output)
_dtype = [
    ("type", "S1"), ("basis", "i4"), ("input", "i4"), ("hinge", "f8"),
    ("r2", "f4"), ("r2_cv", "f4"), ("order", "i4"), ("time", "f4"),
]


def _node(type, basis=0, input=0, hinge=np.nan):
    return (type, basis, input, hinge, 0, 0, 0, 0)


def test_no_pruning():
    """When all betas are nonzero, nothing changes."""
    model = np.array([_node("i"), _node("l", 0, 0), _node("+", 1, 1, 3.0)], dtype=_dtype)
    beta = np.array([1.0, 2.0, 3.0])
    new_model, new_beta = compact(model, beta)
    assert len(new_model) == 3
    np.testing.assert_array_equal(new_beta, beta)


def test_prune_leaf():
    """Pruning a leaf node removes just that node."""
    model = np.array([_node("i"), _node("l", 0, 0), _node("+", 0, 1, 2.0)], dtype=_dtype)
    beta = np.array([1.0, 2.0, 0.0])
    new_model, new_beta = compact(model, beta)
    assert len(new_model) == 2
    np.testing.assert_array_equal(new_beta, [1.0, 2.0])
    assert new_model[1]["basis"] == 0


def test_parent_kept_for_child():
    """A parent with beta=0 is kept if a child needs it."""
    # intercept -> linear (pruned) -> hinge (kept)
    model = np.array([
        _node("i"),
        _node("l", 0, 0),
        _node("+", 1, 1, 5.0),
    ], dtype=_dtype)
    beta = np.array([1.0, 0.0, 3.0])
    new_model, new_beta = compact(model, beta)
    assert len(new_model) == 3
    # Parent's beta stays zero
    assert new_beta[1] == 0.0
    # Child still points to its parent
    assert new_model[2]["basis"] == 1


def test_indices_remapped():
    """Parent indices are remapped after removing intermediate nodes."""
    # 0:intercept -> 1:linear(kept) -> 3:hinge(kept)
    # 0:intercept -> 2:linear(pruned, no children)
    model = np.array([
        _node("i"),
        _node("l", 0, 0),
        _node("l", 0, 1),
        _node("+", 1, 2, 1.0),
    ], dtype=_dtype)
    beta = np.array([1.0, 2.0, 0.0, 4.0])
    new_model, new_beta = compact(model, beta)
    assert len(new_model) == 3
    np.testing.assert_array_equal(new_beta, [1.0, 2.0, 4.0])
    # Node 3 (now index 2) should point to node 1 (still index 1)
    assert new_model[2]["basis"] == 1


def test_deep_chain_ancestor_kept():
    """Ancestors multiple levels up are kept for a deep surviving node."""
    # 0:intercept -> 1:linear -> 2:hinge -> 3:hinge (only 3 has nonzero beta)
    model = np.array([
        _node("i"),
        _node("l", 0, 0),
        _node("+", 1, 1, 2.0),
        _node("-", 2, 2, 4.0),
    ], dtype=_dtype)
    beta = np.array([0.0, 0.0, 0.0, 5.0])
    new_model, new_beta = compact(model, beta)
    # All 4 nodes kept (entire chain needed)
    assert len(new_model) == 4
    np.testing.assert_array_equal(new_beta, [0.0, 0.0, 0.0, 5.0])


def test_all_pruned():
    """If all betas are zero, only nodes needed as ancestors survive (none)."""
    model = np.array([_node("i"), _node("l", 0, 0)], dtype=_dtype)
    beta = np.array([0.0, 0.0])
    new_model, new_beta = compact(model, beta)
    assert len(new_model) == 0
    assert len(new_beta) == 0


def test_expand_consistent():
    """Expanding with the compacted model gives the same predictions."""
    rng = np.random.RandomState(7)
    n = 50
    X = rng.randn(n, 3).astype("f")

    model = np.array([
        _node("i"),
        _node("l", 0, 0),       # x0
        _node("l", 0, 1),       # x1 (will be pruned)
        _node("+", 1, 2, 0.5),  # max(x2-0.5,0) * x0
    ], dtype=_dtype)
    beta = np.array([2.0, 3.0, 0.0, -1.0])

    B_full = expand(X, model)
    y_full = B_full @ beta

    new_model, new_beta = compact(model, beta)
    B_compact = expand(X, new_model)
    y_compact = B_compact @ new_beta

    np.testing.assert_allclose(y_compact, y_full, atol=1e-6)


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
