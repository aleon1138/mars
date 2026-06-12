"""End-to-end checks for bf16 basis storage (Phase 1: linear_only only).

bf16 narrows the internal basis matrices B/Bo/Bx; the public X/y/w API stays
f32. The hot loops still widen to f32/f64 for all arithmetic, so on a
well-conditioned linear fit bf16 recovers the same real terms as f32, just to
the coarser bf16 storage floor.
"""
import os
# The mac dev box hits the dual-libomp init abort once a fit() enters the
# OpenMP eval region (numpy already loaded its libomp). Harmless on Linux/CI.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pytest
import mars


def _linear_dataset(seed=0, n=8000, p=10):
    rng = np.random.default_rng(seed)
    X = np.asfortranarray(rng.standard_normal((n, p)).astype("f"))
    beta = np.zeros(p, dtype="f")
    signal = [0, 2, 4, 6]
    beta[signal] = [1.5, -0.8, 0.6, 0.4]
    y = (X @ beta + 0.05 * rng.standard_normal(n)).astype("f")
    return X, y, set(signal)


def test_bf16_requires_linear_only():
    # Gating is enforced in fit() before any OpenMP work runs.
    X, y, _ = _linear_dataset()
    with pytest.raises(ValueError):
        mars.fit(X, y, basis_dtype="bf16")  # linear_only defaults to False


def test_unknown_basis_dtype_rejected():
    X, y, _ = _linear_dataset()
    with pytest.raises(Exception):
        mars.fit(X, y, linear_only=True, basis_dtype="f16")


def test_bf16_recovers_linear_signal():
    X, y, signal = _linear_dataset()
    kw = dict(linear_only=True, threads=1, max_terms=20)
    mf = mars.fit(X, y, basis_dtype="f32", **kw)
    mb = mars.fit(X, y, basis_dtype="bf16", **kw)

    def linear_inputs(m):
        return {int(i) for t, i in zip(m["type"], m["input"]) if t == b"l"}

    # bf16 recovers every true signal input that f32 does.
    assert signal <= linear_inputs(mf)
    assert signal <= linear_inputs(mb)

    # The meaningful prefix (intercept + the real terms) is selected in the same
    # order: bf16 widens to f32 before every computation, so while real signal
    # dominates the search the two paths agree.
    k = 1 + len(signal)
    pf = list(zip(mf["type"][:k], mf["basis"][:k], mf["input"][:k]))
    pb = list(zip(mb["type"][:k], mb["basis"][:k], mb["input"][:k]))
    assert pf == pb

    # r2 over that prefix tracks to the bf16 storage floor (~few e-3).
    assert abs(float(mf["r2"][:k][-1]) - float(mb["r2"][:k][-1])) < 5e-3
