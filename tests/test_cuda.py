"""End-to-end check of the CUDA forward pass (hinges included) against the CPU.

Skipped unless marslib was built with CUDA and a device is present. The GPU
fit uses blocked-f32 GEMMs for the orthonormalize and a chunked f64 recurrence
for the hinge sweep, so it is not bit-identical to the CPU fit: on a problem
with well-separated planted terms the two must still select the same basis
(type, parent, input) with hinge cuts that agree to f32 resolution, and reach
the same R² to ~1e-4.
"""

import numpy as np
import pytest

import mars


def _cuda_usable():
    X = np.asfortranarray(np.random.RandomState(0).randn(200, 3).astype("f"))
    y = X[:, 0].copy()
    try:
        mars.fit(X, y, max_terms=3, cuda=True)
    except RuntimeError as e:  # "built without CUDA support" or a driver error
        return False, str(e)
    return True, ""


def _problem(n=40_000, p=8, seed=1):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, p).astype("f")
    X[:, 0] = np.round(X[:, 0] * 4) / 4  # ties in the sort
    y = (
        np.maximum(X[:, 0] - 0.3, 0)
        - 0.8 * np.maximum(0.5 - X[:, 1], 0)
        + 0.6 * X[:, 2] * np.maximum(X[:, 3], 0)
        + 0.4 * X[:, 4]
        + 0.3 * rng.randn(n)
    ).astype("f")
    return np.asfortranarray(X), y


def test_fit_cuda_hinges_match_cpu():
    """The GPU hinge sweep: same basis, same cuts, same R² as the CPU fit."""
    ok, why = _cuda_usable()
    if not ok:
        pytest.skip(f"CUDA path unavailable: {why}")
    X, y = _problem()
    kw = dict(max_terms=15, threads=1)
    m_cpu = mars.fit(X, y, cuda=False, **kw)
    m_gpu = mars.fit(X, y, cuda=True, **kw)
    assert len(m_gpu) == len(m_cpu)
    np.testing.assert_array_equal(m_gpu["type"], m_cpu["type"])
    np.testing.assert_array_equal(m_gpu["basis"], m_cpu["basis"])
    np.testing.assert_array_equal(m_gpu["input"], m_cpu["input"])
    hc, hg = m_cpu["hinge"], m_gpu["hinge"]
    np.testing.assert_array_equal(np.isnan(hc), np.isnan(hg))
    sel = ~np.isnan(hc)
    np.testing.assert_allclose(hg[sel], hc[sel], rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(m_gpu["r2"], m_cpu["r2"], atol=1e-4)


def test_fit_cuda_linear_only_matches_cpu():
    """The batched linear_only path (pre-existing) reorders near-tied terms on
    correlated features (see CLAUDE.md, 2026-06-16), so only the fit quality
    is compared: same term count, same R² trajectory to ~1e-4."""
    ok, why = _cuda_usable()
    if not ok:
        pytest.skip(f"CUDA path unavailable: {why}")
    X, y = _problem()
    kw = dict(max_terms=15, linear_only=True, threads=1)
    m_cpu = mars.fit(X, y, cuda=False, **kw)
    m_gpu = mars.fit(X, y, cuda=True, **kw)
    assert len(m_gpu) == len(m_cpu)
    np.testing.assert_allclose(m_gpu["r2"], m_cpu["r2"], atol=1e-4)
