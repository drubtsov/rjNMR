from __future__ import annotations

import numpy as np
import pytest

from rjnmr import Component, simulate_fid
from rjnmr.baselines import estimate_order, hankel_singular_values, matrix_pencil
from rjnmr.init import initial_poles, peak_poles
from rjnmr.preprocess import normalize, select_window, trim_to_maximum

COMPS = [
    Component(-1.5, 0.01, 8.0, 0.2),
    Component(0.3, 0.02, 5.0, -1.0),
    Component(2.2, 0.005, 3.0, 2.0),
]


def test_matrix_pencil_exact_on_noiseless_data() -> None:
    y = simulate_fid(COMPS, 200, 0.0)
    est = matrix_pencil(y, 3)
    for e, t in zip(est, COMPS, strict=True):
        assert e.frequency == pytest.approx(t.frequency, abs=1e-8)
        assert e.damping == pytest.approx(t.damping, abs=1e-8)
        assert e.amplitude == pytest.approx(t.amplitude, rel=1e-7)
        assert e.phase == pytest.approx(t.phase, abs=1e-7)


@pytest.mark.parametrize("method", ["mdl", "aic"])
def test_order_selection(method: str) -> None:
    y = simulate_fid(COMPS, 300, 0.01, seed=1)
    est = estimate_order(y, method, max_order=20)  # type: ignore[arg-type]
    assert est.criterion.size == 21
    if method == "mdl":
        assert est.order == 3
    else:  # AIC is known to overestimate the order (as the paper's comparison shows)
        assert est.order >= 3
    assert len(matrix_pencil(y, criterion=method)) == est.order  # type: ignore[arg-type]


def test_order_selection_errors() -> None:
    with pytest.raises(ValueError, match="aic"):
        estimate_order(np.ones(20, complex), "bic")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="pencil"):
        matrix_pencil(np.ones(10, complex), 1, pencil=20)
    assert matrix_pencil(np.ones(10, complex), 0) == []
    assert hankel_singular_values(np.ones(30, complex)).size == 11


def test_peak_poles_finds_lines() -> None:
    y = simulate_fid(COMPS, 512, 0.01, seed=2)
    z = peak_poles(y)
    freqs = np.angle(z)
    for c in COMPS:
        assert np.min(np.abs(freqs - c.frequency)) < 2 * np.pi / 512
    assert peak_poles(y, max_peaks=1).size == 1


@pytest.mark.parametrize("method", ["empty", "random", "peaks", "matrix_pencil"])
def test_initial_poles(method: str) -> None:
    y = simulate_fid(COMPS, 256, 0.01, seed=3)
    z = initial_poles(y, method, seed=0)  # type: ignore[arg-type]
    assert np.all(np.abs(z) < 1)
    assert {"empty": 0, "random": 1}.get(method, 3) == z.size
    with pytest.raises(ValueError, match="unknown"):
        initial_poles(y, "magic")  # type: ignore[arg-type]


def test_preprocess() -> None:
    y = np.array([0, 1, 5, 2, 1], complex)
    t, i = trim_to_maximum(y)
    assert i == 2 and t[0] == 5
    np.testing.assert_array_equal(select_window(y, 1, 2), [1, 5])
    with pytest.raises(ValueError):
        select_window(y, 3, 5)
    with pytest.raises(ValueError):
        select_window(y, 10)
    z, s = normalize(y + 3)
    assert abs(z.mean()) < 1e-12 and np.std(z) == pytest.approx(1)
    np.testing.assert_allclose(s.apply(y + 3), z)
    assert s.amplitude_to_original(1.0) == pytest.approx(s.scale)
    _, s0 = normalize(np.zeros(3, complex))
    assert s0.scale == 1.0
