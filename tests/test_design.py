from __future__ import annotations

import math

import numpy as np
import pytest

from rjnmr.design import TimeDesign, WindowedSpectralDesign, _contiguous_support, make_design
from rjnmr.intervals import raised_cosine_window
from rjnmr.model import basis

N = 256
Z = np.exp(-np.array([0.01, 0.03, 0.0]) + 1j * np.array([0.3, -1.0, 2.0]))


@pytest.fixture
def y(rng: np.random.Generator) -> np.ndarray:
    return rng.standard_normal(N) + 1j * rng.standard_normal(N)


def _explicit(y: np.ndarray, window: np.ndarray | None, baseline: bool):
    """Brute-force Gram, projection and energy of a design."""
    D = basis(Z, N)
    if window is None:
        F, d = D, y
        v = np.zeros(N, complex)
        v[0] = 1.0
    else:
        bins = np.flatnonzero(window > 0)
        F = np.fft.fft(D, axis=0)[bins] * window[bins, None] / math.sqrt(N)
        d = np.fft.fft(y)[bins] * window[bins] / math.sqrt(N)
        v = window[bins] / np.linalg.norm(window[bins])
    P = np.eye(len(d)) - (np.outer(v, v.conj()) if baseline else 0)
    F, d = P @ F, P @ d
    return F.conj().T @ F, F.conj().T @ d, float(np.vdot(d, d).real)


@pytest.mark.parametrize("baseline", [False, True])
@pytest.mark.parametrize("windowed", [False, True])
def test_design_matches_brute_force(y: np.ndarray, baseline: bool, windowed: bool) -> None:
    window = raised_cosine_window(N, 0.1, 0.9, 0.3) if windowed else None
    design = make_design(y, window=window, baseline=baseline)
    G, b, yy = _explicit(y, window, baseline)
    zz = Z[:2] if windowed else Z
    np.testing.assert_allclose(design.gram(zz), G[: zz.size, : zz.size], atol=1e-10)
    np.testing.assert_allclose(design.project(zz), b[: zz.size], atol=1e-10)
    np.testing.assert_allclose(design.gram_column(zz, complex(Z[2])), G[: zz.size, 2], atol=1e-10)
    assert design.project_one(complex(Z[2])) == pytest.approx(b[2], abs=1e-10)
    assert design.yy == pytest.approx(yy, rel=1e-12)


@pytest.mark.parametrize("baseline", [False, True])
def test_flat_window_equals_time_domain(y: np.ndarray, baseline: bool) -> None:
    t = TimeDesign(y, baseline=baseline)
    w = WindowedSpectralDesign(y, np.ones(N), baseline=baseline)
    np.testing.assert_allclose(t.gram(Z), w.gram(Z), atol=1e-10)
    np.testing.assert_allclose(t.project(Z), w.project(Z), atol=1e-10)
    assert t.yy == pytest.approx(w.yy)
    assert t.n_obs == w.n_obs == N - baseline


def test_windowed_band_and_support() -> None:
    w = raised_cosine_window(N, 3.0, 3.4, 0.0)  # straddles ±π
    d = WindowedSpectralDesign(np.ones(N, complex), w)
    start, width = d.frequency_support
    assert width == pytest.approx(2 * math.pi * d.bins.size / N)
    assert (start - 3.0 + math.pi) % (2 * math.pi) - math.pi == pytest.approx(
        0.0, abs=2 * math.pi / N
    )
    np.testing.assert_array_equal(np.diff(d.bins) % N, 1)


def test_support_helpers_and_errors() -> None:
    mask = np.zeros(10, bool)
    mask[[8, 9, 0, 1]] = True
    np.testing.assert_array_equal(_contiguous_support(mask), [8, 9, 0, 1])
    np.testing.assert_array_equal(_contiguous_support(np.ones(4, bool)), [0, 1, 2, 3])
    mask[5] = True
    with pytest.raises(ValueError, match="contiguous"):
        _contiguous_support(mask)
    with pytest.raises(ValueError, match="bins"):
        WindowedSpectralDesign(np.ones(8), np.ones(4))
    with pytest.raises(ValueError, match="non-negative"):
        WindowedSpectralDesign(np.ones(8), np.zeros(8))
