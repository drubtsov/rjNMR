"""Tests for the ports of the 2010 toolbox: peaks, phasing, bands, post-processing."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from rjnmr import Component, SamplerConfig, simulate_fid
from rjnmr.intervals import auto_bands, deconvolve, raised_cosine_window
from rjnmr.peaks import detect_peaks, detect_spectrum_peaks, lorentzian_from_bins
from rjnmr.phasing import PhaseCorrection, auto_phase
from rjnmr.postprocess import merge_close_components, refit_amplitudes

COMPS = [
    Component(-1.5, 0.01, 8.0, 0.2),
    Component(0.3, 0.005, 5.0, -1.0),
    Component(2.2, 0.003, 3.0, 2.0),
    Component(3.1, 0.004, 6.0, 1.0),
]


# ---------------------------------------------------------------------------------- peaks ---


def test_detect_peaks_recovers_lorentzian_parameters() -> None:
    y = simulate_fid(COMPS, 2048, 1e-4, seed=1)
    peaks = detect_peaks(y)
    assert len(peaks) == len(COMPS)
    for p, c in zip(peaks.components(), sorted(COMPS, key=lambda c: c.frequency), strict=True):
        assert p.frequency == pytest.approx(c.frequency, abs=1e-3)
        assert p.damping == pytest.approx(c.damping, rel=0.05)
        assert p.amplitude == pytest.approx(c.amplitude, rel=0.03)
        assert p.phase == pytest.approx(c.phase, abs=0.03)
    assert peaks.taken.size > 0
    assert np.all(np.abs(peaks.poles()) < 1)
    assert len(detect_peaks(y, max_peaks=2)) == 2


def test_lorentzian_three_point_fit_is_exact() -> None:
    n, w0, d, a = 512, 1.2345, 0.02, 3.0
    omega = 2 * math.pi * np.arange(n) / n
    spec = a / (d + 1j * (w0 - omega))  # exact Lorentzian, |Y|² = a²/(d² + Δ²)
    k = round(w0 * n / (2 * math.pi))
    w, dd, aa, _ = lorentzian_from_bins(spec, k)
    assert w == pytest.approx(w0, abs=1e-10)
    assert dd == pytest.approx(d, rel=1e-8)
    assert aa == pytest.approx(a, rel=1e-8)


def test_detect_peaks_on_noise_only_and_region() -> None:
    assert len(detect_spectrum_peaks(np.ones(64, complex))) == 0
    y = simulate_fid(COMPS, 1024, 1e-4, seed=0)
    spec = np.fft.fft(y)
    all_peaks = detect_spectrum_peaks(spec)
    some = detect_spectrum_peaks(spec, region=(0, 300))
    assert 0 < len(some) < len(all_peaks)


# -------------------------------------------------------------------------------- phasing ---


@pytest.mark.parametrize("delay", [0.0, 5.0, 12.0])
def test_auto_phase_removes_known_phase_error(delay: float) -> None:
    rng = np.random.default_rng(7)
    n = 4096
    lines = [
        Component(f, rng.uniform(0.002, 0.01), rng.uniform(2, 10), 0.0)
        for f in np.sort(rng.uniform(-2.8, 2.8, 12))
    ]
    clean = simulate_fid(lines, n, 1e-3, seed=3)
    error = PhaseCorrection(phase0=0.9, slope=-2 * math.pi * delay / n, pivot=1000.0, n=n)
    phased, corr, peaks = auto_phase(error.apply(clean))
    assert len(peaks) >= 8
    assert corr.group_delay == pytest.approx(delay, abs=0.5)
    s_true, s_est = np.fft.fft(clean), np.fft.fft(phased)
    assert np.corrcoef(s_true.real, s_est.real)[0, 1] > 0.97
    with pytest.raises(ValueError, match="points"):
        corr.apply(np.ones(10))


def test_auto_phase_requires_peaks() -> None:
    with pytest.raises(ValueError, match="no peaks"):
        auto_phase(np.zeros(256, complex))


# -------------------------------------------------------------------------------- windows ---


def test_raised_cosine_window_shape() -> None:
    n = 1000
    w = raised_cosine_window(n, -0.5, 0.5, rolloff=0.2)
    omega = 2 * math.pi * np.fft.fftfreq(n)
    assert np.all(w[np.abs(omega) <= 0.5] == 1.0)
    lobe = 0.5 * 2 * 0.2 / 0.8
    assert np.all(w[np.abs(omega) > 0.5 + lobe] == 0.0)
    assert np.all((w >= 0) & (w <= 1))
    order = np.argsort(omega)
    assert np.max(np.abs(np.diff(w[order]))) < 0.05  # smooth taper
    rect = raised_cosine_window(n, -0.5, 0.5, rolloff=0.0)
    assert set(np.unique(rect)) == {0.0, 1.0}
    wrap = raised_cosine_window(n, 3.0, 3.5, 0.2)
    assert wrap[n // 2] == 1.0  # ω = -π is inside [3.0, 3.5] modulo 2π
    with pytest.raises(ValueError, match="rolloff"):
        raised_cosine_window(n, 0, 1, 1.0)
    with pytest.raises(ValueError, match="high - low"):
        raised_cosine_window(n, 1, 0)


def test_auto_bands_cover_peaks() -> None:
    y = simulate_fid(COMPS, 4096, 0.01, seed=0)
    bands = auto_bands(y, max_width=0.1)
    for c in COMPS:
        assert any(lo <= c.frequency <= hi for lo, hi in bands)
    assert all(hi - lo <= 0.1 + 1e-9 + 2 * math.pi / 4096 for lo, hi in bands)
    assert auto_bands(np.zeros(64, complex)) == []


# ---------------------------------------------------------------------- band deconvolution ---


@pytest.fixture(scope="module")
def band_signal() -> tuple[list[Component], np.ndarray]:
    comps = [
        Component(-2.0, 0.004, 8.0, 0.0),
        Component(-1.95, 0.006, 4.0, 0.0),
        Component(0.5, 0.003, 10.0, 0.0),
        Component(0.52, 0.003, 3.0, 0.0),
        Component(2.0, 0.01, 6.0, 0.0),
    ]
    return comps, simulate_fid(comps, 2048, 0.5, seed=2)


def _check_recovery(found: list[Component], truth: list[Component]) -> None:
    """Every true line has a close match; any extra components are small."""
    matched = set()
    for t in truth:
        j = int(np.argmin([abs(f.frequency - t.frequency) for f in found]))
        assert found[j].frequency == pytest.approx(t.frequency, abs=2e-3)
        assert found[j].amplitude == pytest.approx(t.amplitude, rel=0.15)
        matched.add(j)
    smallest = min(t.amplitude for t in truth)
    for j, f in enumerate(found):
        if j not in matched:
            assert f.amplitude < 0.3 * smallest


@pytest.mark.parametrize("amplitude_model", ["complex", "real"])
def test_deconvolve_recovers_components(band_signal, amplitude_model: str) -> None:
    truth, y = band_signal
    cfg = SamplerConfig(
        n_iterations=800,
        burn_in=800,
        baseline=True,
        update_kernel="polar",
        max_damping=0.05,
        min_independence=1e-3,
        amplitude_model=amplitude_model,  # type: ignore[arg-type]
    )
    res = deconvolve(y, auto_bands(y), cfg, seed=1)
    _check_recovery(res.components, truth)
    assert len(res.band_results) == len(res.bands)
    assert np.mean(np.abs(res.residual()) ** 2) == pytest.approx(1.0, rel=0.25)


def test_deconvolve_parallel_matches_serial_and_serialises(band_signal, tmp_path: Path) -> None:
    _, y = band_signal
    cfg = SamplerConfig(n_iterations=100, burn_in=100, baseline=True)
    bands = [(-2.1, -1.85), (0.4, 0.6)]
    a = deconvolve(y, bands, cfg, seed=5, n_jobs=1, refit=None, merge=False)
    b = deconvolve(y, bands, cfg, seed=5, n_jobs=2, refit=None, merge=False)
    assert [c.frequency for c in a.components] == [c.frequency for c in b.components]
    path = tmp_path / "dec.json"
    a.save_json(path)
    payload = json.loads(path.read_text())
    assert len(payload["components"]) == len(a.components)
    assert "components from 2 bands" in a.report()
    assert "Hz" in a.report(spectral_width=1000.0)


# ------------------------------------------------------------------------- post-processing ---


def test_merge_close_components() -> None:
    comps = [
        Component(1.0, 0.010, 3.0, 0.0),
        Component(1.0004, 0.011, 1.0, 0.0),
        Component(1.0008, 0.012, 1.0, 0.0),  # chained to the first through the second
        Component(2.0, 0.01, 5.0, 0.0),
        Component(2.0, 0.03, 5.0, 0.0),  # same frequency, very different damping
    ]
    merged = merge_close_components(comps)
    assert len(merged) == 3
    first = merged[0]
    assert first.amplitude == pytest.approx(5.0)
    w = np.array([9.0, 1.0, 1.0])
    z = np.array([c.pole for c in comps[:3]])
    assert first.pole == pytest.approx(complex(np.sum(w * z) / w.sum()))
    assert merge_close_components(comps[:1]) == comps[:1]


def test_refit_amplitudes() -> None:
    y = simulate_fid(COMPS, 512, 0.0)
    guess = [Component(c.frequency, c.damping, 1.0, 0.0) for c in COMPS]
    fitted = refit_amplitudes(y, guess)
    for f, t in zip(fitted, sorted(COMPS, key=lambda c: c.frequency), strict=True):
        assert f.amplitude == pytest.approx(t.amplitude, rel=1e-8)
        assert f.phase == pytest.approx(t.phase, abs=1e-8)
    zero = [Component(c.frequency, c.damping, c.amplitude, 0.0) for c in COMPS]
    yz = simulate_fid(zero, 4096, 0.0)
    pw = refit_amplitudes(yz, guess, method="peak_weighted")
    for f, t in zip(pw, sorted(zero, key=lambda c: c.frequency), strict=True):
        assert f.amplitude == pytest.approx(t.amplitude, rel=1e-6)
    assert refit_amplitudes(y, []) == []
    with pytest.raises(ValueError, match="method"):
        refit_amplitudes(y, guess, method="median")  # type: ignore[arg-type]
