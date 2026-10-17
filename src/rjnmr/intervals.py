"""Band-wise deconvolution of a full spectrum (2010 toolbox pipeline).

Fitting a whole high-resolution spectrum with one model is slow, because
the cost of each MCMC sweep grows with the number of components. Lightly damped
sinusoids are well localised in frequency, so the 2010 toolbox (``rjNMR_DEMO_script.m``)
instead:

1. splits the spectrum into bands, each weighted by a raised-cosine window
   (``rCosFilter.m``) that is flat inside the band and tapers smoothly to zero
   outside it;
2. fits each band separately with the RJ-MCMC sampler (``rjNMRMain.m``),
   starting from detected peaks, with ``Λ = 4 × number of peaks`` and a
   baseline offset component;
3. keeps only the MAP components that fall inside the flat part of each band;
4. merges near-duplicate components (``compCleanup.m``) and re-estimates all
   amplitudes on the whole signal (``getPeakWeightAmps.m``).

Here step 2 uses :class:`~rjnmr.design.WindowedSpectralDesign`, the exact
windowed-spectrum likelihood. Poles and amplitudes therefore come out directly
in global coordinates, and the decimation and rescaling of the Matlab script
(``wglob = … + w·length(iffy)/N``) are not needed.
"""

from __future__ import annotations

import dataclasses
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .config import SamplerConfig
from .model import Component, poles_from_components, synthesize
from .peaks import detect_peaks
from .postprocess import merge_close_components, refit_amplitudes
from .proposals import in_band
from .results import RJMCMCResult
from .sampler import RJMCMCSampler

__all__ = [
    "Band",
    "DeconvolutionResult",
    "auto_bands",
    "deconvolve",
    "raised_cosine_window",
]

ComplexArray = NDArray[np.complex128]
FloatArray = NDArray[np.float64]

Band = tuple[float, float]


def _wrap(x: FloatArray | float) -> Any:
    return (np.asarray(x) + math.pi) % (2.0 * math.pi) - math.pi


def raised_cosine_window(n: int, low: float, high: float, rolloff: float = 0.5) -> FloatArray:
    """Raised-cosine window over the ``n`` DFT bins (natural order) (``rCosFilter.m``).

    The window is 1 for frequencies in ``[low, high]`` (rad/sample) and falls
    to 0 along a half-cosine over a further ``rolloff/(1 - rolloff)`` times
    the half-width on each side. ``rolloff = 0`` gives a rectangular window.
    The band may straddle ±π.
    """
    if not 0.0 <= rolloff < 1.0:
        raise ValueError("rolloff must lie in [0, 1)")
    if not 0.0 < high - low < 2.0 * math.pi:
        raise ValueError("need 0 < high - low < 2π")
    omega = 2.0 * math.pi * np.fft.fftfreq(n)
    center = 0.5 * (low + high)
    half = 0.5 * (high - low)
    d = np.abs(_wrap(omega - center))
    lobe = half * 2.0 * rolloff / (1.0 - rolloff)
    w = np.zeros(n)
    w[d <= half + 1e-12] = 1.0
    if lobe > 0:
        in_lobe = (d > half + 1e-12) & (d <= half + lobe)
        w[in_lobe] = 0.5 * (1.0 + np.cos(math.pi * (d[in_lobe] - half) / lobe))
    return w


def auto_bands(
    y: ArrayLike,
    *,
    max_width: float = 0.2,
    threshold: float = 4.0,
    padding: int = 8,
    smoothing: int = 10,
) -> list[Band]:
    """Choose bands that cover the signal-bearing parts of the spectrum.

    Bins whose smoothed magnitude exceeds ``threshold`` times the median
    magnitude (a robust noise level) are taken as signal, padded by
    ``padding`` bins, and grouped into runs. Runs wider than ``max_width``
    rad/sample are split at the lowest points of the smoothed spectrum, so
    that band edges fall between peaks.
    """
    yc = np.asarray(y, dtype=np.complex128).ravel()
    n = yc.size
    mag = np.abs(np.fft.fftshift(np.fft.fft(yc)))
    tri = np.concatenate([np.arange(1, smoothing + 1), np.arange(smoothing - 1, 0, -1)])
    smooth = np.convolve(mag, tri / tri.sum(), mode="same")
    mask = smooth > threshold * float(np.median(mag))
    if not mask.any():
        return []
    mask = np.convolve(mask.astype(float), np.ones(2 * padding + 1), mode="same") > 0
    edges = np.flatnonzero(np.diff(np.concatenate([[0], mask.astype(int), [0]])))
    runs = list(zip(edges[::2], edges[1::2] - 1, strict=True))
    max_bins = max(4, int(max_width * n / (2.0 * math.pi)))
    pieces: list[tuple[int, int]] = []
    for a, b in runs:
        while b - a + 1 > max_bins:
            lo_cut, hi_cut = a + max_bins // 2, a + max_bins
            cut = lo_cut + int(np.argmin(smooth[lo_cut : hi_cut + 1]))
            pieces.append((a, cut))
            a = cut + 1
        pieces.append((a, b))
    shift = n // 2
    step = 2.0 * math.pi / n
    return [(float((a - shift - 0.5) * step), float((b - shift + 0.5) * step)) for a, b in pieces]


@dataclass(slots=True)
class DeconvolutionResult:
    """Output of :func:`deconvolve`."""

    data: ComplexArray
    components: list[Component]
    bands: list[Band]
    band_results: list[RJMCMCResult]
    rolloff: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def model_signal(self) -> ComplexArray:
        """Noise-free FID of all components."""
        z, a = poles_from_components(self.components)
        return synthesize(z, a, self.data.size)

    def residual(self) -> ComplexArray:
        """Data minus model."""
        out: ComplexArray = self.data - self.model_signal()
        return out

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly summary."""
        return {
            "n_points": int(self.data.size),
            "rolloff": self.rolloff,
            "bands": [list(b) for b in self.bands],
            "components": [dataclasses.asdict(c) for c in self.components],
            "band_k_map": [r.k_map if len(r.trace) else 0 for r in self.band_results],
            "metadata": self.metadata,
        }

    def save_json(self, path: str | Path) -> None:
        """Write :meth:`to_dict` as JSON."""
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    def report(self, spectral_width: float | None = None) -> str:
        """Component table."""
        unit = "Hz" if spectral_width else "rad"
        lines = [
            f"{len(self.components)} components from {len(self.bands)} bands",
            f"{'#':>3} {'freq [' + unit + ']':>14} {'damping':>12} {'amplitude':>12} {'phase':>8}",
        ]
        for i, c in enumerate(self.components, 1):
            f = c.frequency_hz(spectral_width) if spectral_width else c.frequency
            lines.append(f"{i:>3} {f:14.6g} {c.damping:12.5g} {c.amplitude:12.5g} {c.phase:8.3f}")
        return "\n".join(lines)


def _fit_band(
    args: tuple[
        ComplexArray, Band, SamplerConfig, float, float | None, str, int | np.random.SeedSequence
    ],
) -> tuple[RJMCMCResult, list[Component]]:
    y, band, config, rolloff, expected_per_peak, init, seed = args
    lo, hi = band
    n = y.size
    window = raised_cosine_window(n, lo, hi, rolloff)
    cfg = config
    poles = np.zeros(0, dtype=np.complex128)
    if init == "peaks":
        spectrum = np.fft.fft(y) * window
        peaks = detect_peaks(np.fft.ifft(spectrum))
        sampler_probe = RJMCMCSampler(
            y, dataclasses.replace(cfg, n_iterations=0, burn_in=0), window=window
        )
        support = sampler_probe.support
        poles = np.array(
            [z for z in peaks.poles() if support.contains(complex(z))], dtype=np.complex128
        )
        if expected_per_peak is not None:
            cfg = dataclasses.replace(
                cfg, expected_components=expected_per_peak * max(1, poles.size)
            )
        poles = poles[: cfg.max_components]
    result = RJMCMCSampler(y, cfg, seed=np.random.default_rng(seed), window=window).run(poles)
    kept: list[Component] = []
    if len(result.trace):
        kept = [c for c in result.map_components() if in_band(c.frequency, lo, hi - lo)]
    return result, kept


def deconvolve(
    y: ArrayLike,
    bands: Sequence[Band] | None = None,
    config: SamplerConfig | None = None,
    *,
    rolloff: float = 0.2,
    init: Literal["peaks", "empty"] = "peaks",
    expected_per_peak: float | None = 4.0,
    merge: bool = True,
    refit: Literal["lstsq", "peak_weighted"] | None = "lstsq",
    seed: int | None = None,
    n_jobs: int | None = 1,
) -> DeconvolutionResult:
    """Deconvolve an FID band by band (2010 toolbox pipeline).

    Parameters
    ----------
    y
        Complex FID, ideally phase-corrected (:func:`rjnmr.phasing.auto_phase`).
    bands
        ``(low, high)`` frequency bands in rad/sample whose flat regions tile
        the part of the spectrum to analyse. ``None`` uses :func:`auto_bands`.
    config
        Sampler settings for every band. The default enables the baseline
        component and the polar random walk, requires ``min_independence=1e-3``
        (no unresolvable pairs of poles), bounds the damping at 0.05
        rad/sample (FWHM ≈ SW/60), and runs 2000 + 2000 iterations. Without a
        bound, pairs of very broad, mutually cancelling components can soak up
        baseline distortions. The Matlab sampler avoided them only because its
        births had near-zero damping.
    rolloff
        Raised-cosine roll-off (Matlab demo: 0.2).
    init
        Start each band from detected peaks (Matlab) or from an empty model.
    expected_per_peak
        Set ``Λ`` to this multiple of the number of detected peaks per band
        (Matlab: 4). ``None`` keeps ``config.expected_components``.
    merge, refit
        Merge near-duplicate components and re-estimate amplitudes jointly on
        the whole FID (``None`` keeps the per-band posterior-mean amplitudes).
    seed
        Seed for reproducibility; each band gets an independent stream.
    n_jobs
        Number of processes used to fit bands in parallel (``None`` = all cores).
    """
    yc = np.ascontiguousarray(np.asarray(y, dtype=np.complex128).ravel())
    cfg = config or SamplerConfig(
        n_iterations=2000,
        burn_in=2000,
        baseline=True,
        update_kernel="polar",
        max_damping=0.05,
        min_independence=1e-3,
    )
    band_list = list(auto_bands(yc) if bands is None else bands)
    seeds = np.random.SeedSequence(seed).spawn(len(band_list))
    jobs = [
        (yc, (float(lo), float(hi)), cfg, rolloff, expected_per_peak, init, s)
        for (lo, hi), s in zip(band_list, seeds, strict=True)
    ]
    if n_jobs == 1 or len(jobs) <= 1:
        outputs = [_fit_band(j) for j in jobs]
    else:
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(max_workers=n_jobs) as pool:
            outputs = list(pool.map(_fit_band, jobs))
    components = sorted((c for _, kept in outputs for c in kept), key=lambda c: c.frequency)
    if merge:
        components = merge_close_components(components)
    if refit is not None and components:
        components = refit_amplitudes(yc, components, method=refit)
    return DeconvolutionResult(
        data=yc,
        components=components,
        bands=band_list,
        band_results=[r for r, _ in outputs],
        rolloff=rolloff,
    )
