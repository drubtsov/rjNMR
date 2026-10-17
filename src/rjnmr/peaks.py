r"""Spectral peak picking with approximate Lorentzian parameters (2010 toolbox).

Port of ``detectPeaks.m``, ``getApproxPeakParams.m`` and ``getSecDer.m``. Peaks
are local maxima of the magnitude spectrum, located on a triangle-smoothed copy
and snapped to the raw spectrum. Each peak's extent is bounded by inflection
points of the smoothed spectrum. A Lorentzian

.. math:: |Y(\omega)|^2 \approx \frac{A^2}{\alpha^2 + (\omega - \omega_0)^2}

is fitted exactly through the three bins around the maximum, giving the
frequency :math:`\omega_0`, damping :math:`\alpha` and amplitude :math:`A`.
The phase is interpolated from the unwrapped spectral phase at
:math:`\omega_0`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .model import Component

__all__ = ["Peak", "PeakList", "detect_peaks", "detect_spectrum_peaks", "lorentzian_from_bins"]

ComplexArray = NDArray[np.complex128]
FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class Peak:
    """A detected spectral peak.

    ``bin`` is the DFT bin of the maximum and ``start``/``stop`` (inclusive)
    the bins of its neighbourhood. ``frequency`` is in rad/sample (in the
    frequency convention of the spectrum the peak was detected in), and
    ``damping``, ``amplitude`` and ``phase`` are the three-point Lorentzian
    estimates.
    """

    bin: int
    start: int
    stop: int
    frequency: float
    damping: float
    amplitude: float
    phase: float

    def component(self) -> Component:
        """The peak as a model :class:`~rjnmr.model.Component`."""
        return Component(self.frequency, self.damping, self.amplitude, self.phase)


@dataclass(frozen=True, slots=True)
class PeakList:
    """Detected peaks plus the union of their neighbourhoods."""

    peaks: tuple[Peak, ...]
    taken: NDArray[np.int64]  # bins inside any peak neighbourhood

    def __len__(self) -> int:
        return len(self.peaks)

    def __iter__(self):  # type: ignore[no-untyped-def]
        return iter(self.peaks)

    def components(self) -> list[Component]:
        """All peaks as model components, sorted by frequency."""
        return sorted((p.component() for p in self.peaks), key=lambda c: c.frequency)

    def poles(self) -> ComplexArray:
        """Signal poles of the peaks (damping floored at a tiny positive value)."""
        return np.array(
            [np.exp(complex(-max(p.damping, 1e-12), p.frequency)) for p in self.peaks],
            dtype=np.complex128,
        )


def _derivatives(x: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Centred first and second differences (``getSecDer.m``)."""
    dr = np.append(np.diff(x), 0.0)
    dl = np.insert(dr[:-1], 0, 0.0)
    d1 = 0.5 * (dr + dl)
    dr2 = np.append(np.diff(d1), 0.0)
    dl2 = np.insert(dr2[:-1], 0, 0.0)
    return d1, 0.5 * (dl2 + dr2)


def lorentzian_from_bins(
    spectrum: ComplexArray, center: int, *, unwrapped_phase: FloatArray | None = None
) -> tuple[float, float, float, float]:
    """Fit a Lorentzian through bins ``center-1, center, center+1``.

    Returns ``(frequency, damping, amplitude, phase)`` with the frequency in
    ``[0, 2π)`` on the DFT grid ``ω_k = 2πk/N``. ``1/|Y|²`` of a Lorentzian is a
    parabola in ω, so the three points determine it exactly (this is the
    quadratic solved in ``getApproxPeakParams.m``).
    """
    n = spectrum.size
    k = np.array([center - 1, center, center + 1])
    f = 2.0 * math.pi * k / n
    m = np.abs(spectrum[k % n]) ** 2
    inv = 1.0 / np.maximum(m, np.finfo(float).tiny)
    c2, c1, c0 = np.polyfit(f - f[1], inv, 2)
    step = 2.0 * math.pi / n
    if c2 > 0:
        w_rel = -c1 / (2.0 * c2)
        w_rel = float(np.clip(w_rel, -step, step))
        amp = 1.0 / math.sqrt(c2)
        d2 = c0 / c2 - w_rel**2
        damping = math.sqrt(abs(d2))
    else:  # not peak-shaped: fall back to the bin itself
        w_rel, damping = 0.0, step
        amp = math.sqrt(m[1]) * damping
    w = f[1] + w_rel
    phase_u = np.unwrap(np.angle(spectrum)) if unwrapped_phase is None else unwrapped_phase
    j = int(np.floor(w / step)) % n
    j1 = (j + 1) % n
    frac = (w - j * step) / step
    ph = float(phase_u[j] + frac * (phase_u[j1] - phase_u[j]))
    return w % (2.0 * math.pi), damping, amp, (ph + math.pi) % (2.0 * math.pi) - math.pi


def detect_spectrum_peaks(
    spectrum: ArrayLike,
    *,
    threshold: float | None = None,
    region: tuple[int, int] | None = None,
    max_peaks: int = 0,
    smoothing: int = 10,
) -> PeakList:
    """Detect peaks in a complex spectrum (``detectPeaks.m``).

    Parameters
    ----------
    spectrum
        Complex spectrum (DFT bins in the order to be scanned).
    threshold
        Minimum height of the maximum and minimum rise of the peak above the
        lower edge of its neighbourhood. Defaults to the mean magnitude.
    region
        ``(first, last)`` bins (inclusive) to search; default all.
    max_peaks
        Keep only the ``max_peaks`` highest peaks (0 keeps all).
    smoothing
        Half-length of the triangular smoothing window.

    Frequencies of the returned peaks are ``2π·bin/N`` wrapped to ``[-π, π)``.
    """
    ffy = np.asarray(spectrum, dtype=np.complex128).ravel()
    n = ffy.size
    mag = np.abs(ffy)
    amp_test = float(mag.mean()) if threshold is None else threshold
    lo, hi = (0, n - 1) if region is None else region

    h = smoothing
    tri = np.concatenate([np.arange(1, h + 1), np.arange(h - 1, 0, -1)]) / h
    smooth = np.convolve(mag, tri, mode="same") / h
    d1, d2 = _derivatives(smooth)
    d2 = -d2

    raw_d = np.diff(mag)
    raw_max = np.flatnonzero((raw_d[:-1] > 0) & (raw_d[1:] <= 0)) + 1  # local maxima of |Y|
    smooth_max = np.flatnonzero((d1[:-1] > 0) & (d1[1:] <= 0))
    if raw_max.size == 0:
        return PeakList((), np.zeros(0, dtype=np.int64))

    found: dict[int, tuple[int, int]] = {}
    for zc in smooth_max:
        j = int(raw_max[np.argmin(np.abs(raw_max - 1 - zc))])
        if not lo <= j <= hi or j in found or mag[j] <= amp_test or not 0 < j < n - 1:
            continue
        left = d2[: j + 1]
        starts = np.flatnonzero((left[:-1] < 0) & (left[1:] > 0))
        start = int(starts[-1]) if starts.size else 0
        right = d2[j:]
        fins = np.flatnonzero((right[:-1] > 0) & (right[1:] < 0))
        fin = int(fins[0]) + 2 if fins.size else right.size
        half = max(1, min(j - start, fin))
        a, b = max(j - half, 0), min(j + half, n - 1)
        rise = max(mag[j] - mag[a], mag[j] - mag[b])
        if rise > amp_test:
            found[j] = (a, b)

    centers = sorted(found)
    if max_peaks > 0 and len(centers) > max_peaks:
        centers = sorted(sorted(centers, key=lambda c: -mag[c])[:max_peaks])
    phase_u = np.unwrap(np.angle(ffy))
    peaks = []
    for c in centers:
        w, damping, amp, ph = lorentzian_from_bins(ffy, c, unwrapped_phase=phase_u)
        a, b = found[c]
        peaks.append(Peak(c, a, b, (w + math.pi) % (2 * math.pi) - math.pi, damping, amp, ph))
    taken = (
        np.unique(np.concatenate([np.arange(p.start, p.stop + 1) for p in peaks]))
        if peaks
        else np.zeros(0, dtype=np.int64)
    )
    return PeakList(tuple(peaks), taken.astype(np.int64))


def detect_peaks(
    y: ArrayLike,
    *,
    threshold: float | None = None,
    max_peaks: int = 0,
    smoothing: int = 10,
) -> PeakList:
    """Detect peaks in the spectrum of a time-domain FID.

    The spectrum is scanned in ``fftshift`` order, so peaks near ±π are found
    as well; ``bin``/``start``/``stop`` refer to that shifted order, while
    frequencies are in the natural ``[-π, π)`` convention of the model.
    """
    yc = np.asarray(y, dtype=np.complex128).ravel()
    n = yc.size
    shifted = np.fft.fftshift(np.fft.fft(yc))
    found = detect_spectrum_peaks(
        shifted, threshold=threshold, max_peaks=max_peaks, smoothing=smoothing
    )
    # shifted bin j has frequency 2π(j - ⌊N/2⌋)/N
    shift = 2.0 * math.pi * (n // 2) / n
    peaks = tuple(
        Peak(
            p.bin,
            p.start,
            p.stop,
            (p.frequency - shift + math.pi) % (2 * math.pi) - math.pi,
            p.damping,
            p.amplitude,
            p.phase,
        )
        for p in found.peaks
    )
    return PeakList(peaks, found.taken)
