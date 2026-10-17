r"""Automatic zero- and first-order phase correction (2010 toolbox).

Port of ``autoPhasing.m`` and ``phCostFun.m``. The spectrum is handled in
``fftshift`` order so that bin index is proportional to frequency, and the
correction is

.. math:: Y_k \leftarrow Y_k\, e^{-i(\phi_0 + \phi_1 (k - k_0))}.

The procedure is:

1. **First-order term.** Unwrap the spectral phase, split it into 64 blocks,
   fit a line to each block, and take the median slope as the starting
   :math:`\phi_1`. A linear phase is a time shift, so this mostly removes the
   digital-filter group delay of Bruker data.
2. **Zero-order term.** For each detected peak (:func:`rjnmr.peaks.detect_spectrum_peaks`),
   try :math:`\phi_0` equal to that peak's Lorentzian phase with pivot
   :math:`k_0` at the peak. Keep the candidate with the least negative
   absorption inside the peak neighbourhoods.
3. **Refinement.** Minimise the squared sum of negative absorption inside the
   peak neighbourhoods,
   :math:`\left(\sum_{k} \min(\operatorname{Re} Y_k, 0)\right)^2`,
   with Nelder–Mead. The Matlab code optimises :math:`(\phi_0, \phi_1, k_0)`
   jointly. Here the redundant pivot :math:`k_0` is fixed, :math:`\phi_1` is
   expressed in points of delay, and the search starts from several delays.

As the Matlab documentation warns, this works when there are strong,
well-separated peaks to anchor it. Otherwise phase the data by other means.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import minimize

from .peaks import PeakList, detect_spectrum_peaks

__all__ = ["PhaseCorrection", "auto_phase"]

ComplexArray = NDArray[np.complex128]
FloatArray = NDArray[np.float64]

_N_BLOCKS = 64


@dataclass(frozen=True, slots=True)
class PhaseCorrection:
    """Linear phase correction ``φ(k) = phase0 + slope · (k - pivot)``.

    ``k`` indexes the ``fftshift``-ordered spectrum of an ``n``-point FID.
    """

    phase0: float
    slope: float
    pivot: float
    n: int

    def phase(self) -> FloatArray:
        """Correction phase for every bin of the shifted spectrum."""
        k = np.arange(self.n, dtype=np.float64)
        out: FloatArray = self.phase0 + self.slope * (k - self.pivot)
        return out

    def apply_spectrum(self, shifted_spectrum: ArrayLike) -> ComplexArray:
        """Phase a spectrum given in ``fftshift`` order."""
        s = np.asarray(shifted_spectrum, dtype=np.complex128)
        out: ComplexArray = s * np.exp(-1j * self.phase())
        return out

    def apply(self, y: ArrayLike) -> ComplexArray:
        """Phase a time-domain FID; returns the corrected FID."""
        yc = np.asarray(y, dtype=np.complex128).ravel()
        if yc.size != self.n:
            raise ValueError(f"correction is for {self.n} points, got {yc.size}")
        shifted = np.fft.fftshift(np.fft.fft(yc))
        out: ComplexArray = np.fft.ifft(np.fft.ifftshift(self.apply_spectrum(shifted)))
        return out

    @property
    def group_delay(self) -> float:
        """Time shift (in points) equivalent to the first-order term."""
        return self.slope * self.n / (2.0 * math.pi)


def _negative_absorption(spec: ComplexArray, taken: NDArray[np.int64]) -> float:
    re = spec.real[taken]
    return float(np.sum(re[re < 0]))


def auto_phase(
    y: ArrayLike,
    *,
    n_peaks: int = 0,
    region: tuple[int, int] | None = None,
    refine: bool = True,
    refine_offsets: tuple[float, ...] = (0.0, -1.0, 1.0, -2.0, 2.0, -4.0, 4.0),
) -> tuple[ComplexArray, PhaseCorrection, PeakList]:
    """Phase-correct an FID automatically.

    Parameters
    ----------
    y
        Complex FID.
    n_peaks
        Use only the ``n_peaks`` highest peaks (0 uses all detected peaks).
    region
        ``(first, last)`` bins of the ``fftshift``-ordered spectrum to use,
        e.g. to exclude the residual water signal.
    refine
        Refine the best candidate by Nelder–Mead (as the Matlab code does).
    refine_offsets
        Offsets, in points of delay, added to the estimated first-order term
        to give the starting points of the refinement.

    Returns
    -------
    phased, correction, peaks
        The corrected FID, the correction applied and the peaks used.
    """
    yc = np.asarray(y, dtype=np.complex128).ravel()
    n = yc.size
    spec = np.fft.fftshift(np.fft.fft(yc))
    lo, hi = (0, n - 1) if region is None else region
    peaks = detect_spectrum_peaks(spec, region=(lo, hi), max_peaks=n_peaks)
    if len(peaks) == 0:
        raise ValueError("no peaks detected; cannot phase automatically")

    # 1. first-order slope from blockwise regression of the unwrapped phase
    unwrapped = np.unwrap(np.angle(spec))[lo : hi + 1]
    usable = unwrapped[: unwrapped.size - unwrapped.size % _N_BLOCKS]
    slope = 0.0
    if usable.size >= 2 * _N_BLOCKS:
        blocks = usable.reshape(_N_BLOCKS, -1).T  # columns are contiguous blocks
        x = np.arange(blocks.shape[0], dtype=np.float64)
        slopes = np.polyfit(x, blocks, 1)[0]
        slope = float(np.median(slopes))
        if not math.isfinite(slope):
            slope = 0.0

    # 2. zero-order candidates anchored at each peak
    taken = peaks.taken
    best: PhaseCorrection | None = None
    best_cost = math.inf
    for p in peaks:
        cand = PhaseCorrection(p.phase, slope, float(p.bin), n)
        cand_cost = abs(_negative_absorption(cand.apply_spectrum(spec), taken))
        if cand_cost < best_cost:
            best, best_cost = cand, cand_cost
    assert best is not None

    # 3. refinement. The pivot is redundant with phase0, so it stays fixed at the anchor
    # peak; the slope is optimised in delay units (points), which makes the problem well
    # scaled. A few starting delays guard against a poor blockwise estimate.
    if refine:
        pivot = best.pivot
        to_delay = n / (2.0 * math.pi)

        def cost(x: FloatArray) -> float:
            c = PhaseCorrection(float(x[0]), float(x[1]) / to_delay, pivot, n)
            return _negative_absorption(c.apply_spectrum(spec), taken) ** 2

        start_delay = best.slope * to_delay
        best_x = np.array([best.phase0, start_delay])
        best_f = cost(best_x)
        for offset in refine_offsets:
            res = minimize(
                cost,
                np.array([best.phase0, start_delay + offset]),
                method="Nelder-Mead",
                options={"xatol": 1e-6, "fatol": 1e-14 * max(best_f, 1.0), "maxiter": 2000},
            )
            if res.fun < best_f:
                best_x, best_f = res.x, float(res.fun)
        best = PhaseCorrection(float(best_x[0]), float(best_x[1]) / to_delay, pivot, n)

    return best.apply(yc), best, peaks
