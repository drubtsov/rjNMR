"""Starting values for the sampler.

The paper starts every chain from a single random component; the original
Matlab code offered an FFT peak-picking start (``ISCLEVER_INI``). A good start
does not change the target distribution, it only shortens burn-in.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.signal import find_peaks

from .baselines import matrix_pencil

__all__ = ["InitMethod", "initial_poles", "peak_poles"]

InitMethod = Literal["empty", "random", "peaks", "matrix_pencil"]
ComplexArray = NDArray[np.complex128]


def peak_poles(
    y: ArrayLike,
    *,
    threshold: float = 0.003,
    damping: float | None = None,
    max_peaks: int | None = None,
) -> ComplexArray:
    """Poles at the peaks of the magnitude spectrum.

    Parameters
    ----------
    y
        Complex FID.
    threshold
        Minimum peak prominence as a fraction of the total spectral magnitude
        (the Matlab ``peakScan`` default was 0.003).
    damping
        Damping assigned to every pole; defaults to ``2/N``.
    max_peaks
        Keep only the most prominent peaks.
    """
    yc = np.asarray(y, dtype=np.complex128).ravel()
    n = yc.size
    mag = np.abs(np.fft.fft(yc))
    mag = mag / max(float(mag.sum()), np.finfo(float).tiny)
    # wrap-around padding so that peaks near ±π are found
    pad = 8
    ext = np.concatenate([mag[-pad:], mag, mag[:pad]])
    idx, props = find_peaks(ext, prominence=threshold)
    idx = idx - pad
    keep = (idx >= 0) & (idx < n)
    idx, prom = idx[keep], props["prominences"][keep]
    if max_peaks is not None and idx.size > max_peaks:
        top = np.argsort(prom)[::-1][:max_peaks]
        idx = idx[top]
    omega = 2.0 * np.pi * np.fft.fftfreq(n)[idx]
    alpha = 2.0 / n if damping is None else damping
    out: ComplexArray = np.exp(-alpha + 1j * np.sort(omega))
    return out


def initial_poles(
    y: ArrayLike,
    method: InitMethod = "empty",
    *,
    seed: int | np.random.Generator | None = None,
    max_components: int | None = None,
) -> ComplexArray:
    """Initial poles for :meth:`rjnmr.sampler.RJMCMCSampler.run`.

    ``"empty"`` – no components; ``"random"`` – one pole drawn uniformly on the
    disk (the paper's choice); ``"peaks"`` – FFT peak picking;
    ``"matrix_pencil"`` – matrix pencil estimate with MDL order selection.
    """
    yc = np.asarray(y, dtype=np.complex128).ravel()
    if method == "empty":
        z = np.zeros(0, dtype=np.complex128)
    elif method == "random":
        rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
        r = np.sqrt(rng.random())
        z = np.array([r * np.exp(1j * rng.uniform(-np.pi, np.pi))])
    elif method == "peaks":
        z = peak_poles(yc, max_peaks=max_components)
    elif method == "matrix_pencil":
        comps = matrix_pencil(yc)
        z = np.array([c.pole for c in comps], dtype=np.complex128)
        z = z[(np.abs(z) > 0) & (np.abs(z) < 1)]
        if max_components is not None:
            z = z[:max_components]
    else:
        raise ValueError(f"unknown init method {method!r}")
    return z
