"""Post-processing of fitted components (2010 toolbox).

* :func:`merge_close_components` ports ``compCleanup.m``, ``mergeComps.m`` and
  ``recurrClust.m``. A non-Lorentzian line is often modelled by several
  Lorentzians that are close in frequency and damping; such clusters are merged
  into one component, which helps interpretation and quantitation.
* :func:`refit_amplitudes` re-estimates all amplitudes jointly on the full
  signal, either by ordinary least squares or by the peak-weighted
  frequency-domain regression of ``getPeakWeightAmps.m``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from .model import Component, basis, components_from_poles

__all__ = ["merge_close_components", "refit_amplitudes"]

ComplexArray = NDArray[np.complex128]


def merge_close_components(
    components: Sequence[Component],
    *,
    frequency_tol: float = 0.0009,
    damping_tol: float = 0.004,
) -> list[Component]:
    """Merge clusters of components closer than both tolerances.

    Two components are linked when their frequencies differ by less than
    ``frequency_tol`` (rad/sample) *and* their dampings by less than
    ``damping_tol``. Linked groups (connected components of that graph) are
    replaced by one component. Its pole is the ``|a|²``-weighted mean of the
    poles, pulled back inside the unit disk if needed, and its complex
    amplitude is the sum of the amplitudes. The defaults are the Matlab
    thresholds.
    """
    comps = list(components)
    if len(comps) < 2:
        return comps
    w = np.array([c.frequency for c in comps])
    dr = np.array([c.damping for c in comps])
    dw = np.abs((w[:, None] - w[None, :] + math.pi) % (2 * math.pi) - math.pi)
    linked = (dw < frequency_tol) & (np.abs(dr[:, None] - dr[None, :]) < damping_tol)
    n_groups, labels = connected_components(csr_matrix(linked), directed=False)
    z = np.array([c.pole for c in comps])
    a = np.array([c.complex_amplitude for c in comps])
    zs, as_ = [], []
    for g in range(n_groups):
        idx = np.flatnonzero(labels == g)
        weights = np.abs(a[idx]) ** 2
        zg = complex(np.sum(weights * z[idx]) / weights.sum()) if weights.sum() > 0 else z[idx[0]]
        if abs(zg) >= 1.0:
            zg = zg / abs(zg) * (1.0 - 1e-12)
        zs.append(zg)
        as_.append(complex(a[idx].sum()))
    return components_from_poles(np.array(zs), np.array(as_))


def refit_amplitudes(
    y: ArrayLike,
    components: Sequence[Component],
    *,
    method: Literal["lstsq", "peak_weighted"] = "lstsq",
) -> list[Component]:
    """Re-estimate amplitudes of fixed poles on the whole FID.

    ``"lstsq"``
        Complex least squares in the time domain.
    ``"peak_weighted"``
        The 2010 toolbox's ``getPeakWeightAmps``: weighted least squares on
        the real (absorption) part of the spectrum, with each frequency bin
        weighted by the summed absorption of all model components there, so
        that bins under peaks dominate. It returns real, non-negative
        amplitudes (zero phase) and is meant for phase-corrected data.
    """
    yc = np.asarray(y, dtype=np.complex128).ravel()
    comps = list(components)
    if not comps:
        return []
    z = np.array([c.pole for c in comps])
    D = basis(z, yc.size)
    if method == "lstsq":
        a, *_ = np.linalg.lstsq(D, yc, rcond=None)
        return components_from_poles(z, a)
    if method == "peak_weighted":
        F = np.fft.fft(D, axis=0).real
        weights = np.clip(F.sum(axis=1), 0.0, None)
        if weights.sum() <= 0:
            weights = np.ones_like(weights)
        sw = np.sqrt(weights / weights.sum())
        a_real, *_ = np.linalg.lstsq(F * sw[:, None], np.fft.fft(yc).real * sw, rcond=None)
        return components_from_poles(z, np.abs(a_real).astype(np.complex128))
    raise ValueError("method must be 'lstsq' or 'peak_weighted'")
