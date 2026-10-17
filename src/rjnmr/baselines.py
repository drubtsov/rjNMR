r"""Non-Bayesian reference methods used for comparison in the paper.

* :func:`matrix_pencil` – the matrix pencil method of Hua & Sarkar (1990/1991)
  for estimating the poles of a sum of damped exponentials.
* :func:`estimate_order` – AIC and MDL model-order criteria (Wax & Kailath 1985)
  applied to the singular values of the data Hankel matrix, as in the
  combined detection–estimation scheme of Lin et al. (1997) that the paper
  benchmarks against.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .model import Component, basis, components_from_poles

__all__ = ["OrderEstimate", "estimate_order", "hankel_singular_values", "matrix_pencil"]

ComplexArray = NDArray[np.complex128]
FloatArray = NDArray[np.float64]


def _hankel(y: ComplexArray, pencil: int) -> ComplexArray:
    """``(N - L) × (L + 1)`` Hankel data matrix with ``Y[i, j] = y[i + j]``."""
    n = y.size
    rows = n - pencil
    idx = np.arange(rows)[:, None] + np.arange(pencil + 1)[None, :]
    out: ComplexArray = y[idx]
    return out


def _default_pencil(n: int) -> int:
    # L between N/3 and N/2 gives the best noise performance (Hua & Sarkar)
    return max(1, n // 3)


def hankel_singular_values(y: ArrayLike, pencil: int | None = None) -> FloatArray:
    """Singular values of the Hankel data matrix used by the matrix pencil method."""
    yc = np.asarray(y, dtype=np.complex128).ravel()
    L = _default_pencil(yc.size) if pencil is None else pencil
    s: FloatArray = np.linalg.svd(_hankel(yc, L), compute_uv=False)
    return s


@dataclass(frozen=True, slots=True)
class OrderEstimate:
    """Model-order estimate and the criterion values it was chosen from."""

    order: int
    criterion: FloatArray  # criterion value for each candidate order 0..p-1
    method: str


def estimate_order(
    y: ArrayLike,
    method: Literal["mdl", "aic"] = "mdl",
    *,
    pencil: int | None = None,
    max_order: int | None = None,
) -> OrderEstimate:
    r"""Estimate the number of components with the AIC or MDL criterion.

    The eigenvalues :math:`\lambda_i = s_i^2` of the sample covariance of the
    Hankel matrix (``p = L + 1`` columns, ``n = N - L`` snapshots) are tested
    for equality of the ``p - m`` smallest ones:

    .. math::

        \Lambda(m) = -n (p - m) \log \frac{\left(\prod_{i>m} \lambda_i\right)^{1/(p-m)}}
                                          {\frac{1}{p-m} \sum_{i>m} \lambda_i},

    with :math:`\mathrm{AIC}(m) = 2\Lambda(m) + 2m(2p - m)` and
    :math:`\mathrm{MDL}(m) = \Lambda(m) + \tfrac12 m(2p - m) \log n`.
    """
    yc = np.asarray(y, dtype=np.complex128).ravel()
    L = _default_pencil(yc.size) if pencil is None else pencil
    lam = hankel_singular_values(yc, L) ** 2
    p = lam.size
    n_snap = yc.size - L
    mmax = p - 1 if max_order is None else min(max_order, p - 1)
    lam = np.maximum(lam, np.finfo(float).tiny)
    crit = np.empty(mmax + 1)
    for m in range(mmax + 1):
        tail = lam[m:]
        q = p - m
        log_geo = float(np.mean(np.log(tail)))
        log_ari = math.log(float(np.mean(tail)))
        loglik = -n_snap * q * (log_geo - log_ari)
        penalty_free = m * (2 * p - m)
        if method == "aic":
            crit[m] = 2.0 * loglik + 2.0 * penalty_free
        elif method == "mdl":
            crit[m] = loglik + 0.5 * penalty_free * math.log(n_snap)
        else:
            raise ValueError("method must be 'aic' or 'mdl'")
    return OrderEstimate(order=int(np.argmin(crit)), criterion=crit, method=method)


def matrix_pencil(
    y: ArrayLike,
    order: int | None = None,
    *,
    pencil: int | None = None,
    criterion: Literal["mdl", "aic"] = "mdl",
) -> list[Component]:
    """Estimate damped sinusoids with the SVD-based matrix pencil method.

    Parameters
    ----------
    y
        Complex FID.
    order
        Number of components. ``None`` estimates it with ``criterion``.
    pencil
        Pencil parameter ``L`` (default ``N // 3``).
    criterion
        ``"mdl"`` or ``"aic"``; used only when ``order`` is ``None``.

    Returns
    -------
    list of Component
        Components sorted by frequency, with least-squares amplitudes.
        Poles that fall outside the unit disk (growing exponentials) are
        reflected inside it.
    """
    yc = np.asarray(y, dtype=np.complex128).ravel()
    n = yc.size
    L = _default_pencil(n) if pencil is None else pencil
    if not 1 <= L < n - 1:
        raise ValueError("pencil parameter must satisfy 1 <= L < N - 1")
    if order is None:
        order = estimate_order(yc, criterion, pencil=L).order
    if order == 0:
        return []
    Y = _hankel(yc, L)
    _, _, vh = np.linalg.svd(Y, full_matrices=False)
    order = min(order, vh.shape[0], L)
    # Rows of Y are combinations of [1, z, ..., z^L]; so are the leading rows of Vᴴ.
    V = vh[:order].T
    V1, V2 = V[:-1], V[1:]
    z = np.linalg.eigvals(np.linalg.pinv(V1) @ V2)
    r = np.abs(z)
    z = np.where(r >= 1.0, z / (r**2) * (1 - 1e-9), z)  # reflect growing poles inside
    D = basis(z, n)
    a, *_ = np.linalg.lstsq(D, yc, rcond=None)
    return components_from_poles(z, a)
