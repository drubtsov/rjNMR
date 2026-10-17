r"""Likelihood "designs": how poles map onto the data that the sampler fits.

The sampler only ever needs the Gram matrix :math:`G = \tilde D^H \tilde D`,
the projection :math:`b = \tilde D^H \tilde y`, the data energy
:math:`\tilde y^H \tilde y` and the number of complex observations, where
:math:`\tilde D` and :math:`\tilde y` are the design matrix and data in
whatever domain the likelihood is written. Two designs are provided:

:class:`TimeDesign`
    The paper's time-domain likelihood, with the closed-form Gram matrix of
    geometric sequences.

:class:`WindowedSpectralDesign`
    The 2010 toolbox's band-limited likelihood. The data are the DFT bins of
    the FID inside the support of a frequency-domain window (typically a
    raised-cosine band, :func:`rjnmr.intervals.raised_cosine_window`), each
    multiplied by the window, and so are the model columns. The DFT of a
    geometric sequence has the closed form
    :math:`\sum_{t<N} z^t e^{-2\pi i k t/N} = (1 - z^N)/(1 - z e^{-2\pi i k/N})`,
    so the columns are exact and the poles and amplitudes stay in the global
    time-domain parametrisation. No decimation or rescaling is needed.

Both support a *baseline* nuisance component, the "flat" component
``w = 0, dr = -1e3`` of the Matlab code: a spike at ``t = 0``, i.e. a constant
offset of the spectrum. Its amplitude gets a flat prior and is integrated out
exactly, which amounts to projecting it out of the data and of every column
and removing one complex observation.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .model import _geometric_sum, gram, gram_column, project
from .proposals import SpectralDensity

__all__ = ["Design", "TimeDesign", "WindowedSpectralDesign", "make_design"]

ComplexArray = NDArray[np.complex128]
FloatArray = NDArray[np.float64]


class Design(ABC):
    """Interface used by :class:`rjnmr.sampler.RJMCMCSampler`."""

    #: length of the time-domain FID
    n_time: int
    #: number of complex observations entering the likelihood exponent
    n_obs: int
    #: energy of the (baseline-projected) data
    yy: float

    @abstractmethod
    def gram(self, z: ComplexArray) -> ComplexArray:
        """Gram matrix of the columns for poles ``z``."""

    @abstractmethod
    def gram_column(self, z: ComplexArray, z_new: complex) -> ComplexArray:
        """Inner products of the columns for ``z`` with the column of ``z_new``."""

    @abstractmethod
    def project(self, z: ComplexArray) -> ComplexArray:
        """Inner products of the columns for ``z`` with the data."""

    def project_one(self, z_new: complex) -> complex:
        """Inner product of a single column with the data."""
        return complex(self.project(np.array([z_new], dtype=np.complex128))[0])

    @abstractmethod
    def proposal_spectrum(
        self, z: ComplexArray, coef: ComplexArray, power: float, oversample: int
    ) -> SpectralDensity:
        """Spectral density of the residual ``data - columns(z) @ coef``."""

    @property
    def frequency_support(self) -> tuple[float, float] | None:
        """``(start, width)`` of the frequency band the data constrain, if limited."""
        return None


class TimeDesign(Design):
    """Time-domain likelihood of the 2007 paper: ``y_t = Σ a_i z_i^t + e_t``."""

    def __init__(self, y: ArrayLike, *, baseline: bool = False) -> None:
        self.y = np.ascontiguousarray(np.asarray(y, dtype=np.complex128).ravel())
        self.n_time = int(self.y.size)
        self.baseline = baseline
        self._t = np.arange(self.n_time, dtype=np.float64)
        y0 = self.y[0] if baseline else 0.0
        self._y0 = complex(y0)
        self.yy = float(np.vdot(self.y, self.y).real - abs(self._y0) ** 2)
        self.n_obs = self.n_time - int(baseline)

    # The baseline column is e_0 (a spike at t = 0); every pole column has the value 1 at
    # t = 0, so projecting e_0 out subtracts 1 from every Gram entry and y_0 from b.

    def gram(self, z: ComplexArray) -> ComplexArray:
        """Gram matrix of the columns for poles ``z``."""
        g = gram(z, self.n_time)
        return g - 1.0 if self.baseline else g

    def gram_column(self, z: ComplexArray, z_new: complex) -> ComplexArray:
        """Inner products of the columns for ``z`` with the column of ``z_new``."""
        col = gram_column(z, z_new, self.n_time)
        return col - 1.0 if self.baseline else col

    def project(self, z: ComplexArray) -> ComplexArray:
        """Inner products of the columns for ``z`` with the data."""
        b = project(z, self.y)
        return b - self._y0 if self.baseline else b

    def project_one(self, z_new: complex) -> complex:
        """Inner product of a single column with the data."""
        s = complex(np.log(z_new))
        return complex(np.dot(np.exp(np.conj(s) * self._t), self.y)) - self._y0

    def proposal_spectrum(
        self, z: ComplexArray, coef: ComplexArray, power: float, oversample: int
    ) -> SpectralDensity:
        """Spectral density of the residual ``data - columns(z) @ coef``."""
        r = self.y
        if z.size:
            r = self.y - np.exp(np.outer(self._t, np.log(z))) @ coef
        if self.baseline:
            r = r.copy()
            r[0] = 0.0
        return SpectralDensity.from_signal(r, power, oversample)


class WindowedSpectralDesign(Design):
    """Band-limited frequency-domain likelihood with a window (2010 toolbox).

    Parameters
    ----------
    y
        Full complex FID of length ``N``.
    window
        Non-negative weights for each of the ``N`` DFT bins in natural
        (``numpy.fft``) order. The likelihood uses the bins where the window
        is positive; this support must be one contiguous run of bins, possibly
        wrapping around the end of the array.
    baseline
        Project out a constant spectral offset.
    """

    def __init__(self, y: ArrayLike, window: ArrayLike, *, baseline: bool = False) -> None:
        yt = np.asarray(y, dtype=np.complex128).ravel()
        w = np.asarray(window, dtype=np.float64).ravel()
        n = yt.size
        if w.size != n:
            raise ValueError(f"window has {w.size} bins but the FID has {n} points")
        if np.any(w < 0) or not np.any(w > 0):
            raise ValueError("window must be non-negative with a non-empty support")
        self.n_time = n
        self.window = w
        self.bins = _contiguous_support(w > 0)
        self.offset = int(self.bins[0])
        self.baseline = baseline
        m = self.bins.size
        self._w = w[self.bins]
        scale = 1.0 / math.sqrt(n)
        self._d = self._w * np.fft.fft(yt)[self.bins] * scale
        # u_k = -2πi k/N for the bins in the support
        self._shift = -2j * np.pi * self.bins.astype(np.float64) / n
        self._scale = scale
        if baseline:
            v = self._w * scale
            self._v = (v / np.linalg.norm(v)).astype(np.complex128)
            self._vd = complex(np.vdot(self._v, self._d))
        else:
            self._v = np.zeros(m, dtype=np.complex128)
            self._vd = 0j
        self._cache: dict[complex, tuple[ComplexArray, complex, complex]] = {}
        self._cache_size = 4096
        self.yy = float(np.vdot(self._d, self._d).real - abs(self._vd) ** 2)
        self.n_obs = m - int(baseline)

    @property
    def frequency_support(self) -> tuple[float, float]:
        """``(start, width)`` of the band covered by the window."""
        start = 2.0 * math.pi * (self.offset - 0.5) / self.n_time
        width = 2.0 * math.pi * self.bins.size / self.n_time
        return (start + math.pi) % (2.0 * math.pi) - math.pi, width

    def _column(self, z: complex) -> tuple[ComplexArray, complex, complex]:
        """Column of one pole plus its baseline overlap ``vᴴc`` and projection ``cᴴd``.

        Results are memoised per pole value: the sampler evaluates the same
        poles many times, and the transcendental functions dominate the cost.
        """
        hit = self._cache.get(z)
        if hit is not None:
            return hit
        u = np.asarray(complex(np.log(z)) + self._shift, dtype=np.complex128)
        c: ComplexArray = self._w * _geometric_sum(u, self.n_time) * self._scale
        vc = complex(np.vdot(self._v, c))
        cd = complex(np.vdot(c, self._d)) - vc.conjugate() * self._vd
        if len(self._cache) >= self._cache_size:
            self._cache.clear()
        self._cache[z] = (c, vc, cd)
        return c, vc, cd

    def columns(self, z: ArrayLike) -> ComplexArray:
        """Windowed, normalised DFT columns ``W_k Σ_t z^t e^{-2πikt/N} / √N`` (``M × k``)."""
        zs = np.asarray(z, dtype=np.complex128).ravel()
        if zs.size == 0:
            return np.zeros((self.bins.size, 0), dtype=np.complex128)
        return np.stack([self._column(complex(zi))[0] for zi in zs], axis=1)

    def _overlaps(self, z: ComplexArray) -> ComplexArray:
        return np.array([self._column(complex(zi))[1] for zi in z], dtype=np.complex128)

    def gram(self, z: ComplexArray) -> ComplexArray:
        """Gram matrix of the columns for poles ``z``."""
        c = self.columns(z)
        g: ComplexArray = c.conj().T @ c
        if self.baseline:
            vc = self._overlaps(z)
            g = g - np.outer(vc.conj(), vc)
        g = 0.5 * (g + g.conj().T)
        return g

    def gram_column(self, z: ComplexArray, z_new: complex) -> ComplexArray:
        """Inner products of the columns for ``z`` with the column of ``z_new``."""
        cn, vcn, _ = self._column(complex(z_new))
        cached = [self._column(complex(zi)) for zi in z]
        col = np.array([np.vdot(c, cn) for c, _, _ in cached], dtype=np.complex128)
        if self.baseline:
            col -= np.array([vc for _, vc, _ in cached]).conj() * vcn
        return col

    def project(self, z: ComplexArray) -> ComplexArray:
        """Inner products of the columns for ``z`` with the data."""
        return np.array([self._column(complex(zi))[2] for zi in z], dtype=np.complex128)

    def project_one(self, z_new: complex) -> complex:
        """Inner product of a single column with the data."""
        return self._column(complex(z_new))[2]

    def proposal_spectrum(
        self, z: ComplexArray, coef: ComplexArray, power: float, oversample: int
    ) -> SpectralDensity:
        """Spectral density of the residual ``data - columns(z) @ coef``."""
        r = self._d
        if z.size:
            r = self._d - self.columns(z) @ coef
        if self.baseline:
            r = r - self._v * complex(np.vdot(self._v, r))
        return SpectralDensity.from_bins(np.abs(r) ** power, self.bins, self.n_time)


def _contiguous_support(mask: NDArray[np.bool_]) -> NDArray[np.int64]:
    """Indices of the single (possibly wrapping) run of ``True`` in ``mask``."""
    n = mask.size
    idx = np.flatnonzero(mask)
    if idx.size == n:
        return np.arange(n, dtype=np.int64)
    # rotate so that the run starts after a False entry
    first_false = int(np.flatnonzero(~mask)[0])
    rolled = np.roll(mask, -first_false)
    ridx = np.flatnonzero(rolled)
    if ridx[-1] - ridx[0] + 1 != ridx.size:
        raise ValueError("window support must be a single contiguous band of bins")
    out: NDArray[np.int64] = ((ridx + first_false) % n).astype(np.int64)
    return out


def make_design(
    y: ArrayLike, *, window: ArrayLike | None = None, baseline: bool = False
) -> Design:
    """Time-domain design, or windowed spectral design when ``window`` is given."""
    if window is None:
        return TimeDesign(y, baseline=baseline)
    return WindowedSpectralDesign(y, window, baseline=baseline)
