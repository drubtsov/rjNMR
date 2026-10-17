r"""Signal model: a sum of exponentially damped complex sinusoids.

The free induction decay (FID) is modelled as (Rubtsov & Griffin 2007, eq. 1)

.. math::

    y_t = \sum_{i=1}^{k} a_i z_i^t + e_t, \qquad t = 0, \dots, N-1,

with signal poles :math:`z_i = e^{-\alpha_i + j\omega_i}` inside the unit disk
(:math:`\alpha_i \ge 0` is the damping factor, :math:`\omega_i` the angular
frequency in radians per sample), complex amplitudes
:math:`a_i = A_i e^{j\phi_i}` and circular white Gaussian noise :math:`e_t`
with variance :math:`\sigma^2` in each of the real and imaginary parts.

In matrix form :math:`y = D a + e` where :math:`D_{ti} = z_i^t`. Because the
columns of :math:`D` are geometric sequences, the Gram matrix
:math:`D^H D` has a closed form. This module never needs to build the full
:math:`N \times k` matrix during sampling: only :math:`D^H D` (``k × k``) and
:math:`D^H y` (``k``) are needed, and each costs O(N) per component.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "Component",
    "basis",
    "components_from_poles",
    "gram",
    "gram_column",
    "poles_from_components",
    "project",
    "synthesize",
    "wrap_angle",
]

ComplexArray = NDArray[np.complex128]
FloatArray = NDArray[np.float64]

_SMALL = 1e-12


@dataclass(frozen=True, slots=True)
class Component:
    r"""One damped sinusoid.

    Parameters
    ----------
    frequency
        Angular frequency :math:`\omega` in radians per sample, in ``[-π, π)``.
    damping
        Damping factor :math:`\alpha \ge 0` in nepers per sample.
    amplitude
        Magnitude :math:`A \ge 0` of the complex amplitude.
    phase
        Phase :math:`\phi` of the complex amplitude, in radians.
    """

    frequency: float
    damping: float
    amplitude: float = 1.0
    phase: float = 0.0

    @property
    def pole(self) -> complex:
        r"""Signal pole :math:`z = e^{-\alpha + j\omega}`."""
        return complex(np.exp(complex(-self.damping, self.frequency)))

    @property
    def complex_amplitude(self) -> complex:
        r"""Complex amplitude :math:`a = A e^{j\phi}`."""
        return complex(self.amplitude * np.exp(1j * self.phase))

    def frequency_hz(self, spectral_width: float) -> float:
        """Frequency in Hz relative to the carrier, given the spectral width in Hz."""
        return self.frequency / (2.0 * np.pi) * spectral_width

    def linewidth_hz(self, spectral_width: float) -> float:
        """Full width at half maximum of the Lorentzian line, in Hz."""
        return self.damping * spectral_width / np.pi

    def snr_db(self, noise_variance: float) -> float:
        """Component SNR as defined in the paper: ``10 log10(A² / 2σ²)``."""
        return float(10.0 * np.log10(self.amplitude**2 / (2.0 * noise_variance)))


def _as_complex(x: ArrayLike) -> ComplexArray:
    return np.ascontiguousarray(np.asarray(x, dtype=np.complex128).ravel())


def _log_poles(z: ComplexArray) -> ComplexArray:
    # Principal logarithm: s = -alpha + j*omega with omega in (-pi, pi].
    with np.errstate(divide="ignore"):
        return np.log(z)


def basis(z: ArrayLike, n: int) -> ComplexArray:
    """Return the ``n × k`` design matrix ``D`` with ``D[t, i] = z_i**t``."""
    s = _log_poles(_as_complex(z))
    t = np.arange(n, dtype=np.float64)
    out: ComplexArray = np.exp(np.outer(t, s))
    return out


def _geometric_sum(u: ComplexArray, n: int) -> ComplexArray:
    """Return ``sum_{t=0}^{n-1} exp(u t)`` element-wise, stable near ``u = 0``."""
    small = np.abs(u) < _SMALL
    u_safe = np.where(small, 1j, u)  # placeholder keeps the unused branch finite
    out: ComplexArray = np.where(
        small,
        n * (1.0 + 0.5 * (n - 1) * u),
        np.expm1(n * u_safe) / np.expm1(u_safe),
    )
    return out


def gram(z: ArrayLike, n: int) -> ComplexArray:
    r"""Closed-form Gram matrix :math:`G = D^H D`.

    :math:`G_{il} = \sum_t (\bar z_i z_l)^t = (1 - (\bar z_i z_l)^N) / (1 - \bar z_i z_l)`,
    evaluated through ``expm1`` in log-pole space for numerical stability.
    """
    s = _log_poles(_as_complex(z))
    u = np.conj(s)[:, None] + s[None, :]
    g = _geometric_sum(u, n)
    # enforce exact Hermitian symmetry (and real diagonal)
    g = 0.5 * (g + g.conj().T)
    return g


def gram_column(z: ArrayLike, z_new: complex, n: int) -> ComplexArray:
    r"""Column :math:`D^H d_{new}` of the Gram matrix for a new pole ``z_new``."""
    s = _log_poles(_as_complex(z))
    s_new = complex(np.log(z_new))
    return _geometric_sum(np.conj(s) + s_new, n)


def project(z: ArrayLike, y: ArrayLike) -> ComplexArray:
    r"""Return :math:`D^H y`, i.e. :math:`\sum_t \bar z_i^t y_t` for every pole."""
    yc = _as_complex(y)
    zc = _as_complex(z)
    if zc.size == 0:
        return np.zeros(0, dtype=np.complex128)
    d = basis(zc, yc.size)
    out: ComplexArray = d.conj().T @ yc
    return out


def synthesize(
    z: ArrayLike,
    a: ArrayLike,
    n: int,
) -> ComplexArray:
    """Noise-free time-domain signal ``D a`` for poles ``z`` and complex amplitudes ``a``."""
    zc, ac = _as_complex(z), _as_complex(a)
    if zc.shape != ac.shape:
        raise ValueError(f"z and a must have the same length, got {zc.size} and {ac.size}")
    if zc.size == 0:
        return np.zeros(n, dtype=np.complex128)
    out: ComplexArray = basis(zc, n) @ ac
    return out


def poles_from_components(components: Iterable[Component]) -> tuple[ComplexArray, ComplexArray]:
    """Return ``(z, a)`` arrays for a sequence of :class:`Component`."""
    comps = list(components)
    z = np.array([c.pole for c in comps], dtype=np.complex128)
    a = np.array([c.complex_amplitude for c in comps], dtype=np.complex128)
    return z, a


def components_from_poles(z: ArrayLike, a: ArrayLike | None = None) -> list[Component]:
    """Convert pole/amplitude arrays into a list of :class:`Component`, sorted by frequency."""
    zc = _as_complex(z)
    ac = np.ones_like(zc) if a is None else _as_complex(a)
    s = _log_poles(zc)
    freq = np.angle(zc)
    out = [
        Component(
            frequency=float(f),
            damping=float(-si.real),
            amplitude=float(abs(ai)),
            phase=float(np.angle(ai)),
        )
        for f, si, ai in zip(freq, s, ac, strict=True)
    ]
    return sorted(out, key=lambda c: c.frequency)


def wrap_angle(x: FloatArray | float) -> FloatArray | float:
    """Wrap an angle into ``[-π, π)``."""
    return (np.asarray(x) + np.pi) % (2.0 * np.pi) - np.pi
