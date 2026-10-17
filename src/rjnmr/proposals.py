r"""Proposal distributions for newborn signal poles.

All proposals are densities on the complex unit disk, expressed with respect to
area (Lebesgue) measure so that they combine directly with the uniform prior
:math:`p(z) = 1/\pi` in the birth/death acceptance ratio.

For a pole parametrised as :math:`z = e^{-\alpha + j\omega}` the area element
is :math:`dA = e^{-2\alpha}\, d\alpha\, d\omega`, hence a density
:math:`q(\omega, \alpha)` in polar coordinates corresponds to
:math:`q(\omega, \alpha) e^{2\alpha}` on the disk.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

__all__ = ["PoleProposal", "SpectralDensity", "UniformDisk", "in_band"]


def in_band(omega: float, start: float, width: float) -> bool:
    """Whether angle ``omega`` lies in ``[start, start + width)`` modulo 2π."""
    return width >= 2.0 * math.pi or (omega - start) % (2.0 * math.pi) < width


@dataclass(frozen=True, slots=True)
class UniformDisk:
    """Uniform distribution on an annular sector of the unit disk.

    The support is ``r_min < |z| < 1`` and ``arg z ∈ [omega_start, omega_start + omega_width)``.
    The defaults give the whole disk, which is the paper's birth proposal (and pole prior).
    """

    r_min: float = 0.0
    omega_start: float = -math.pi
    omega_width: float = 2.0 * math.pi

    @property
    def log_area(self) -> float:
        """Log of the area of the support."""
        return math.log(0.5 * self.omega_width) + math.log1p(-(self.r_min**2))

    def contains(self, z: complex) -> bool:
        """Whether ``z`` lies in the support."""
        return self.r_min < abs(z) < 1.0 and in_band(
            math.atan2(z.imag, z.real), self.omega_start, self.omega_width
        )

    def sample(self, rng: np.random.Generator) -> complex:
        """Draw one pole."""
        r2 = self.r_min**2
        r = math.sqrt(r2 + (1.0 - r2) * rng.random())
        theta = self.omega_start + self.omega_width * rng.random()
        return complex(r * math.cos(theta), r * math.sin(theta))

    def log_density(self, z: complex) -> float:
        """Log density with respect to area measure."""
        return -self.log_area if self.contains(z) else -math.inf


@dataclass(frozen=True, slots=True)
class SpectralDensity:
    """Piecewise-constant density on frequency built from a spectrum.

    Bin ``j`` of an ``N``-point DFT is centred on ``ω_j = 2πj/N`` (wrapped to
    ``[-π, π)``) and has width ``2π/N``; its probability is proportional to
    ``|Y_j|**power``.
    """

    weights: NDArray[np.float64]  # normalised bin probabilities, length N
    cumulative: NDArray[np.float64]

    @classmethod
    def from_signal(
        cls, y: NDArray[np.complex128], power: float = 1.0, oversample: int = 1
    ) -> SpectralDensity:
        """Build from the DFT magnitude of a time-domain signal.

        ``oversample > 1`` zero-pads the signal to ``oversample * N`` points,
        giving proportionally narrower frequency bins.
        """
        mag = np.abs(np.fft.fft(y, n=oversample * y.size)) ** power
        return cls._normalised(mag)

    @classmethod
    def from_bins(
        cls, magnitude: NDArray[np.float64], bins: NDArray[np.int64], n_bins: int
    ) -> SpectralDensity:
        """Build from magnitudes given on a subset ``bins`` of an ``n_bins``-point DFT grid.

        Bins outside the subset get zero probability.
        """
        mag = np.zeros(n_bins)
        mag[bins] = magnitude
        if not np.isfinite(mag.sum()) or mag.sum() <= 0.0:
            mag[bins] = 1.0
        return cls._normalised(mag)

    @classmethod
    def _normalised(cls, mag: NDArray[np.float64]) -> SpectralDensity:
        total = float(mag.sum())
        if not np.isfinite(total) or total <= 0.0:
            mag = np.ones_like(mag)
            total = float(mag.size)
        w = mag / total
        return cls(weights=w, cumulative=np.cumsum(w))

    @property
    def n(self) -> int:
        """Number of frequency bins."""
        return int(self.weights.size)

    def sample(self, rng: np.random.Generator) -> float:
        """Draw a frequency in ``[-π, π)``."""
        j = int(np.searchsorted(self.cumulative, rng.random() * self.cumulative[-1], side="right"))
        j = min(j, self.n - 1)
        width = 2.0 * math.pi / self.n
        omega = j * width + (rng.random() - 0.5) * width
        return (omega + math.pi) % (2.0 * math.pi) - math.pi

    def log_density(self, omega: float) -> float:
        """Log density of a frequency (per radian)."""
        j = round(omega * self.n / (2.0 * math.pi)) % self.n
        p = float(self.weights[j])
        if p <= 0.0:
            return -math.inf
        return math.log(p * self.n / (2.0 * math.pi))


@dataclass(frozen=True, slots=True)
class PoleProposal:
    """Mixture proposal for new poles.

    With probability ``uniform_weight`` the pole is drawn uniformly on
    ``support`` (the prior support); otherwise its frequency is drawn from
    ``spectrum`` and its damping from an exponential distribution with mean
    ``damping_scale``. The mixture keeps the proposal's support equal to the
    prior support, which the
    reversible birth/death pair requires.
    """

    spectrum: SpectralDensity | None
    uniform_weight: float = 1.0
    damping_scale: float = 0.01
    support: UniformDisk = UniformDisk()

    def sample(self, rng: np.random.Generator) -> complex:
        """Draw one pole."""
        if self.spectrum is None or rng.random() < self.uniform_weight:
            return self.support.sample(rng)
        omega = self.spectrum.sample(rng)
        alpha = rng.exponential(self.damping_scale)
        return complex(np.exp(complex(-alpha, omega)))

    def log_density(self, z: complex) -> float:
        """Log density with respect to area measure on the unit disk."""
        r = abs(z)
        if not 0.0 < r < 1.0:
            return -math.inf
        log_uniform = self.support.log_density(z)
        if self.spectrum is None:
            return log_uniform
        alpha = -math.log(r)
        omega = math.atan2(z.imag, z.real)
        log_spec = (
            self.spectrum.log_density(omega)
            - math.log(self.damping_scale)
            - alpha / self.damping_scale
            + 2.0 * alpha
        )
        w = self.uniform_weight
        if w >= 1.0:
            return log_uniform
        if not math.isfinite(log_uniform):
            return math.log1p(-w) + log_spec
        return float(np.logaddexp(math.log(w) + log_uniform, math.log1p(-w) + log_spec))
