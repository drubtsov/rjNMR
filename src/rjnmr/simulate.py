"""Synthetic FIDs, including the eight-component benchmark of the paper (Table 1)."""

from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np
from numpy.typing import NDArray

from .model import Component, poles_from_components, synthesize

__all__ = ["PAPER_NOISE_VARIANCES", "paper_components", "simulate_fid"]

#: Noise variances of the three experiments in Table 1 of the paper.
PAPER_NOISE_VARIANCES: tuple[float, float, float] = (4.0, 16.0, 36.0)


def paper_components(n: int = 1024) -> list[Component]:
    """The eight-component synthetic signal of Table 1.

    Components 5–7 are separated by exactly one Fourier bin (``2π/N``), so the
    benchmark depends on the record length ``n``. The paper does not state
    ``N``; 1024 is a value for which every component decays within the record.
    """
    step = 2.0 * math.pi / n
    pi = math.pi
    return [
        Component(0.1, 0.006, 4.0, pi / 4),
        Component(0.05, 0.05, 250.0, 0.0),
        Component(1.0, 0.008, 4.0, pi / 4),
        Component(1.18, 0.2, 300.0, 0.0),
        Component(2.0, 0.003, 30.0, pi / 3),
        Component(2.0 + step, 0.003, 6.0, pi / 2),
        Component(2.0 + 2 * step, 0.0035, 30.0, 2 * pi / 3),
        Component(3.0, 0.016, 6.0, 3 * pi / 4),
    ]


def simulate_fid(
    components: Iterable[Component],
    n: int,
    noise_variance: float = 0.0,
    seed: int | np.random.Generator | None = None,
) -> NDArray[np.complex128]:
    """Simulate ``y = Σ a_i z_i^t + e_t`` with circular complex Gaussian noise.

    ``noise_variance`` is the variance of *each* of the real and imaginary
    parts, as in the paper.
    """
    if noise_variance < 0:
        raise ValueError("noise_variance must be non-negative")
    z, a = poles_from_components(components)
    y = synthesize(z, a, n)
    if noise_variance > 0:
        rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
        sd = math.sqrt(noise_variance)
        y = y + sd * (rng.standard_normal(n) + 1j * rng.standard_normal(n))
    return y
