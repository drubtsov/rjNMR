from __future__ import annotations

import math

import numpy as np
import pytest

from rjnmr.proposals import PoleProposal, SpectralDensity, UniformDisk


def _integrate_on_disk(logpdf, r_min: float = 0.0, nr: int = 400, nt: int = 512) -> float:
    """∫ q dA in polar coordinates with a midpoint rule."""
    r = r_min + (np.arange(nr) + 0.5) * (1 - r_min) / nr
    t = -math.pi + (np.arange(nt) + 0.5) * 2 * math.pi / nt
    total = 0.0
    for ri in r:
        vals = np.array([math.exp(logpdf(ri * complex(math.cos(ti), math.sin(ti)))) for ti in t])
        total += vals.sum() * ri
    return total * (1 - r_min) / nr * 2 * math.pi / nt


@pytest.mark.parametrize("r_min", [0.0, 0.6])
def test_uniform_disk_normalised_and_supported(r_min: float, rng: np.random.Generator) -> None:
    u = UniformDisk(r_min)
    assert _integrate_on_disk(u.log_density, r_min, nr=20, nt=16) == pytest.approx(1.0, rel=1e-9)
    draws = np.array([u.sample(rng) for _ in range(2000)])
    assert np.all((np.abs(draws) > r_min) & (np.abs(draws) < 1))
    # radius² is uniform on (r_min², 1)
    assert np.mean(np.abs(draws) ** 2) == pytest.approx((1 + r_min**2) / 2, abs=0.02)
    assert u.log_density(1.5 + 0j) == -math.inf


def test_spectral_density_normalised_and_sampling_matches(rng: np.random.Generator) -> None:
    n = 64
    t = np.arange(n)
    y = 5 * np.exp((-0.05 + 1.0j) * t) + np.exp((-0.02 - 2.0j) * t)
    spec = SpectralDensity.from_signal(y)
    grid = np.linspace(-math.pi, math.pi, 20001)[:-1]
    dens = np.exp([spec.log_density(w) for w in grid])
    assert dens.sum() * (2 * math.pi / grid.size) == pytest.approx(1.0, rel=1e-3)
    draws = np.array([spec.sample(rng) for _ in range(20000)])
    assert np.all((draws >= -math.pi) & (draws < math.pi))
    bins = np.round(draws * n / (2 * math.pi)).astype(int) % n
    freq = np.bincount(bins, minlength=n) / draws.size
    np.testing.assert_allclose(freq, spec.weights, atol=0.01)


def test_spectral_density_handles_zero_signal() -> None:
    spec = SpectralDensity.from_signal(np.zeros(16, complex))
    np.testing.assert_allclose(spec.weights, 1 / 16)


@pytest.mark.parametrize("r_min", [0.0, 0.7])
def test_mixture_proposal_is_a_normalised_density(r_min: float) -> None:
    n = 32
    t = np.arange(n)
    y = np.exp((-0.1 + 0.7j) * t)
    prop = PoleProposal(
        SpectralDensity.from_signal(y),
        uniform_weight=0.3,
        damping_scale=0.2,
        support=UniformDisk(r_min),
    )
    # spectral part may put mass below r_min, which is then rejected by the sampler,
    # so integrate over the whole disk
    total = _integrate_on_disk(prop.log_density, 0.0, nr=600, nt=n * 16)
    assert total == pytest.approx(1.0, abs=5e-3)


def test_mixture_proposal_sampling_matches_density(rng: np.random.Generator) -> None:
    n = 32
    t = np.arange(n)
    y = np.exp((-0.1 + 0.7j) * t)
    prop = PoleProposal(SpectralDensity.from_signal(y), uniform_weight=0.25, damping_scale=0.3)
    draws = np.array([prop.sample(rng) for _ in range(40000)])
    # P(|z| > 0.8) by Monte Carlo vs numerical integral of the density
    mc = np.mean(np.abs(draws) > 0.8)
    exact = _integrate_on_disk(prop.log_density, 0.8, nr=200, nt=n * 16)
    assert mc == pytest.approx(exact, abs=0.01)
