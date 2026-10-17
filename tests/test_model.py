from __future__ import annotations

import math

import numpy as np
import pytest

from rjnmr.model import (
    Component,
    basis,
    components_from_poles,
    gram,
    gram_column,
    poles_from_components,
    project,
    synthesize,
    wrap_angle,
)


def _random_poles(rng: np.random.Generator, k: int) -> np.ndarray:
    return np.exp(-rng.uniform(0, 0.05, k) + 1j * rng.uniform(-np.pi, np.pi, k))


@pytest.mark.parametrize("n", [1, 7, 256, 4096])
def test_gram_matches_explicit_product(rng: np.random.Generator, n: int) -> None:
    z = _random_poles(rng, 6)
    d = basis(z, n)
    np.testing.assert_allclose(gram(z, n), d.conj().T @ d, rtol=1e-10, atol=1e-9)


def test_gram_is_stable_for_undamped_and_identical_poles() -> None:
    z = np.array([np.exp(0.3j), np.exp(0.3j), np.exp(0.3j + 1e-14)])
    g = gram(z, 1000)
    assert np.all(np.isfinite(g))
    np.testing.assert_allclose(g.real, 1000.0, rtol=1e-9)


def test_gram_column(rng: np.random.Generator) -> None:
    z = _random_poles(rng, 4)
    z_new = 0.98 * np.exp(1.1j)
    expected = basis(z, 300).conj().T @ basis([z_new], 300)[:, 0]
    np.testing.assert_allclose(gram_column(z, z_new, 300), expected, rtol=1e-10, atol=1e-9)


def test_project(rng: np.random.Generator) -> None:
    z = _random_poles(rng, 3)
    y = rng.standard_normal(200) + 1j * rng.standard_normal(200)
    np.testing.assert_allclose(project(z, y), basis(z, 200).conj().T @ y, rtol=1e-12, atol=1e-9)
    assert project([], y).size == 0


def test_synthesize_and_component_roundtrip() -> None:
    comps = [Component(0.4, 0.01, 3.0, 0.5), Component(-2.0, 0.05, 1.5, -1.0)]
    z, a = poles_from_components(comps)
    y = synthesize(z, a, 64)
    t = np.arange(64)
    expected = sum(
        c.amplitude * np.exp(1j * c.phase) * np.exp((-c.damping + 1j * c.frequency) * t)
        for c in comps
    )
    np.testing.assert_allclose(y, expected, rtol=1e-12)
    back = components_from_poles(z, a)
    assert [round(c.frequency, 12) for c in back] == [-2.0, 0.4]
    assert back[0].amplitude == pytest.approx(1.5)
    assert back[1].phase == pytest.approx(0.5)
    with pytest.raises(ValueError, match="same length"):
        synthesize(z, a[:1], 10)
    assert np.all(synthesize([], [], 5) == 0)


def test_component_units() -> None:
    c = Component(frequency=math.pi / 2, damping=0.01, amplitude=4.0)
    assert c.frequency_hz(1000.0) == pytest.approx(250.0)
    assert c.linewidth_hz(1000.0) == pytest.approx(0.01 * 1000.0 / math.pi)
    assert c.snr_db(4.0) == pytest.approx(10 * math.log10(16 / 8))
    assert abs(c.pole) == pytest.approx(math.exp(-0.01))


def test_wrap_angle() -> None:
    np.testing.assert_allclose(
        wrap_angle(np.array([0.0, math.pi, -math.pi, 3 * math.pi])),
        [0.0, -math.pi, -math.pi, -math.pi],
    )
