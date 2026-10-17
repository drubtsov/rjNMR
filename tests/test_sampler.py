from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.stats import poisson

from rjnmr import Component, RJMCMCSampler, SamplerConfig, run_chains, simulate_fid
from rjnmr.model import basis


def _brute_yPy(y: np.ndarray, z: np.ndarray, delta2: float, gamma0: float = 0.0) -> float:
    """yᴴ(I − D M Dᴴ)y with M = (DᴴD + Σ⁻¹)⁻¹, Σ⁻¹ = δ⁻² DᴴD — the paper's eq. (6)."""
    d = basis(z, y.size)
    dd = d.conj().T @ d
    m = np.linalg.inv(dd + dd / delta2)
    p = np.eye(y.size) - d @ m @ d.conj().T
    return float(gamma0 + (y.conj() @ p @ y).real)


def test_marginal_likelihood_matches_brute_force(rng: np.random.Generator) -> None:
    y = rng.standard_normal(80) + 1j * rng.standard_normal(80)
    z = np.exp(-np.array([0.01, 0.05, 0.2]) + 1j * np.array([0.3, -1.2, 2.5]))
    s = RJMCMCSampler(y, SamplerConfig(gamma0=0.5))
    st = s.initial_state(z)
    for d2 in (0.5, 10.0, 1e4):
        assert s._yPy(st.fit.Q, d2) == pytest.approx(_brute_yPy(y, z, d2, 0.5), rel=1e-10)


def test_incremental_updates_keep_sufficient_statistics_exact(two_peaks) -> None:
    _, y = two_peaks
    s = RJMCMCSampler(y, SamplerConfig(n_iterations=0, burn_in=0), seed=4)
    state = s.initial_state()
    for _ in range(300):
        s.step(state)
    z = state.poles
    d = basis(z, y.size)
    fresh = s._fit(z, d.conj().T @ d, d.conj().T @ y)
    assert fresh is not None
    np.testing.assert_allclose(state.fit.G, fresh.G, rtol=1e-9, atol=1e-8)
    np.testing.assert_allclose(state.fit.b, fresh.b, rtol=1e-9, atol=1e-8)
    assert state.fit.Q == pytest.approx(fresh.Q, rel=1e-9)


@pytest.mark.parametrize(
    ("birth", "max_damping", "split_merge"),
    [
        ("uniform", None, 0.0),
        ("uniform", 0.5, 0.0),
        ("residual", None, 0.0),
        ("spectrum", 0.5, 0.0),
        ("residual", None, 0.3),
        ("uniform", 0.5, 0.3),
    ],
)
def test_prior_recovery(birth: str, max_damping: float | None, split_merge: float) -> None:
    """With a flat likelihood the chain must sample the prior: k ~ Poisson(Λ / (1 + δ²)).

    This checks the birth/death acceptance ratios, including the dimension
    matching and proposal densities, independently of any data. The data are a
    non-trivial sinusoid (so the spectral and residual proposals are far from
    uniform) but ``gamma0`` is so large that the likelihood is numerically flat.
    """
    lam, delta2 = 6.0, 3.0
    y = np.exp((-0.05 + 0.7j) * np.arange(8))
    cfg = SamplerConfig(
        n_iterations=60_000,
        burn_in=500,
        delta2=delta2,
        gamma0=1e15,
        expected_components=lam,
        birth_proposal=birth,  # type: ignore[arg-type]
        max_damping=max_damping,
        move_scale=0.2 if split_merge == 0 else 0.1,
        split_merge_scale=split_merge,
        split_scale=0.3,  # large offsets so that splits explore the whole disk
        store_amplitudes=False,
    )
    res = RJMCMCSampler(y, cfg, seed=7).run()
    if split_merge:
        assert res.stats.split_accepted > 1000
        assert res.stats.merge_accepted > 1000
    pk = res.k_posterior()
    expected = poisson.pmf(np.arange(pk.size), lam / (1 + delta2))
    np.testing.assert_allclose(pk[:5], expected[:5], atol=0.02)
    # and the paper's extra 1/(k+1) factor would give p(k) ∝ μ^k/(k!)², clearly different
    mu = lam / (1 + delta2)
    wrong = np.array([mu**k / math.factorial(k) ** 2 for k in range(5)])
    wrong /= wrong.sum()
    assert np.abs(pk[:5] - wrong).max() > 0.05


def test_prior_recovery_poles_uniform_on_disk() -> None:
    cfg = SamplerConfig(
        n_iterations=20_000,
        burn_in=200,
        delta2=1.0,
        gamma0=1.0,
        expected_components=4,
        birth_proposal="uniform",
        store_amplitudes=False,
    )
    res = RJMCMCSampler(np.zeros(8, complex), cfg, seed=3).run()
    r2 = np.abs(res.trace.poles) ** 2
    assert np.mean(r2) == pytest.approx(0.5, abs=0.03)
    assert np.mean(np.angle(res.trace.poles)) == pytest.approx(0.0, abs=0.1)


@pytest.mark.parametrize("kernel", ["cartesian", "polar"])
def test_recovers_two_components(two_peaks, kernel: str) -> None:
    comps, y = two_peaks
    cfg = SamplerConfig(n_iterations=1500, burn_in=1500, update_kernel=kernel)  # type: ignore[arg-type]
    res = RJMCMCSampler(y, cfg, seed=11).run()
    assert res.k_map == 2
    est = res.map_components()
    truth = sorted(comps, key=lambda c: c.frequency)
    for e, t in zip(est, truth, strict=True):
        assert e.frequency == pytest.approx(t.frequency, abs=2e-3)
        assert e.damping == pytest.approx(t.damping, abs=3e-3)
        assert e.amplitude == pytest.approx(t.amplitude, rel=0.1)
        assert e.phase == pytest.approx(t.phase, abs=0.1)
    s = res.summarize()
    assert all(c.probability > 0.95 for c in s)
    assert res.stats.update_proposed > 0 and res.stats.birth_accepted > 0


def test_single_line_is_not_split() -> None:
    """A high-SNR single line must not end up as two overlapping components."""
    y = simulate_fid([Component(0.5, 0.01, 10.0)], n=256, noise_variance=0.1, seed=0)
    res = RJMCMCSampler(y, SamplerConfig(n_iterations=1500, burn_in=1500), seed=1).run()
    assert res.k_map == 1
    (c,) = res.map_components()
    assert c.frequency == pytest.approx(0.5, abs=1e-3)
    assert c.amplitude == pytest.approx(10.0, rel=0.02)


def test_paper_sampler_without_extensions(two_peaks) -> None:
    _, y = two_peaks
    cfg = SamplerConfig(
        n_iterations=300,
        burn_in=300,
        split_merge_scale=0.0,
        rw_scales=(1.0,),
        birth_proposal="uniform",
    )
    res = RJMCMCSampler(y, cfg, seed=0).run()
    assert res.stats.split_proposed == res.stats.merge_proposed == 0


def test_noise_variance_and_delta_are_sampled_sensibly(two_peaks) -> None:
    _, y = two_peaks
    res = RJMCMCSampler(y, SamplerConfig(n_iterations=1000, burn_in=1000), seed=2).run()
    assert np.median(res.trace.sigma2) == pytest.approx(0.5, rel=0.15)
    assert np.all(res.trace.delta2 > 0)


def test_reproducible_with_seed(two_peaks) -> None:
    _, y = two_peaks
    cfg = SamplerConfig(n_iterations=200, burn_in=100)
    a = RJMCMCSampler(y, cfg, seed=5).run()
    b = RJMCMCSampler(y, cfg, seed=5).run()
    np.testing.assert_array_equal(a.trace.poles, b.trace.poles)
    np.testing.assert_array_equal(a.k_chain, b.k_chain)


def test_run_chains_is_deterministic(two_peaks) -> None:
    _, y = two_peaks
    cfg = SamplerConfig(n_iterations=100, burn_in=50)
    r1 = run_chains(y, cfg, n_chains=2, seed=1, n_jobs=1)
    r2 = run_chains(y, cfg, n_chains=2, seed=1, n_jobs=2)
    for a, b in zip(r1, r2, strict=True):
        np.testing.assert_array_equal(a.trace.poles, b.trace.poles)
    assert not np.array_equal(r1[0].k_chain, r1[1].k_chain) or not np.array_equal(
        r1[0].trace.poles, r1[1].trace.poles
    )


def test_initial_state_validation(two_peaks) -> None:
    _, y = two_peaks
    s = RJMCMCSampler(y, SamplerConfig(max_damping=0.1))
    with pytest.raises(ValueError, match="prior support"):
        s.initial_state([1.2])
    with pytest.raises(ValueError, match="prior support"):
        s.initial_state([0.5])  # damping log(2) > 0.1
    # duplicated poles are dropped rather than producing a singular Gram matrix
    st = s.initial_state([0.99 * np.exp(0.5j), 0.99 * np.exp(0.5j)])
    assert st.k == 1


def test_zero_data_requires_gamma0() -> None:
    with pytest.raises(ValueError, match="gamma0"):
        RJMCMCSampler(np.zeros(4, complex))


def test_thinning_and_callback(two_peaks) -> None:
    _, y = two_peaks
    seen: list[int] = []
    res = RJMCMCSampler(y, SamplerConfig(n_iterations=100, burn_in=10, thin=10), seed=0).run(
        callback=lambda it, st: seen.append(it)
    )
    assert len(res.trace) == 10
    assert seen == list(range(110))
    assert res.k_chain.size == 110


@pytest.mark.slow
def test_paper_benchmark_high_snr() -> None:
    """Experiment 1 of the paper (σ² = 4): all eight components should be found."""
    from rjnmr import paper_components

    y = simulate_fid(paper_components(1024), 1024, 4.0, seed=0)
    res = RJMCMCSampler(y, SamplerConfig(n_iterations=10_000, burn_in=10_000), seed=3).run()
    assert res.k_map in (8, 9, 10)
    found = [c.frequency for c in res.map_components()]
    for comp in paper_components(1024):
        if comp.amplitude >= 30:
            assert min(abs(f - comp.frequency) for f in found) < 2e-3


def test_component_count_is_bounded(two_peaks) -> None:
    _, y = two_peaks
    res = RJMCMCSampler(
        y, SamplerConfig(n_iterations=300, burn_in=300, max_components=1), seed=1
    ).run()
    assert res.k_chain.max() <= 1
    assert res.k_map == 1
    assert isinstance(res.map_components()[0], Component)


# ------------------------------------------------- 2010 extensions: windows, baseline, real ---


@pytest.mark.parametrize(
    ("amplitude_model", "windowed", "baseline"),
    [("real", False, False), ("complex", True, True), ("real", True, True)],
)
def test_prior_recovery_extensions(amplitude_model: str, windowed: bool, baseline: bool) -> None:
    """Flat-likelihood check of the real-amplitude model, windowed design and band prior."""
    from rjnmr.intervals import raised_cosine_window

    lam, delta2, n = 6.0, 3.0, 16
    y = np.exp((-0.05 + 0.7j) * np.arange(n))
    window = raised_cosine_window(n, 0.0, 2.0, 0.3) if windowed else None
    cfg = SamplerConfig(
        n_iterations=60_000,
        burn_in=500,
        delta2=delta2,
        gamma0=1e15,
        expected_components=lam,
        amplitude_model=amplitude_model,  # type: ignore[arg-type]
        baseline=baseline,
        move_scale=0.15,
        split_merge_scale=0.15,
        split_scale=0.3,
        store_amplitudes=False,
    )
    sampler = RJMCMCSampler(y, cfg, seed=11, window=window)
    res = sampler.run()
    kappa = 0.5 if amplitude_model == "real" else 1.0
    expected = poisson.pmf(np.arange(5), lam / (1 + delta2) ** kappa)
    np.testing.assert_allclose(res.k_posterior()[:5], expected, atol=0.02)
    if windowed:
        start, width = sampler.support.omega_start, sampler.support.omega_width
        rel = (np.angle(res.trace.poles) - start) % (2 * np.pi)
        assert np.all(rel < width)
        # uniform on the sector: relative angle is uniform on [0, width)
        assert np.mean(rel) == pytest.approx(width / 2, rel=0.05)


def test_frequency_range_restricts_poles(two_peaks) -> None:
    _, y = two_peaks
    cfg = SamplerConfig(n_iterations=300, burn_in=300, frequency_range=(0.0, 1.0))
    res = RJMCMCSampler(y, cfg, seed=3).run()
    assert np.all((np.angle(res.trace.poles) >= 0) & (np.angle(res.trace.poles) < 1.0))
    assert any(abs(c.frequency - 0.5) < 2e-3 for c in res.map_components())
    with pytest.raises(ValueError, match="prior support"):
        RJMCMCSampler(y, cfg).initial_state([0.9 * np.exp(-1j)])


def test_real_amplitudes_recover_zero_phase_lines() -> None:
    comps = [Component(0.5, 0.01, 10.0, 0.0), Component(-1.0, 0.02, 5.0, 0.0)]
    y = simulate_fid(comps, 512, 0.5, seed=0)
    cfg = SamplerConfig(
        n_iterations=1000,
        burn_in=1000,
        amplitude_model="real",
        baseline=True,
        expected_components=2,
        max_damping=0.1,  # real amplitudes halve the Occam penalty; bound broad lines
    )
    res = RJMCMCSampler(y, cfg, seed=2).run()
    assert res.k_map == 2
    for c, t in zip(res.map_components(), sorted(comps, key=lambda c: c.frequency), strict=True):
        assert c.frequency == pytest.approx(t.frequency, abs=2e-3)
        assert c.amplitude == pytest.approx(t.amplitude, rel=0.1)
        assert min(abs(c.phase), abs(abs(c.phase) - np.pi)) < 1e-12  # real amplitudes
    assert np.all(res.trace.amplitudes.imag == 0)


def test_windowed_fit_recovers_in_band_components() -> None:
    from rjnmr.intervals import raised_cosine_window

    comps = [
        Component(0.5, 0.005, 10.0, 0.4),
        Component(0.56, 0.008, 4.0, -0.3),
        Component(-1.0, 0.004, 50.0, 0.0),  # strong line outside the band
    ]
    n = 1024
    y = simulate_fid(comps, n, 0.5, seed=7)
    window = raised_cosine_window(n, 0.4, 0.66, 0.2)
    cfg = SamplerConfig(n_iterations=1000, burn_in=1000, baseline=True)
    res = RJMCMCSampler(y, cfg, seed=1, window=window).run()
    est = res.map_components()
    assert res.k_map == 2
    for e, t in zip(est, comps[:2], strict=True):
        assert e.frequency == pytest.approx(t.frequency, abs=2e-3)
        assert e.amplitude == pytest.approx(t.amplitude, rel=0.1)
        assert e.phase == pytest.approx(t.phase, abs=0.1)
