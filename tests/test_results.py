from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from rjnmr import RJMCMCResult, RJMCMCSampler, SamplerConfig
from rjnmr.results import MoveStats, Trace


@pytest.fixture(scope="module")
def result(two_peaks) -> RJMCMCResult:
    _, y = two_peaks
    return RJMCMCSampler(y, SamplerConfig(n_iterations=800, burn_in=800), seed=21).run()


def test_trace_indexing(result: RJMCMCResult) -> None:
    t = result.trace
    assert len(t) == 800
    assert t.offsets[-1] == t.poles.size == t.amplitudes.size
    s = t[-1]
    assert s.k == t.k[-1]
    assert len(s.components()) == s.k


def test_k_posterior_and_map(result: RJMCMCResult) -> None:
    pk = result.k_posterior()
    assert pk.sum() == pytest.approx(1.0)
    i = result.map_index()
    assert result.trace.k[i] == result.k_map
    same_k = result.trace.k == result.k_map
    assert result.trace.log_posterior[i] == result.trace.log_posterior[same_k].max()
    with pytest.raises(ValueError, match="no draws"):
        result.map_index(k=99)


def test_map_amplitude_modes(result: RJMCMCResult) -> None:
    mean = result.map_components(amplitudes="mean")
    sample = result.map_components(amplitudes="sample")
    assert [c.frequency for c in mean] == [c.frequency for c in sample]
    with pytest.raises(ValueError, match="amplitudes"):
        result.map_components(amplitudes="median")


def test_residual_is_noise_sized(result: RJMCMCResult) -> None:
    r = result.residual()
    # σ² = 0.5 per part → E|e|² = 1
    assert np.mean(np.abs(r) ** 2) == pytest.approx(1.0, rel=0.2)


def test_summarize_gate(result: RJMCMCResult) -> None:
    s = result.summarize(gate=-1.0)  # nothing can match
    assert all(c.probability == 0.0 for c in s)
    s = result.summarize()
    assert all(0.0 <= c.probability <= 1.0 for c in s)
    d = s[0].as_dict()
    assert {"probability", "frequency", "map_frequency", "map_amplitude"} <= d.keys()


def test_save_load_roundtrip(result: RJMCMCResult, tmp_path: Path) -> None:
    result.metadata["spectral_width"] = 1234.5
    path = tmp_path / "res.npz"
    result.save(path)
    back = RJMCMCResult.load(path)
    np.testing.assert_array_equal(back.trace.poles, result.trace.poles)
    np.testing.assert_array_equal(back.k_chain, result.k_chain)
    assert back.config == result.config
    assert isinstance(back.config.rw_scales, tuple)
    assert back.stats == result.stats
    assert back.metadata["spectral_width"] == 1234.5
    assert back.k_map == result.k_map


def test_concatenate(result: RJMCMCResult) -> None:
    both = RJMCMCResult.concatenate([result, result])
    assert len(both.trace) == 2 * len(result.trace)
    assert both.trace[len(result.trace)].iteration == result.trace[0].iteration
    np.testing.assert_array_equal(both.trace[len(result.trace) + 3].poles, result.trace[3].poles)
    assert both.stats.birth_proposed == 2 * result.stats.birth_proposed
    other = RJMCMCResult(
        result.data * 2, result.config, result.trace, result.k_chain, result.log_posterior_chain
    )
    with pytest.raises(ValueError, match="same data"):
        RJMCMCResult.concatenate([result, other])
    with pytest.raises(ValueError):
        RJMCMCResult.concatenate([])


def test_report(result: RJMCMCResult) -> None:
    text = result.report()
    assert "k_MAP" in text and "acceptance" in text
    assert "Hz" in result.report(spectral_width=1000.0)


def test_empty_trace_and_stats() -> None:
    assert len(Trace.empty()) == 0
    assert np.isnan(MoveStats().birth_rate)
