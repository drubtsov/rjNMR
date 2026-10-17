from __future__ import annotations

import pytest

from rjnmr import SamplerConfig


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_iterations": -1},
        {"thin": 0},
        {"expected_components": 0},
        {"max_components": 0},
        {"move_scale": 0.6},
        {"move_scale": 0.3, "split_merge_scale": 0.3},
        {"nu0": -1},
        {"delta2": 0},
        {"delta2_init": 0},
        {"alpha_delta": 0},
        {"update_kernel": "hmc"},
        {"birth_proposal": "prior"},
        {"uniform_mixture": 0},
        {"rw_scales": ()},
        {"rw_scales": (1.0, -1.0)},
        {"spectral_oversample": 0},
        {"max_damping": -0.1},
        {"split_scale": 0},
    ],
)
def test_invalid_config(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        SamplerConfig(**kwargs)


def test_resolved_defaults_and_roundtrip() -> None:
    cfg = SamplerConfig().resolved(300)
    assert cfg.rw_variance == pytest.approx(1 / 900)
    assert cfg.split_scale == pytest.approx(2 / 300)
    assert cfg.damping_scale == pytest.approx(10 / 300)
    assert SamplerConfig.from_dict({**cfg.to_dict(), "unknown": 1}) == cfg
    assert SamplerConfig(delta2=5.0).sample_delta2 is False
