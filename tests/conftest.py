from __future__ import annotations

import numpy as np
import pytest

from rjnmr import Component, simulate_fid


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(12345)


@pytest.fixture(scope="session")
def two_peaks() -> tuple[list[Component], np.ndarray]:
    comps = [Component(0.5, 0.01, 10.0, 0.3), Component(-1.0, 0.02, 5.0, 1.0)]
    return comps, simulate_fid(comps, 512, 0.5, seed=0)
