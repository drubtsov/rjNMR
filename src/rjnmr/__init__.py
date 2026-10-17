"""rjnmr — Bayesian detection and estimation of damped sinusoids in NMR FIDs.

A Python implementation of the Reversible Jump MCMC algorithm of

    D. V. Rubtsov and J. L. Griffin, "Time-domain Bayesian detection and
    estimation of noisy damped sinusoidal signals applied to NMR
    spectroscopy", J. Magn. Reson. 188 (2007) 367–379,

and of the full-spectrum pipeline of the 2010 rjNMR toolbox (Rubtsov et al.,
Anal. Chem. 82 (2010) 4479–4485).
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from .config import SamplerConfig
from .intervals import DeconvolutionResult, auto_bands, deconvolve
from .model import Component, components_from_poles, poles_from_components, synthesize
from .phasing import auto_phase
from .results import ComponentSummary, RJMCMCResult
from .sampler import RJMCMCSampler, run_chains
from .simulate import paper_components, simulate_fid

try:
    __version__ = version("rjnmr")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0"

__all__ = [
    "Component",
    "ComponentSummary",
    "DeconvolutionResult",
    "RJMCMCResult",
    "RJMCMCSampler",
    "SamplerConfig",
    "__version__",
    "auto_bands",
    "auto_phase",
    "components_from_poles",
    "deconvolve",
    "paper_components",
    "poles_from_components",
    "run_chains",
    "simulate_fid",
    "synthesize",
]
