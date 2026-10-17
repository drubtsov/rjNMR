"""Configuration of the Reversible Jump MCMC sampler."""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from typing import Any, Literal

__all__ = ["AmplitudeModel", "BirthProposal", "SamplerConfig", "UpdateKernel"]

UpdateKernel = Literal["cartesian", "polar"]
BirthProposal = Literal["uniform", "spectrum", "residual"]
AmplitudeModel = Literal["complex", "real"]


@dataclass(frozen=True, slots=True)
class SamplerConfig:
    r"""Settings for :class:`rjnmr.sampler.RJMCMCSampler`.

    Defaults follow Rubtsov & Griffin (2007) wherever the paper specifies a value.

    Parameters
    ----------
    n_iterations
        Number of iterations recorded after burn-in.
    burn_in
        Number of initial iterations that are discarded.
    thin
        Keep every ``thin``-th post-burn-in iteration.
    expected_components
        Mean :math:`\Lambda` of the truncated Poisson prior on ``k``
        (paper: 20).
    max_components
        Truncation :math:`k_{max}` of the prior on ``k``.
    move_scale
        Constant ``c`` in :math:`b_k = c\,\min(1, p(k+1)/p(k))`, which sets
        the share of birth/death moves (paper: 0.2).
    split_merge_scale
        Same constant for split/merge moves, which replace one pole ``z`` by
        ``z ± ε`` and vice versa (the extension suggested in Section 5 of the
        paper). They let the chain escape states in which a single line is
        represented by two overlapping components. ``0`` disables them and
        reproduces the paper's sampler. ``move_scale + split_merge_scale``
        must not exceed 0.5.
    split_scale
        Standard deviation per real/imaginary part of the split offset ``ε``.
        ``None`` uses ``2/N``.
    nu0, gamma0
        Inverse-gamma prior :math:`\sigma^2 \sim IG(\nu_0/2, \gamma_0/2)` on the
        noise variance. ``0, 0`` is the (improper) Jeffreys prior.
    delta2
        Fixed value of the g-prior scale :math:`\delta^2` (≈ expected SNR).
        ``None`` (default) samples it from its full conditional, eq. (14).
    delta2_init
        Starting value of :math:`\delta^2` when it is sampled.
    alpha_delta, beta_delta
        Hyper-parameters of :math:`\delta^2 \sim IG(\alpha_\delta, \beta_\delta)`
        (paper: 2 and 10).
    amplitude_model
        ``"complex"`` (paper): free complex amplitudes. ``"real"``: real
        amplitudes, i.e. zero-phase components, appropriate for
        phase-corrected spectra (the 2010 toolbox takes ``abs`` of the fitted
        amplitudes to the same end). It halves the number of amplitude
        parameters; the per-component penalty becomes ``(1+δ²)^{-1/2}``, so
        combine it with ``max_damping`` to keep broad components from being
        cheap.
    baseline
        Include a constant spectral offset (a spike at ``t = 0``, the "flat"
        component of the 2010 toolbox) as a nuisance component with a flat
        prior. It is integrated out exactly.
    frequency_range
        ``(low, high)`` in rad/sample: restrict the pole prior to this band
        (``high - low <= 2π``, interpreted modulo 2π). By default the prior
        covers all frequencies, or the window's band for windowed fits.
    max_damping
        Optional upper bound :math:`\alpha_{max}` on the damping factor. The
        pole prior then becomes uniform on the annulus
        :math:`e^{-\alpha_{max}} < |z| < 1`. ``None`` (default) keeps the
        paper's uniform prior on the whole disk. A bound such as ``0.5`` stops
        the chain from wasting components on extremely broad lines.
    min_independence
        Reject pole configurations in which some column of the design keeps
        less than this fraction of its energy after projection onto the
        others (relative Cholesky pivot of the Gram matrix). The default only
        guards against numerical singularity. Larger values such as ``1e-3``
        act as a resolvability prior. They forbid nearly coincident pairs of
        poles, which otherwise model lineshape distortions with huge
        mutually cancelling amplitudes.
    update_kernel
        ``"cartesian"`` – the paper's Gaussian random walk on the complex pole
        ``z`` (eq. 12); ``"polar"`` – independent random walks on frequency and
        damping, usually more efficient for narrow NMR lines.
    rw_variance
        Variance of the cartesian random walk per real/imaginary part.
        ``None`` uses the paper's ``1/(3N)``.
    rw_scales
        Multipliers of the random-walk step size. Each proposal picks one
        uniformly at random, independently of the state, so the kernel stays
        symmetric. Small multipliers let well-determined, high-SNR poles be
        refined, whose posterior width can be orders of magnitude below
        ``sqrt(1/(3N))``. ``(1.0,)`` reproduces the paper's single-scale walk.
    freq_step, damping_step
        Standard deviations of the polar random walk. ``None`` uses ``π/N``
        and ``1/N`` respectively.
    birth_proposal
        Distribution new poles are drawn from: ``"uniform"`` on the unit disk
        (paper), ``"spectrum"`` – frequency drawn from the magnitude spectrum
        of the data, or ``"residual"`` – from the spectrum of the current
        residual (the approach of the original Matlab code). The two spectral
        proposals are mixed with the uniform one (see ``uniform_mixture``) and
        their densities enter the acceptance ratio exactly.
    uniform_mixture
        Weight of the uniform-disk proposal in the spectral birth proposals.
    spectral_power
        Spectral proposal density is proportional to ``|FFT|**spectral_power``.
    spectral_oversample
        Zero-padding factor of the FFT used by the spectral proposals.
    damping_scale
        Mean of the exponential distribution used to draw the damping of a
        newborn pole for the spectral proposals. ``None`` uses ``10/N``.
    store_amplitudes
        Draw and store complex amplitudes and noise variance at every kept
        iteration.
    """

    n_iterations: int = 40_000
    burn_in: int = 10_000
    thin: int = 1
    expected_components: float = 20.0
    max_components: int = 200
    move_scale: float = 0.2
    split_merge_scale: float = 0.1
    split_scale: float | None = None
    nu0: float = 0.0
    gamma0: float = 0.0
    delta2: float | None = None
    delta2_init: float = 100.0
    alpha_delta: float = 2.0
    beta_delta: float = 10.0
    amplitude_model: AmplitudeModel = "complex"
    baseline: bool = False
    frequency_range: tuple[float, float] | None = None
    max_damping: float | None = None
    min_independence: float = 1e-10
    update_kernel: UpdateKernel = "cartesian"
    rw_variance: float | None = None
    rw_scales: tuple[float, ...] = (1.0, 0.1, 0.01, 0.001)
    freq_step: float | None = None
    damping_step: float | None = None
    birth_proposal: BirthProposal = "residual"
    uniform_mixture: float = 0.1
    spectral_power: float = 1.0
    spectral_oversample: int = 4
    damping_scale: float | None = None
    store_amplitudes: bool = True

    def __post_init__(self) -> None:
        if self.n_iterations < 0 or self.burn_in < 0:
            raise ValueError("n_iterations and burn_in must be non-negative")
        if self.thin < 1:
            raise ValueError("thin must be >= 1")
        if self.expected_components <= 0:
            raise ValueError("expected_components must be positive")
        if self.max_components < 1:
            raise ValueError("max_components must be >= 1")
        if not 0.0 < self.move_scale <= 0.5:
            raise ValueError("move_scale must lie in (0, 0.5]")
        if self.split_merge_scale < 0 or self.move_scale + self.split_merge_scale > 0.5:
            raise ValueError(
                "need split_merge_scale >= 0 and move_scale + split_merge_scale <= 0.5"
            )
        if self.nu0 < 0 or self.gamma0 < 0:
            raise ValueError("nu0 and gamma0 must be non-negative")
        if self.delta2 is not None and self.delta2 <= 0:
            raise ValueError("delta2 must be positive")
        if self.delta2_init <= 0:
            raise ValueError("delta2_init must be positive")
        if self.alpha_delta <= 0 or self.beta_delta <= 0:
            raise ValueError("alpha_delta and beta_delta must be positive")
        if self.update_kernel not in ("cartesian", "polar"):
            raise ValueError(f"unknown update_kernel {self.update_kernel!r}")
        if self.birth_proposal not in ("uniform", "spectrum", "residual"):
            raise ValueError(f"unknown birth_proposal {self.birth_proposal!r}")
        if not self.rw_scales or any(s <= 0 for s in self.rw_scales):
            raise ValueError("rw_scales must be a non-empty tuple of positive numbers")
        if self.amplitude_model not in ("complex", "real"):
            raise ValueError(f"unknown amplitude_model {self.amplitude_model!r}")
        if self.frequency_range is not None:
            lo, hi = self.frequency_range
            if not 0.0 < hi - lo <= 2.0 * math.pi:
                raise ValueError("frequency_range must satisfy 0 < high - low <= 2π")
        if not 0.0 < self.min_independence < 1.0:
            raise ValueError("min_independence must lie in (0, 1)")
        if self.spectral_oversample < 1:
            raise ValueError("spectral_oversample must be >= 1")
        if not 0.0 < self.uniform_mixture <= 1.0:
            raise ValueError("uniform_mixture must lie in (0, 1]")
        for name in (
            "rw_variance",
            "freq_step",
            "damping_step",
            "damping_scale",
            "max_damping",
            "split_scale",
        ):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive")

    @property
    def sample_delta2(self) -> bool:
        r"""Whether :math:`\delta^2` is sampled rather than fixed."""
        return self.delta2 is None

    def resolved(self, n: int) -> SamplerConfig:
        """Return a copy where data-length dependent defaults are filled in for ``n`` points."""
        return dataclasses.replace(
            self,
            rw_variance=self.rw_variance if self.rw_variance is not None else 1.0 / (3.0 * n),
            freq_step=self.freq_step if self.freq_step is not None else math.pi / n,
            damping_step=self.damping_step if self.damping_step is not None else 1.0 / n,
            damping_scale=self.damping_scale if self.damping_scale is not None else 10.0 / n,
            split_scale=self.split_scale if self.split_scale is not None else 2.0 / n,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-compatible dict."""
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SamplerConfig:
        """Inverse of :meth:`to_dict`; unknown keys are ignored."""
        names = {f.name for f in dataclasses.fields(cls)}
        kwargs = {k: v for k, v in data.items() if k in names}
        for key in ("rw_scales", "frequency_range"):  # JSON turns tuples into lists
            if kwargs.get(key) is not None:
                kwargs[key] = tuple(kwargs[key])
        return cls(**kwargs)
