r"""Reversible Jump MCMC sampler for complex damped sinusoids.

Implements the algorithm of Rubtsov & Griffin, *J. Magn. Reson.* 188 (2007)
367–379, which extends Andrieu & Doucet (1999) to complex-valued signals with
exponential damping, together with the band-limited frequency-domain variant of
the 2010 Matlab toolbox (see :mod:`rjnmr.design`).

Target distribution
-------------------
After integrating out the amplitudes :math:`a` and the noise variance
:math:`\sigma^2` the posterior over the model order ``k`` and the signal poles
:math:`z_{1:k}` is (paper eq. 7)

.. math::

    p(k, z_{1:k} \mid y, \delta^2) \propto
    \left[\gamma_0 + y^H P_k y\right]^{-(N + \nu_0/2)}
    \frac{1}{(1 + \delta^2)^{\kappa k}} \frac{\Lambda^k}{k!} p(z)^k,

with :math:`y^H P_k y = y^H y - \frac{\delta^2}{1+\delta^2} b^H G^{-1} b`,
:math:`G = D^H D`, :math:`b = D^H y`, and :math:`p(z)` uniform on the prior
support (by default the unit disk). :math:`\kappa = 1` for complex amplitudes
and :math:`\kappa = 1/2` for real amplitudes, where the real parts of
:math:`G` and :math:`b` are used.

Moves
-----
Each iteration chooses a birth (probability :math:`b_k`), death (:math:`d_k`),
split (:math:`s_k`), merge (:math:`m_k`) or update move, followed by Gibbs draws of
:math:`\sigma^2`, :math:`a` and (optionally) :math:`\delta^2` from their full
conditionals (eqs. 5, 6, 14).

.. note::

   The birth/death acceptance ratios implemented here do *not* contain the
   factor ``1/(k+1)`` that appears in eqs. (15)–(16) of the paper. With a
   label-exchangeable prior and a death move that removes a uniformly chosen
   component, that factor cancels against the ``(k+1)`` label permutations of
   the newborn component; keeping it biases ``p(k|y)`` towards smaller models.
   ``tests/test_sampler.py::test_prior_recovery`` verifies the corrected
   ratio: with a flat likelihood the sampler reproduces
   ``k ~ Poisson(Λ / (1 + δ²)^κ)`` exactly.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.linalg import lapack

from .config import SamplerConfig
from .design import Design, make_design
from .proposals import PoleProposal, UniformDisk
from .results import MoveStats, RJMCMCResult, Trace

__all__ = ["ChainState", "RJMCMCSampler", "run_chains"]

logger = logging.getLogger(__name__)

ComplexArray = NDArray[np.complex128]
FloatArray = NDArray[np.float64]


# --------------------------------------------------------------------------- linear algebra ---


@dataclass(slots=True)
class _Fit:
    """Sufficient statistics of a pole configuration."""

    z: ComplexArray  # (k,)
    G: ComplexArray  # (k, k) Gram matrix
    b: ComplexArray  # (k,) projection of the data
    L: NDArray[Any]  # lower Cholesky factor of G (or of Re G for real amplitudes)
    Q: float  # b^H G^{-1} b  (or Re b^T (Re G)^{-1} Re b)

    @property
    def k(self) -> int:
        return int(self.z.size)


def _tri_solve(L: NDArray[Any], b: NDArray[Any], *, conj_trans: bool = False) -> NDArray[Any]:
    """Solve ``L x = b`` (or ``Lᴴ x = b``) for lower-triangular ``L`` via LAPACK ``?trtrs``."""
    if np.iscomplexobj(L) or np.iscomplexobj(b):
        x, info = lapack.ztrtrs(L, b, lower=1, trans=2 if conj_trans else 0)
    else:
        x, info = lapack.dtrtrs(L, b, lower=1, trans=1 if conj_trans else 0)
    if info != 0:  # pragma: no cover - L comes from a successful Cholesky factorisation
        raise np.linalg.LinAlgError(f"trtrs failed with info={info}")
    out: NDArray[Any] = x
    return out


# ------------------------------------------------------------------------------ chain state ---


@dataclass(slots=True)
class ChainState:
    """Current state of a chain (exposed to callbacks)."""

    fit: _Fit
    delta2: float
    sigma2: float = math.nan
    amplitudes: ComplexArray = field(default_factory=lambda: np.zeros(0, dtype=np.complex128))

    @property
    def k(self) -> int:
        """Current number of components."""
        return self.fit.k

    @property
    def poles(self) -> ComplexArray:
        """Current signal poles."""
        return self.fit.z


# ---------------------------------------------------------------------------------- sampler ---


class RJMCMCSampler:
    """Reversible Jump MCMC sampler for a complex FID.

    Parameters
    ----------
    y
        Complex time-domain data of length ``N``.
    config
        Sampler settings; see :class:`~rjnmr.config.SamplerConfig`.
    seed
        Seed or generator for reproducible runs.
    window
        Optional frequency-domain window over the ``N`` DFT bins (natural
        order). When given, the likelihood is the band-limited windowed
        spectral likelihood of the 2010 toolbox
        (:class:`~rjnmr.design.WindowedSpectralDesign`) and, unless
        ``config.frequency_range`` says otherwise, the pole prior is restricted
        to the window's frequency band.

    Examples
    --------
    >>> import numpy as np
    >>> from rjnmr import RJMCMCSampler, SamplerConfig, simulate_fid, Component
    >>> y = simulate_fid([Component(0.5, 0.01, 10.0)], n=256, noise_variance=0.1, seed=0)
    >>> cfg = SamplerConfig(n_iterations=1500, burn_in=1500)
    >>> res = RJMCMCSampler(y, cfg, seed=1).run()
    >>> int(res.k_map)
    1
    """

    def __init__(
        self,
        y: ArrayLike,
        config: SamplerConfig | None = None,
        seed: int | np.random.Generator | np.random.SeedSequence | None = None,
        *,
        window: ArrayLike | None = None,
    ) -> None:
        self.y: ComplexArray = np.ascontiguousarray(np.asarray(y, dtype=np.complex128).ravel())
        if self.y.size < 2:
            raise ValueError("need at least two data points")
        self.n = int(self.y.size)
        self.config = (config or SamplerConfig()).resolved(self.n)
        cfg = self.config
        self.window = None if window is None else np.asarray(window, dtype=np.float64).ravel()
        self.design: Design = make_design(self.y, window=self.window, baseline=cfg.baseline)
        self.rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
        self._yy = self.design.yy
        if self._yy <= 0.0 and cfg.gamma0 == 0.0:
            raise ValueError("all-zero data requires gamma0 > 0")
        self._real = cfg.amplitude_model == "real"
        self._min_pivot = cfg.min_independence
        self._kappa = 0.5 if self._real else 1.0
        self._log_exponent = self.design.n_obs + 0.5 * cfg.nu0
        self._data_spectrum = (
            self.design.proposal_spectrum(
                np.zeros(0, np.complex128),
                np.zeros(0, np.complex128),
                cfg.spectral_power,
                cfg.spectral_oversample,
            )
            if cfg.birth_proposal == "spectrum"
            else None
        )
        lam, kmax = cfg.expected_components, cfg.max_components
        ks = np.arange(kmax + 1)
        self._log_prior_k = ks * math.log(lam) - np.array([math.lgamma(k + 1.0) for k in ks])
        self.stats = MoveStats()
        # Uniform prior on an annular sector of the disk.
        r_min = 0.0 if cfg.max_damping is None else math.exp(-cfg.max_damping)
        if cfg.frequency_range is not None:
            lo, hi = cfg.frequency_range
            start, width = lo, hi - lo
        elif self.design.frequency_support is not None:
            start, width = self.design.frequency_support
        else:
            start, width = -math.pi, 2.0 * math.pi
        self.support = UniformDisk(r_min, start, width)
        self._log_prior_pole = -self.support.log_area

    # ------------------------------------------------------------------ target pieces --

    def _yPy(self, Q: float, delta2: float) -> float:
        return self.config.gamma0 + self._yy - delta2 / (1.0 + delta2) * Q

    def _log_lik(self, Q: float, delta2: float) -> float:
        ypy = self._yPy(Q, delta2)
        if ypy <= 0.0:
            return -math.inf
        return -self._log_exponent * math.log(ypy)

    def _log_penalty(self, delta2: float) -> float:
        """Log of the per-component factor ``(1 + δ²)^{-κ}``."""
        return -self._kappa * math.log1p(delta2)

    def log_posterior(self, state: ChainState) -> float:
        """Log of the (unnormalised) marginal posterior ``p(k, z | y, δ²) p(δ²)``."""
        k, d2 = state.k, state.delta2
        lp = (
            self._log_lik(state.fit.Q, d2)
            + k * self._log_penalty(d2)
            + float(self._log_prior_k[k])
            + k * self._log_prior_pole
        )
        if self.config.sample_delta2:
            a, b = self.config.alpha_delta, self.config.beta_delta
            lp += a * math.log(b) - math.lgamma(a) - (a + 1.0) * math.log(d2) - b / d2
        return lp

    def _birth_death_probs(self, k: int) -> tuple[float, float]:
        c, kmax = self.config.move_scale, self.config.max_components
        lp = self._log_prior_k
        b = 0.0 if k >= kmax else c * min(1.0, math.exp(lp[k + 1] - lp[k]))
        d = 0.0 if k == 0 else c * min(1.0, math.exp(lp[k - 1] - lp[k]))
        return b, d

    # ------------------------------------------------------------------- construction --

    def _fit(self, z: ComplexArray, G: ComplexArray, b: ComplexArray) -> _Fit | None:
        """Factorise ``G`` and evaluate the quadratic form; ``None`` if numerically degenerate."""
        if z.size == 0:
            empty = np.zeros((0, 0), dtype=np.float64 if self._real else np.complex128)
            return _Fit(z, G, b, empty, 0.0)
        Gm = G.real if self._real else G
        bm = b.real if self._real else b
        try:
            L = np.linalg.cholesky(Gm)
        except np.linalg.LinAlgError:
            return None
        # Reject numerically near-singular configurations (virtually coincident poles):
        # each column must keep a fraction of its norm after projecting out the others.
        if np.min(np.abs(np.diag(L)) ** 2 / np.abs(np.diag(Gm))) < self._min_pivot:
            return None
        v = _tri_solve(L, bm)
        Q = float(np.vdot(v, v).real)
        # Q is the energy of the projection of the data onto the model space.
        if not np.isfinite(Q) or Q > self._yy * (1.0 + 1e-9):
            return None
        return _Fit(z, G, b, L, Q)

    def initial_state(self, poles: ArrayLike | None = None) -> ChainState:
        """Build a valid starting state from optional initial poles."""
        z = np.asarray([] if poles is None else poles, dtype=np.complex128).ravel()
        outside = [complex(zi) for zi in z if not self.support.contains(complex(zi))]
        if outside:
            raise ValueError(
                f"initial poles must lie inside the prior support (|z| in "
                f"({self.support.r_min:.3g}, 1), frequency band {self.support.omega_start:.4f} "
                f"+ {self.support.omega_width:.4f}); offending: {outside[:3]}"
            )
        if z.size > self.config.max_components:
            raise ValueError("more initial poles than max_components")
        # Drop poles that make the Gram matrix numerically singular (e.g. duplicates).
        kept: list[complex] = []
        for zi in z:
            trial = np.array([*kept, zi], dtype=np.complex128)
            if self._fit(trial, self.design.gram(trial), self.design.project(trial)) is not None:
                kept.append(complex(zi))
        zk = np.array(kept, dtype=np.complex128)
        fit = self._fit(zk, self.design.gram(zk), self.design.project(zk))
        assert fit is not None
        delta2 = self.config.delta2 if self.config.delta2 is not None else self.config.delta2_init
        state = ChainState(fit=fit, delta2=float(delta2))
        self._gibbs(state)
        return state

    def _in_support(self, z: complex) -> bool:
        return self.support.contains(z)

    # ------------------------------------------------------------------------ proposals --

    def _posterior_mean(self, fit: _Fit, delta2: float) -> ComplexArray:
        if fit.k == 0:
            return np.zeros(0, dtype=np.complex128)
        bm = fit.b.real if self._real else fit.b
        w = _tri_solve(fit.L, bm)
        m = delta2 / (1.0 + delta2) * _tri_solve(fit.L, w, conj_trans=True)
        return np.asarray(m, dtype=np.complex128)

    def _proposal_for(self, fit: _Fit, delta2: float) -> PoleProposal:
        cfg = self.config
        assert cfg.damping_scale is not None
        if cfg.birth_proposal == "uniform":
            return PoleProposal(None, support=self.support)
        if cfg.birth_proposal == "spectrum":
            spec = self._data_spectrum
        else:
            spec = self.design.proposal_spectrum(
                fit.z,
                self._posterior_mean(fit, delta2),
                cfg.spectral_power,
                cfg.spectral_oversample,
            )
        return PoleProposal(spec, cfg.uniform_mixture, cfg.damping_scale, self.support)

    # ---------------------------------------------------------------------------- moves --

    def _with_new_pole(self, fit: _Fit, z_new: complex) -> _Fit | None:
        k = fit.k
        z = np.append(fit.z, z_new)
        col = self.design.gram_column(z, z_new)
        G = np.empty((k + 1, k + 1), dtype=np.complex128)
        G[:k, :k] = fit.G
        G[:, k] = col
        G[k, :] = col.conj()
        G[k, k] = col[k].real
        b = np.append(fit.b, self.design.project_one(z_new))
        return self._fit(z, G, b)

    def _birth(self, state: ChainState) -> None:
        fit, d2 = state.fit, state.delta2
        k = fit.k
        self.stats.birth_proposed += 1
        proposal = self._proposal_for(fit, d2)
        z_new = proposal.sample(self.rng)
        log_q = proposal.log_density(z_new)
        if not self._in_support(z_new) or not math.isfinite(log_q):
            return
        new = self._with_new_pole(fit, z_new)
        if new is None:
            return
        bk, _ = self._birth_death_probs(k)
        _, dk1 = self._birth_death_probs(k + 1)
        log_alpha = (
            self._log_lik(new.Q, d2)
            - self._log_lik(fit.Q, d2)
            + self._log_penalty(d2)
            + self._log_prior_pole
            - log_q
            + self._log_prior_k[k + 1]
            - self._log_prior_k[k]
            + math.log(dk1)
            - math.log(bk)
        )
        if math.log(self.rng.random()) < log_alpha:
            state.fit = new
            self.stats.birth_accepted += 1

    def _death(self, state: ChainState) -> None:
        fit, d2 = state.fit, state.delta2
        k = fit.k
        self.stats.death_proposed += 1
        j = int(self.rng.integers(k))
        keep = np.arange(k) != j
        new = self._fit(fit.z[keep], fit.G[np.ix_(keep, keep)], fit.b[keep])
        if new is None:  # pragma: no cover - a sub-configuration of a valid one is valid
            return
        proposal = self._proposal_for(new, d2)
        log_q = proposal.log_density(complex(fit.z[j]))
        bk1, _ = self._birth_death_probs(k - 1)
        _, dk = self._birth_death_probs(k)
        log_alpha = (
            self._log_lik(new.Q, d2)
            - self._log_lik(fit.Q, d2)
            - self._log_penalty(d2)
            - self._log_prior_pole
            + log_q
            + self._log_prior_k[k - 1]
            - self._log_prior_k[k]
            + math.log(bk1)
            - math.log(dk)
        )
        if math.log(self.rng.random()) < log_alpha:
            state.fit = new
            self.stats.death_accepted += 1

    # ------------------------------------------------------------------ split / merge --

    def _split_merge_probs(self, k: int) -> tuple[float, float]:
        c, kmax = self.config.split_merge_scale, self.config.max_components
        lp = self._log_prior_k
        s = 0.0 if (k < 1 or k >= kmax) else c * min(1.0, math.exp(lp[k + 1] - lp[k]))
        m = 0.0 if k < 2 else c * min(1.0, math.exp(lp[k - 1] - lp[k]))
        return s, m

    @staticmethod
    def _nearest(z: ComplexArray, i: int) -> int:
        d = np.abs(z - z[i])
        d[i] = np.inf
        return int(np.argmin(d))

    def _mutual_count(self, z: ComplexArray, a: int, b: int) -> int:
        """Number of ways the merge move can select the pair ``{a, b}`` (0, 1 or 2)."""
        return int(self._nearest(z, a) == b) + int(self._nearest(z, b) == a)

    def _log_q_eps(self, eps: complex) -> float:
        tau = self.config.split_scale
        assert tau is not None
        return -math.log(2.0 * math.pi * tau * tau) - abs(eps) ** 2 / (2.0 * tau * tau)

    def _refit(self, keep: NDArray[np.bool_], fit: _Fit, new_poles: list[complex]) -> _Fit | None:
        z = np.concatenate([fit.z[keep], np.asarray(new_poles, dtype=np.complex128)])
        b = np.concatenate([fit.b[keep], [self.design.project_one(p) for p in new_poles]])
        return self._fit(z, self.design.gram(z), b)

    def _split(self, state: ChainState) -> None:
        """Split pole ``z`` into ``z ± ε`` with ``ε ~ CN(0, 2τ²)``.

        The Jacobian of ``(z, ε) → (z₁, z₂)`` is 4.
        """
        fit, d2 = state.fit, state.delta2
        k = fit.k
        self.stats.split_proposed += 1
        j = int(self.rng.integers(k))
        tau = self.config.split_scale
        assert tau is not None
        eps = complex(self.rng.normal(0.0, tau), self.rng.normal(0.0, tau))
        z1, z2 = complex(fit.z[j]) + eps, complex(fit.z[j]) - eps
        if not (self._in_support(z1) and self._in_support(z2)):
            return
        keep = np.arange(k) != j
        new = self._refit(keep, fit, [z1, z2])
        if new is None:
            return
        count = self._mutual_count(new.z, k - 1, k)
        if count == 0:
            return  # the merge move could never undo this split
        sk, _ = self._split_merge_probs(k)
        _, mk1 = self._split_merge_probs(k + 1)
        log_alpha = (
            self._log_lik(new.Q, d2)
            - self._log_lik(fit.Q, d2)
            + self._log_penalty(d2)
            + self._log_prior_pole
            + self._log_prior_k[k + 1]
            - self._log_prior_k[k]
            + math.log(mk1)
            - math.log(sk)
            + math.log(k * count)
            + math.log(4.0)  # Jacobian
            - math.log(2.0)  # ±ε give the same pair
            - self._log_q_eps(eps)
        )
        if math.log(self.rng.random()) < log_alpha:
            state.fit = new
            self.stats.split_accepted += 1

    def _merge(self, state: ChainState) -> None:
        """Merge a pole with its nearest neighbour into their midpoint (reverse of split)."""
        fit, d2 = state.fit, state.delta2
        k = fit.k
        self.stats.merge_proposed += 1
        i = int(self.rng.integers(k))
        j = self._nearest(fit.z, i)
        count = self._mutual_count(fit.z, i, j)
        zi, zj = complex(fit.z[i]), complex(fit.z[j])
        z_m, eps = 0.5 * (zi + zj), 0.5 * (zi - zj)
        if not self._in_support(z_m):
            return
        keep = (np.arange(k) != i) & (np.arange(k) != j)
        new = self._refit(keep, fit, [z_m])
        if new is None:
            return
        sk1, _ = self._split_merge_probs(k - 1)
        _, mk = self._split_merge_probs(k)
        log_alpha = (
            self._log_lik(new.Q, d2)
            - self._log_lik(fit.Q, d2)
            - self._log_penalty(d2)
            - self._log_prior_pole
            + self._log_prior_k[k - 1]
            - self._log_prior_k[k]
            + math.log(sk1)
            - math.log(mk)
            - math.log((k - 1) * count)
            - math.log(4.0)
            + math.log(2.0)
            + self._log_q_eps(eps)
        )
        if math.log(self.rng.random()) < log_alpha:
            state.fit = new
            self.stats.merge_accepted += 1

    # ------------------------------------------------------------------------- update --

    def _propose_pole(self, z: complex) -> tuple[complex, float]:
        """Random-walk proposal for one pole; returns the pole and the log prior/kernel ratio."""
        cfg = self.config
        # step-size multiplier chosen independently of the state: kernel stays symmetric
        scale = cfg.rw_scales[int(self.rng.integers(len(cfg.rw_scales)))]
        if cfg.update_kernel == "cartesian":
            assert cfg.rw_variance is not None
            sd = scale * math.sqrt(cfg.rw_variance)
            dz = complex(self.rng.normal(0.0, sd), self.rng.normal(0.0, sd))
            return z + dz, 0.0
        assert cfg.freq_step is not None
        assert cfg.damping_step is not None
        alpha = -math.log(abs(z))
        omega = math.atan2(z.imag, z.real)
        omega_new = omega + self.rng.normal(0.0, scale * cfg.freq_step)
        alpha_new = abs(alpha + self.rng.normal(0.0, scale * cfg.damping_step))  # reflect at 0
        # the uniform prior on the disk is proportional to e^{-2 alpha} in (omega, alpha)
        return complex(np.exp(complex(-alpha_new, omega_new))), -2.0 * (alpha_new - alpha)

    def _update(self, state: ChainState) -> None:
        d2 = state.delta2
        for idx in range(state.k):
            fit = state.fit
            self.stats.update_proposed += 1
            z_new, log_ratio = self._propose_pole(complex(fit.z[idx]))
            if not self._in_support(z_new):
                continue
            z = fit.z.copy()
            z[idx] = z_new
            col = self.design.gram_column(z, z_new)
            G = fit.G.copy()
            G[:, idx] = col
            G[idx, :] = col.conj()
            G[idx, idx] = col[idx].real
            b = fit.b.copy()
            b[idx] = self.design.project_one(z_new)
            new = self._fit(z, G, b)
            if new is None:
                continue
            log_alpha = self._log_lik(new.Q, d2) - self._log_lik(fit.Q, d2) + log_ratio
            if math.log(self.rng.random()) < log_alpha:
                state.fit = new
                self.stats.update_accepted += 1

    def _gibbs(self, state: ChainState) -> None:
        """Draw σ², a and (optionally) δ² from their full conditionals (eqs. 5, 6, 14)."""
        cfg, rng, fit = self.config, self.rng, state.fit
        if not (cfg.store_amplitudes or cfg.sample_delta2):
            return
        c = state.delta2 / (1.0 + state.delta2)
        ypy = max(self._yPy(fit.Q, state.delta2), np.finfo(float).tiny)
        state.sigma2 = float(0.5 * ypy / rng.gamma(self._log_exponent))
        k = fit.k
        if k:
            if self._real:
                w = _tri_solve(fit.L, fit.b.real)
                noise = rng.standard_normal(k)
            else:
                w = _tri_solve(fit.L, fit.b)
                noise = rng.standard_normal(k) + 1j * rng.standard_normal(k)
            rhs = c * w + math.sqrt(state.sigma2 * c) * noise
            state.amplitudes = np.asarray(
                _tri_solve(fit.L, rhs, conj_trans=True), dtype=np.complex128
            )
        else:
            state.amplitudes = np.zeros(0, dtype=np.complex128)
        if cfg.sample_delta2:
            a = state.amplitudes
            G = fit.G.real if self._real else fit.G
            aGa = float(np.vdot(a, G @ a).real) if k else 0.0
            rate = cfg.beta_delta + aGa / (2.0 * state.sigma2)
            state.delta2 = float(rate / rng.gamma(cfg.alpha_delta + self._kappa * k))

    # ----------------------------------------------------------------------------- run --

    def step(self, state: ChainState) -> str:
        """Advance the chain by one iteration and return the move type."""
        bk, dk = self._birth_death_probs(state.k)
        sk, mk = self._split_merge_probs(state.k)
        u = self.rng.random()
        if u < bk:
            move = "birth"
            self._birth(state)
        elif u < bk + dk:
            move = "death"
            self._death(state)
        elif u < bk + dk + sk:
            move = "split"
            self._split(state)
        elif u < bk + dk + sk + mk:
            move = "merge"
            self._merge(state)
        else:
            move = "update"
            self._update(state)
        self._gibbs(state)
        return move

    def run(
        self,
        initial_poles: ArrayLike | None = None,
        *,
        callback: Callable[[int, ChainState], None] | None = None,
        log_every: int = 0,
    ) -> RJMCMCResult:
        """Run burn-in plus sampling and return the posterior draws.

        Parameters
        ----------
        initial_poles
            Optional starting poles (e.g. from :func:`rjnmr.init.initial_poles`).
            The paper starts from a single random component; ``None`` starts
            from the empty model.
        callback
            Called as ``callback(iteration, state)`` after every iteration.
        log_every
            Emit an INFO log line every ``log_every`` iterations (0 disables).
        """
        cfg = self.config
        state = self.initial_state(initial_poles)
        total = cfg.burn_in + cfg.n_iterations
        trace = Trace.empty()
        k_chain = np.empty(total, dtype=np.int64)
        lp_chain = np.empty(total, dtype=np.float64)
        for it in range(total):
            self.step(state)
            lp = self.log_posterior(state)
            k_chain[it] = state.k
            lp_chain[it] = lp
            if it >= cfg.burn_in and (it - cfg.burn_in) % cfg.thin == 0:
                trace.append(state.fit.z, state.amplitudes, state.sigma2, state.delta2, lp, it)
            if callback is not None:
                callback(it, state)
            if log_every and (it + 1) % log_every == 0:
                logger.info(
                    "iteration %d/%d  k=%d  log-post=%.3f  delta2=%.3g",
                    it + 1,
                    total,
                    state.k,
                    lp,
                    state.delta2,
                )
        return RJMCMCResult(
            data=self.y,
            config=cfg,
            trace=trace.finalize(),
            k_chain=k_chain,
            log_posterior_chain=lp_chain,
            stats=self.stats,
            window=self.window,
        )


# ---------------------------------------------------------------------------- multi-chain ---


def _run_one(
    args: tuple[
        ComplexArray, SamplerConfig, np.random.SeedSequence, ComplexArray | None, FloatArray | None
    ],
) -> RJMCMCResult:
    y, config, seed, poles, window = args
    return RJMCMCSampler(y, config, seed=np.random.default_rng(seed), window=window).run(poles)


def run_chains(
    y: ArrayLike,
    config: SamplerConfig | None = None,
    *,
    n_chains: int = 4,
    seed: int | None = None,
    initial_poles: ArrayLike | None = None,
    window: ArrayLike | None = None,
    n_jobs: int | None = None,
) -> list[RJMCMCResult]:
    """Run independent chains (in parallel processes when ``n_jobs != 1``).

    Seeds are derived with :class:`numpy.random.SeedSequence` so results are
    reproducible for a given ``seed`` regardless of scheduling.
    """
    yc = np.asarray(y, dtype=np.complex128).ravel()
    cfg = config or SamplerConfig()
    poles = None if initial_poles is None else np.asarray(initial_poles, dtype=np.complex128)
    win = None if window is None else np.asarray(window, dtype=np.float64)
    seeds = np.random.SeedSequence(seed).spawn(n_chains)
    jobs = [(yc, cfg, s, poles, win) for s in seeds]
    if n_jobs == 1 or n_chains == 1:
        return [_run_one(j) for j in jobs]
    from concurrent.futures import ProcessPoolExecutor

    with ProcessPoolExecutor(max_workers=n_jobs) as pool:
        return list(pool.map(_run_one, jobs))
