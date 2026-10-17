"""Posterior draws, point estimates and post-processing of sampler output."""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import linear_sum_assignment

from .config import SamplerConfig
from .design import make_design
from .model import Component, components_from_poles, synthesize

__all__ = [
    "ComponentSummary",
    "MoveStats",
    "RJMCMCResult",
    "Sample",
    "Trace",
]

ComplexArray = NDArray[np.complex128]
FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(slots=True)
class MoveStats:
    """Counts of proposed and accepted moves."""

    birth_proposed: int = 0
    birth_accepted: int = 0
    death_proposed: int = 0
    death_accepted: int = 0
    update_proposed: int = 0
    update_accepted: int = 0
    split_proposed: int = 0
    split_accepted: int = 0
    merge_proposed: int = 0
    merge_accepted: int = 0

    @staticmethod
    def _rate(acc: int, prop: int) -> float:
        return acc / prop if prop else math.nan

    @property
    def birth_rate(self) -> float:
        """Acceptance rate of birth moves."""
        return self._rate(self.birth_accepted, self.birth_proposed)

    @property
    def death_rate(self) -> float:
        """Acceptance rate of death moves."""
        return self._rate(self.death_accepted, self.death_proposed)

    @property
    def split_rate(self) -> float:
        """Acceptance rate of split moves."""
        return self._rate(self.split_accepted, self.split_proposed)

    @property
    def merge_rate(self) -> float:
        """Acceptance rate of merge moves."""
        return self._rate(self.merge_accepted, self.merge_proposed)

    @property
    def update_rate(self) -> float:
        """Acceptance rate of single-pole random-walk updates."""
        return self._rate(self.update_accepted, self.update_proposed)

    def __add__(self, other: MoveStats) -> MoveStats:
        return MoveStats(
            *(a + b for a, b in zip(asdict(self).values(), asdict(other).values(), strict=True))
        )


@dataclass(frozen=True, slots=True)
class Sample:
    """A single posterior draw."""

    poles: ComplexArray
    amplitudes: ComplexArray
    sigma2: float
    delta2: float
    log_posterior: float
    iteration: int

    @property
    def k(self) -> int:
        """Number of components."""
        return int(self.poles.size)

    def components(self) -> list[Component]:
        """Components sorted by frequency."""
        return components_from_poles(self.poles, self.amplitudes if self.amplitudes.size else None)


@dataclass(slots=True)
class Trace:
    """Ragged storage of variable-dimension draws.

    Draw ``i`` owns ``poles[offsets[i]:offsets[i+1]]`` (and likewise for amplitudes).
    """

    k: IntArray
    offsets: IntArray
    poles: ComplexArray
    amplitudes: ComplexArray
    sigma2: FloatArray
    delta2: FloatArray
    log_posterior: FloatArray
    iteration: IntArray
    _buffer: list[tuple[ComplexArray, ComplexArray, float, float, float, int]] = field(
        default_factory=list, repr=False
    )

    @classmethod
    def empty(cls) -> Trace:
        """Create an empty trace for appending."""
        z = np.zeros(0, dtype=np.complex128)
        f = np.zeros(0, dtype=np.float64)
        i = np.zeros(0, dtype=np.int64)
        return cls(i, np.zeros(1, dtype=np.int64), z, z, f, f, f, i)

    def append(
        self,
        poles: ComplexArray,
        amplitudes: ComplexArray,
        sigma2: float,
        delta2: float,
        log_posterior: float,
        iteration: int,
    ) -> None:
        """Buffer one draw (call :meth:`finalize` when done)."""
        self._buffer.append(
            (poles.copy(), amplitudes.copy(), sigma2, delta2, log_posterior, iteration)
        )

    def finalize(self) -> Trace:
        """Convert buffered draws into contiguous arrays."""
        if not self._buffer:
            return self
        rows = self._buffer
        ks = np.array([r[0].size for r in rows], dtype=np.int64)
        amps = [r[1] if r[1].size == r[0].size else np.full(r[0].size, np.nan + 0j) for r in rows]
        new = Trace(
            k=np.concatenate([self.k, ks]),
            offsets=np.concatenate([self.offsets, self.offsets[-1] + np.cumsum(ks)]),
            poles=np.concatenate([self.poles, *[r[0] for r in rows]]),
            amplitudes=np.concatenate([self.amplitudes, *amps]),
            sigma2=np.concatenate([self.sigma2, [r[2] for r in rows]]),
            delta2=np.concatenate([self.delta2, [r[3] for r in rows]]),
            log_posterior=np.concatenate([self.log_posterior, [r[4] for r in rows]]),
            iteration=np.concatenate(
                [self.iteration, np.array([r[5] for r in rows], dtype=np.int64)]
            ),
        )
        return new

    def __len__(self) -> int:
        return int(self.k.size)

    def __getitem__(self, i: int) -> Sample:
        if i < 0:
            i += len(self)
        lo, hi = int(self.offsets[i]), int(self.offsets[i + 1])
        return Sample(
            poles=self.poles[lo:hi],
            amplitudes=self.amplitudes[lo:hi],
            sigma2=float(self.sigma2[i]),
            delta2=float(self.delta2[i]),
            log_posterior=float(self.log_posterior[i]),
            iteration=int(self.iteration[i]),
        )

    @staticmethod
    def concatenate(traces: Sequence[Trace]) -> Trace:
        """Pool several traces (e.g. from independent chains)."""
        out = Trace.empty()
        for t in traces:
            out = Trace(
                k=np.concatenate([out.k, t.k]),
                offsets=np.concatenate([out.offsets, out.offsets[-1] + t.offsets[1:]]),
                poles=np.concatenate([out.poles, t.poles]),
                amplitudes=np.concatenate([out.amplitudes, t.amplitudes]),
                sigma2=np.concatenate([out.sigma2, t.sigma2]),
                delta2=np.concatenate([out.delta2, t.delta2]),
                log_posterior=np.concatenate([out.log_posterior, t.log_posterior]),
                iteration=np.concatenate([out.iteration, t.iteration]),
            )
        return out


@dataclass(frozen=True, slots=True)
class ComponentSummary:
    """Posterior summary of one (relabelled) component.

    ``probability`` is the fraction of posterior draws that contain a pole
    matched to this component; the remaining fields are posterior means and
    standard deviations over those draws.
    """

    reference: Component
    probability: float
    frequency: float
    frequency_sd: float
    damping: float
    damping_sd: float
    amplitude: float
    amplitude_sd: float
    phase: float
    phase_sd: float

    def as_dict(self) -> dict[str, float]:
        """Flat dict representation (JSON friendly)."""
        d = {k: float(v) for k, v in asdict(self).items() if k != "reference"}
        d.update({f"map_{k}": float(v) for k, v in asdict(self.reference).items()})
        return d


def _circ_mean_sd(x: FloatArray) -> tuple[float, float]:
    c = np.mean(np.exp(1j * x))
    r = float(np.clip(abs(c), 1e-300, 1.0))
    return float(np.angle(c)), float(math.sqrt(max(-2.0 * math.log(r), 0.0)))


@dataclass(slots=True)
class RJMCMCResult:
    """Output of :meth:`rjnmr.sampler.RJMCMCSampler.run`.

    Attributes
    ----------
    data
        The analysed complex FID.
    config
        The (resolved) sampler configuration.
    trace
        Kept post-burn-in draws.
    k_chain, log_posterior_chain
        Model order and log posterior for *every* iteration including burn-in,
        for mixing diagnostics.
    stats
        Move acceptance counts.
    window
        Frequency-domain window of a band-limited (windowed spectral) fit, or
        ``None`` for the time-domain likelihood.
    metadata
        Free-form JSON-serialisable information (e.g. acquisition parameters
        such as ``spectral_width``, ``observe_frequency``, ``carrier_offset``).
    """

    data: ComplexArray
    config: SamplerConfig
    trace: Trace
    k_chain: IntArray
    log_posterior_chain: FloatArray
    stats: MoveStats = field(default_factory=MoveStats)
    metadata: dict[str, Any] = field(default_factory=dict)
    window: FloatArray | None = None

    # ------------------------------------------------------------------ model order --

    def k_posterior(self) -> FloatArray:
        """Posterior ``p(k | y)`` estimated from kept draws (index = k)."""
        if len(self.trace) == 0:
            raise ValueError("no posterior draws (n_iterations == 0)")
        counts = np.bincount(self.trace.k, minlength=int(self.trace.k.max()) + 1)
        out: FloatArray = counts / counts.sum()
        return out

    @property
    def k_map(self) -> int:
        """Most probable number of components."""
        return int(np.argmax(self.k_posterior()))

    # ---------------------------------------------------------------- point estimate --

    def map_index(self, k: int | None = None) -> int:
        """Index of the highest-posterior draw with ``k`` components (default: ``k_map``).

        This is the paper's MAP estimator, eq. (17).
        """
        k = self.k_map if k is None else k
        idx = np.flatnonzero(self.trace.k == k)
        if idx.size == 0:
            raise ValueError(f"no draws with k={k}")
        return int(idx[np.argmax(self.trace.log_posterior[idx])])

    def posterior_mean_amplitudes(self, poles: ComplexArray, delta2: float) -> ComplexArray:
        """Conditional posterior mean of the amplitudes, ``m = δ²/(1+δ²) G⁻¹ b``.

        ``G`` and ``b`` come from the same likelihood design (time-domain or
        windowed spectral, with or without baseline) as the sampler used.
        """
        if poles.size == 0:
            return np.zeros(0, dtype=np.complex128)
        design = make_design(self.data, window=self.window, baseline=self.config.baseline)
        G: NDArray[Any] = design.gram(poles)
        b: NDArray[Any] = design.project(poles)
        if self.config.amplitude_model == "real":
            G, b = G.real, b.real
        try:
            sol = cho_solve(cho_factor(G, lower=True), b)
        except np.linalg.LinAlgError:  # numerically singular: minimum-norm solution
            sol = np.linalg.lstsq(G, b, rcond=None)[0]
        m = delta2 / (1.0 + delta2) * sol
        return np.asarray(m, dtype=np.complex128)

    def map_sample(self, k: int | None = None) -> Sample:
        """The MAP draw (eq. 17)."""
        return self.trace[self.map_index(k)]

    def map_components(self, k: int | None = None, *, amplitudes: str = "mean") -> list[Component]:
        """Components of the MAP draw, sorted by frequency.

        Parameters
        ----------
        k
            Model order to condition on (default: ``k_map``).
        amplitudes
            ``"mean"`` uses the conditional posterior mean amplitudes given the
            MAP poles; ``"sample"`` uses the amplitudes drawn at that iteration.
        """
        s = self.map_sample(k)
        if amplitudes == "mean":
            a = self.posterior_mean_amplitudes(s.poles, s.delta2)
        elif amplitudes == "sample":
            a = s.amplitudes
        else:
            raise ValueError("amplitudes must be 'mean' or 'sample'")
        return components_from_poles(s.poles, a)

    def model_signal(self, components: Sequence[Component] | None = None) -> ComplexArray:
        """Noise-free FID of the MAP model (or of the given components)."""
        comps = self.map_components() if components is None else list(components)
        z = np.array([c.pole for c in comps], dtype=np.complex128)
        a = np.array([c.complex_amplitude for c in comps], dtype=np.complex128)
        return synthesize(z, a, self.data.size)

    def residual(self) -> ComplexArray:
        """Data minus MAP model."""
        out: ComplexArray = self.data - self.model_signal()
        return out

    # ------------------------------------------------------------------- relabelling --

    def summarize(
        self,
        reference: Sequence[Component] | None = None,
        *,
        gate: float | None = None,
    ) -> list[ComponentSummary]:
        """Relabel draws against reference components and summarise each component.

        Each draw's poles are matched one-to-one to the reference poles by
        minimum total distance |z - z_ref| (Hungarian algorithm); matches
        further than ``gate`` are discarded. This resolves label switching
        across draws of different dimension, in the spirit of the clustering
        post-processing described in Section 4.1 of the paper.

        Parameters
        ----------
        reference
            Reference components; defaults to :meth:`map_components`.
        gate
            Maximum pole distance for a match; defaults to one Fourier bin, ``2π/N``.
        """
        ref = self.map_components() if reference is None else list(reference)
        n = self.data.size
        gate = 2.0 * math.pi / n if gate is None else gate
        zr = np.array([c.pole for c in ref], dtype=np.complex128)
        nref, ndraw = zr.size, len(self.trace)
        hits: list[list[int]] = [[] for _ in range(nref)]  # global pole indices
        for i in range(ndraw):
            lo, hi = int(self.trace.offsets[i]), int(self.trace.offsets[i + 1])
            if hi == lo or nref == 0:
                continue
            cost = np.abs(zr[:, None] - self.trace.poles[None, lo:hi])
            rows, cols = linear_sum_assignment(cost)
            for r, c in zip(rows, cols, strict=True):
                if cost[r, c] <= gate:
                    hits[r].append(lo + int(c))
        out = []
        for r, comp in enumerate(ref):
            idx = np.asarray(hits[r], dtype=np.int64)
            if idx.size == 0:
                nan = math.nan
                out.append(ComponentSummary(comp, 0.0, nan, nan, nan, nan, nan, nan, nan, nan))
                continue
            z = self.trace.poles[idx]
            a = self.trace.amplitudes[idx]
            freq, damp = np.angle(z), -np.log(np.abs(z))
            f_mean, f_sd = _circ_mean_sd(freq)
            p_mean, p_sd = _circ_mean_sd(np.angle(a))
            out.append(
                ComponentSummary(
                    reference=comp,
                    probability=idx.size / ndraw,
                    frequency=f_mean,
                    frequency_sd=f_sd,
                    damping=float(damp.mean()),
                    damping_sd=float(damp.std()),
                    amplitude=float(np.abs(a).mean()),
                    amplitude_sd=float(np.abs(a).std()),
                    phase=p_mean,
                    phase_sd=p_sd,
                )
            )
        return out

    # ----------------------------------------------------------------- persistence --

    @staticmethod
    def concatenate(results: Sequence[RJMCMCResult]) -> RJMCMCResult:
        """Pool independent chains run on the same data."""
        if not results:
            raise ValueError("need at least one result")
        first = results[0]
        stats = MoveStats()
        for r in results:
            if r.data.shape != first.data.shape or not np.array_equal(r.data, first.data):
                raise ValueError("all results must be computed on the same data")
            same_window = (
                r.window is None
                if first.window is None
                else r.window is not None and np.array_equal(r.window, first.window)
            )
            if not same_window:
                raise ValueError("all results must use the same window")
            stats = stats + r.stats
        return RJMCMCResult(
            data=first.data,
            config=first.config,
            trace=Trace.concatenate([r.trace for r in results]),
            k_chain=np.concatenate([r.k_chain for r in results]),
            log_posterior_chain=np.concatenate([r.log_posterior_chain for r in results]),
            stats=stats,
            metadata=dict(first.metadata),
            window=first.window,
        )

    def save(self, path: str | Path) -> None:
        """Save to a compressed ``.npz`` file."""
        t = self.trace
        extra: dict[str, Any] = {} if self.window is None else {"window": self.window}
        np.savez_compressed(
            path,
            data=self.data,
            k=t.k,
            offsets=t.offsets,
            poles=t.poles,
            amplitudes=t.amplitudes,
            sigma2=t.sigma2,
            delta2=t.delta2,
            log_posterior=t.log_posterior,
            iteration=t.iteration,
            k_chain=self.k_chain,
            log_posterior_chain=self.log_posterior_chain,
            config=np.array(json.dumps(self.config.to_dict())),
            stats=np.array(json.dumps(asdict(self.stats))),
            metadata=np.array(json.dumps(self.metadata)),
            **extra,
        )

    @classmethod
    def load(cls, path: str | Path) -> RJMCMCResult:
        """Load a result written by :meth:`save`."""
        with np.load(path, allow_pickle=False) as f:
            trace = Trace(
                k=f["k"],
                offsets=f["offsets"],
                poles=f["poles"],
                amplitudes=f["amplitudes"],
                sigma2=f["sigma2"],
                delta2=f["delta2"],
                log_posterior=f["log_posterior"],
                iteration=f["iteration"],
            )
            return cls(
                data=f["data"],
                config=SamplerConfig.from_dict(json.loads(str(f["config"]))),
                trace=trace,
                k_chain=f["k_chain"],
                log_posterior_chain=f["log_posterior_chain"],
                stats=MoveStats(**json.loads(str(f["stats"]))),
                metadata=json.loads(str(f["metadata"])) if "metadata" in f else {},
                window=f.get("window"),
            )

    # ------------------------------------------------------------------- reporting --

    def report(self, spectral_width: float | None = None, *, min_probability: float = 0.0) -> str:
        """Human-readable summary table."""
        pk = self.k_posterior()
        top = np.argsort(pk)[::-1][:5]
        lines = [
            f"draws: {len(self.trace)}   k_MAP = {self.k_map}",
            "p(k|y): " + ", ".join(f"k={k}: {pk[k]:.3f}" for k in top if pk[k] > 0),
            (
                f"acceptance: birth {self.stats.birth_rate:.3f}, "
                f"death {self.stats.death_rate:.3f}, split {self.stats.split_rate:.3f}, "
                f"merge {self.stats.merge_rate:.3f}, update {self.stats.update_rate:.3f}"
            ),
            "",
        ]
        unit_f, unit_d = ("Hz", "Hz(FWHM)") if spectral_width else ("rad", "1/sample")
        header = (
            f"{'#':>3} {'P':>5} {'freq [' + unit_f + ']':>16} {'damp [' + unit_d + ']':>18} "
            f"{'amplitude':>12} {'phase':>8}"
        )
        lines.append(header)
        for i, s in enumerate(self.summarize(), start=1):
            if s.probability < min_probability:
                continue
            c = s.reference
            f = c.frequency_hz(spectral_width) if spectral_width else c.frequency
            d = c.linewidth_hz(spectral_width) if spectral_width else c.damping
            lines.append(
                f"{i:>3} {s.probability:5.2f} {f:16.6g} {d:18.6g} "
                f"{c.amplitude:12.5g} {c.phase:8.3f}"
            )
        return "\n".join(lines)
