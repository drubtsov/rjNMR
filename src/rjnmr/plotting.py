"""Matplotlib figures reproducing the plots of the paper (requires ``rjnmr[plot]``).

Every function draws into an optional ``ax`` and returns the axes, so plots
compose freely into larger figures.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from .model import synthesize
from .results import RJMCMCResult

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from .intervals import DeconvolutionResult

__all__ = [
    "plot_components",
    "plot_deconvolution",
    "plot_diagnostics",
    "plot_frequency_trace",
    "plot_k_posterior",
    "plot_k_trace",
    "plot_posterior_spectra",
    "plot_spectrum",
]

# Reference palette (validated categorical slots 1-2 and neutral inks)
DATA_COLOR = "#a3a29a"
MODEL_COLOR = "#2a78d6"
ACCENT_COLOR = "#eb6834"
INK = "#3d3d3a"
MUTED = "#73726c"

FrequencyAxis = Callable[[NDArray[np.float64]], NDArray[np.float64]] | None


def _plt() -> Any:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise ImportError("plotting requires matplotlib: pip install 'rjnmr[plot]'") from exc
    return plt


def _ax(ax: Axes | None) -> Axes:
    if ax is not None:
        return ax
    _, new = _plt().subplots(figsize=(8, 4))
    return new  # type: ignore[no-any-return]


def _style(ax: Axes) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK)
    ax.grid(True, color="#e6e5df", linewidth=0.6)
    ax.set_axisbelow(True)


def _spectrum(y: NDArray[np.complex128]) -> tuple[NDArray[np.float64], NDArray[np.complex128]]:
    n = y.size
    omega = np.fft.fftshift(2.0 * np.pi * np.fft.fftfreq(n))
    return omega, np.fft.fftshift(np.fft.fft(y))


def _xaxis(
    ax: Axes, omega: NDArray[np.float64], freq_axis: FrequencyAxis, label: str | None
) -> NDArray[np.float64]:
    if freq_axis is None:
        ax.set_xlabel(label or "frequency [rad/sample]")
        return omega
    ax.set_xlabel(label or "chemical shift [ppm]")
    ax.invert_xaxis()  # NMR convention: shift decreases to the right
    return freq_axis(omega)


def plot_spectrum(
    result: RJMCMCResult,
    *,
    ax: Axes | None = None,
    freq_axis: FrequencyAxis = None,
    xlabel: str | None = None,
    show_residual: bool = True,
) -> Axes:
    """Magnitude spectrum of the data, the MAP model and the residual (paper Fig. 6b)."""
    ax = _ax(ax)
    omega, Y = _spectrum(result.data)
    _, M = _spectrum(result.model_signal())
    x = _xaxis(ax, omega, freq_axis, xlabel)
    ax.plot(x, np.abs(Y), color=DATA_COLOR, lw=1.2, label="data")
    ax.plot(x, np.abs(M), color=MODEL_COLOR, lw=1.0, label=f"MAP model (k={result.k_map})")
    if show_residual:
        ax.plot(x, np.abs(Y - M), color=ACCENT_COLOR, lw=0.8, label="residual")
    ax.set_ylabel("magnitude")
    ax.legend(frameon=False, labelcolor=INK)
    _style(ax)
    return ax


def plot_components(
    result: RJMCMCResult,
    *,
    ax: Axes | None = None,
    freq_axis: FrequencyAxis = None,
    xlabel: str | None = None,
) -> Axes:
    """Deconvolved, phase-corrected Lorentzian components (paper Fig. 6c).

    Each component is drawn as the real part of the Fourier transform of
    ``|a| z^t`` (its phase removed), over the data magnitude spectrum.
    """
    ax = _ax(ax)
    omega, Y = _spectrum(result.data)
    x = _xaxis(ax, omega, freq_axis, xlabel)
    ax.plot(x, np.abs(Y), color=DATA_COLOR, lw=1.2, label="data (magnitude)")
    n = result.data.size
    for i, c in enumerate(result.map_components()):
        line = synthesize([c.pole], [c.amplitude], n)
        line[0] *= 0.5  # first-point correction for a pure absorption line shape
        _, L = _spectrum(line)
        ax.plot(x, L.real, color=MODEL_COLOR, lw=0.8, label="components" if i == 0 else None)
    ax.set_ylabel("intensity")
    ax.legend(frameon=False, labelcolor=INK)
    _style(ax)
    return ax


def plot_posterior_spectra(
    result: RJMCMCResult,
    *,
    n_samples: int = 100,
    ax: Axes | None = None,
    freq_axis: FrequencyAxis = None,
    xlabel: str | None = None,
    seed: int | None = 0,
) -> Axes:
    """Model magnitude spectra of random posterior draws with ``k = k_MAP`` (paper Fig. 4)."""
    ax = _ax(ax)
    n = result.data.size
    idx = np.flatnonzero(result.trace.k == result.k_map)
    rng = np.random.default_rng(seed)
    pick = rng.choice(idx, size=min(n_samples, idx.size), replace=False)
    omega, _ = _spectrum(result.data)
    x = _xaxis(ax, omega, freq_axis, xlabel)
    for j, i in enumerate(pick):
        s = result.trace[int(i)]
        _, S = _spectrum(synthesize(s.poles, s.amplitudes, n))
        ax.plot(
            x,
            np.abs(S),
            color=DATA_COLOR,
            lw=0.5,
            alpha=0.5,
            label="posterior draws" if j == 0 else None,
        )
    _, M = _spectrum(result.model_signal())
    ax.plot(x, np.abs(M), color=MODEL_COLOR, lw=1.2, label="MAP")
    ax.set_ylabel("magnitude")
    ax.legend(frameon=False, labelcolor=INK)
    _style(ax)
    return ax


def plot_k_posterior(result: RJMCMCResult, *, ax: Axes | None = None) -> Axes:
    """Bar chart of ``p(k|y)`` (paper Fig. 2)."""
    ax = _ax(ax)
    pk = result.k_posterior()
    ks = np.flatnonzero(pk > 0)
    ax.bar(ks, pk[ks], color=MODEL_COLOR, width=0.6)
    ax.set_xlabel("number of components k")
    ax.set_ylabel("p(k | y)")
    ax.set_xticks(ks)
    ax.set_xlim(float(ks.min()) - 2, float(ks.max()) + 2)  # narrow bars even for a single k
    _style(ax)
    ax.grid(False, axis="x")
    return ax


def plot_k_trace(result: RJMCMCResult, *, ax: Axes | None = None) -> Axes:
    """Model order against iteration, burn-in included (paper Fig. 3)."""
    ax = _ax(ax)
    ax.plot(
        np.arange(result.k_chain.size),
        result.k_chain,
        color=MODEL_COLOR,
        lw=1.0,
        drawstyle="steps-post",
    )
    if result.config.burn_in:
        ax.axvline(result.config.burn_in, color=MUTED, lw=1.0, ls="--")
        ax.annotate(
            "end of burn-in",
            (result.config.burn_in, ax.get_ylim()[1]),
            xytext=(4, -12),
            textcoords="offset points",
            color=MUTED,
            fontsize=8,
        )
    ax.set_xlabel("iteration")
    ax.set_ylabel("k")
    _style(ax)
    return ax


def plot_frequency_trace(result: RJMCMCResult, *, every: int = 1, ax: Axes | None = None) -> Axes:
    """Sampled pole frequencies against iteration (Matlab ``plotFreqvItn``)."""
    ax = _ax(ax)
    t = result.trace
    it = np.repeat(t.iteration, t.k)
    keep = np.isin(it, t.iteration[::every])
    ax.plot(it[keep], np.angle(t.poles[keep]), ".", color=MODEL_COLOR, ms=1.5)
    ax.set_xlabel("iteration")
    ax.set_ylabel("frequency [rad/sample]")
    _style(ax)
    return ax


def plot_diagnostics(
    result: RJMCMCResult,
    *,
    freq_axis: FrequencyAxis = None,
    xlim: tuple[float, float] | None = None,
) -> Figure:
    """Four-panel overview: fit, components, ``p(k|y)`` and the ``k`` trace.

    ``xlim`` restricts the two spectral panels to a frequency (or ppm) range.
    """
    plt = _plt()
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), layout="constrained")
    plot_spectrum(result, ax=axes[0, 0], freq_axis=freq_axis)
    plot_components(result, ax=axes[0, 1], freq_axis=freq_axis)
    plot_k_posterior(result, ax=axes[1, 0])
    plot_k_trace(result, ax=axes[1, 1])
    axes[0, 0].set_title("Fit", loc="left", color=INK)
    axes[0, 1].set_title("Deconvolved components", loc="left", color=INK)
    axes[1, 0].set_title("Posterior of the model order", loc="left", color=INK)
    axes[1, 1].set_title("Chain mixing", loc="left", color=INK)
    if xlim is not None:
        lo, hi = sorted(xlim)
        for ax in axes[0]:
            # ppm axes run right-to-left
            ax.set_xlim((hi, lo) if freq_axis is not None else (lo, hi))
    return fig  # type: ignore[no-any-return]


def plot_deconvolution(
    result: DeconvolutionResult,
    *,
    ax: Axes | None = None,
    freq_axis: FrequencyAxis = None,
    xlabel: str | None = None,
    xlim: tuple[float, float] | None = None,
    show_components: bool = True,
) -> Axes:
    """Absorption spectrum of a band-wise deconvolution (2010 demo figure).

    Shows the real part of the spectrum of the (phase-corrected) data, the
    model, the residual and, optionally, every component. Dotted lines mark
    the band edges.
    """
    ax = _ax(ax)
    omega, Y = _spectrum(result.data)
    _, M = _spectrum(result.model_signal())
    x = _xaxis(ax, omega, freq_axis, xlabel)
    ax.plot(x, Y.real, color=DATA_COLOR, lw=1.2, label="data")
    if show_components:
        n = result.data.size
        for i, c in enumerate(result.components):
            _, C = _spectrum(synthesize([c.pole], [c.complex_amplitude], n))
            ax.plot(
                x,
                C.real,
                color=MODEL_COLOR,
                lw=0.6,
                alpha=0.6,
                label="components" if i == 0 else None,
            )
    ax.plot(x, M.real, color=INK, lw=0.9, label=f"model ({len(result.components)} comp.)")
    ax.plot(x, (Y - M).real, color=ACCENT_COLOR, lw=0.7, label="residual")
    edges = sorted({e for band in result.bands for e in band})
    if edges:
        xe = np.array(edges) if freq_axis is None else freq_axis(np.array(edges))
        for e in xe:
            ax.axvline(float(e), color=MUTED, lw=0.6, ls=":")
    if xlim is not None:
        lo, hi = sorted(xlim)
        ax.set_xlim((hi, lo) if freq_axis is not None else (lo, hi))
    ax.set_ylabel("intensity (real part)")
    ax.legend(frameon=False, labelcolor=INK)
    _style(ax)
    return ax
