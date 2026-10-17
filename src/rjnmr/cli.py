"""Command-line interface: ``rjnmr {fit,simulate,summary,plot,baseline}``."""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from . import __version__
from .baselines import estimate_order, matrix_pencil
from .config import SamplerConfig
from .init import initial_poles
from .intervals import auto_bands, deconvolve
from .io import FID, read_fid
from .phasing import auto_phase
from .preprocess import normalize, select_window, trim_to_maximum
from .results import RJMCMCResult
from .sampler import RJMCMCSampler, run_chains
from .simulate import paper_components, simulate_fid

__all__ = ["main"]

log = logging.getLogger("rjnmr")


def _load(args: argparse.Namespace) -> tuple[FID, dict[str, Any]]:
    fid = read_fid(args.input, byte_order=args.byte_order, dtype=args.dtype)
    y = fid.data
    info: dict[str, Any] = {"source": str(args.input), "original_length": int(y.size)}
    if args.phase:
        y, corr, _ = auto_phase(y)
        info["phase_correction"] = dataclasses.asdict(corr)
        log.info(
            "phase corrected: φ0=%.3f, group delay %.2f points", corr.phase0, corr.group_delay
        )
    start = args.start
    if args.trim_to_max:
        _, start = trim_to_maximum(y)
    y = select_window(y, start, args.length)
    info["start"] = int(start)
    if args.normalize:
        y, scaling = normalize(y)
        info["scale"] = scaling.scale
        info["offset"] = [scaling.offset.real, scaling.offset.imag]
    for key in ("spectral_width", "observe_frequency", "carrier_offset"):
        value = getattr(fid, key)
        if value is not None:
            info[key] = value
    fid.data = y
    return fid, info


def _add_data_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("input", type=Path, help="FID: JCAMP-DX (.jdx), Bruker dir/fid, or .npy")
    p.add_argument("--start", type=int, default=0, help="first data point to use")
    p.add_argument("--length", type=int, default=None, help="number of points to use")
    p.add_argument(
        "--trim-to-max",
        action="store_true",
        help="start at the point of maximal magnitude (removes Bruker group delay)",
    )
    p.add_argument(
        "--normalize", action="store_true", help="centre and scale to unit standard deviation"
    )
    p.add_argument(
        "--phase", action="store_true", help="automatic zero/first-order phase correction"
    )
    p.add_argument("--byte-order", choices=["big", "little"], default=None, help="raw Bruker")
    p.add_argument("--dtype", choices=["int32", "float64"], default=None, help="raw Bruker")


def _add_model_args(p: argparse.ArgumentParser, *, baseline_flag: bool = True) -> None:
    if baseline_flag:
        p.add_argument("--baseline", action="store_true", help="add a constant spectral offset")
    p.add_argument(
        "--real-amplitudes", action="store_true", help="zero-phase components (phased data)"
    )
    p.add_argument("--max-damping", type=float, default=None)
    p.add_argument("--update-kernel", choices=["cartesian", "polar"], default=None)


def _cmd_fit(args: argparse.Namespace) -> int:
    fid, info = _load(args)
    y = fid.data
    cfg = SamplerConfig(
        n_iterations=args.iterations,
        burn_in=args.burn_in,
        thin=args.thin,
        expected_components=args.expected_components,
        max_components=args.max_components,
        update_kernel=args.update_kernel or "cartesian",
        birth_proposal=args.birth_proposal,
        max_damping=args.max_damping,
        delta2=args.delta2,
        baseline=args.baseline,
        amplitude_model="real" if args.real_amplitudes else "complex",
        frequency_range=tuple(args.frequency_range) if args.frequency_range else None,
    )
    z0 = initial_poles(y, args.init, seed=args.seed, max_components=cfg.max_components)
    log.info("fitting %d points, %d initial components, %d chain(s)", y.size, z0.size, args.chains)
    t0 = time.perf_counter()
    if args.chains == 1:
        result = RJMCMCSampler(y, cfg, seed=args.seed).run(z0, log_every=args.log_every)
    else:
        chains = run_chains(
            y, cfg, n_chains=args.chains, seed=args.seed, initial_poles=z0, n_jobs=args.jobs
        )
        result = RJMCMCResult.concatenate(chains)
    info["runtime_s"] = time.perf_counter() - t0
    result.metadata.update(info)
    result.save(args.out)
    print(result.report(info.get("spectral_width")))
    print(f"\nsaved {args.out}  ({info['runtime_s']:.1f} s)")
    return 0


def _cmd_simulate(args: argparse.Namespace) -> int:
    comps = paper_components(args.n)
    y = simulate_fid(comps, args.n, args.noise_variance, seed=args.seed)
    np.save(args.out, y)
    print(f"saved {args.n}-point simulated FID (σ²={args.noise_variance}) to {args.out}")
    for i, c in enumerate(comps, 1):
        snr = c.snr_db(args.noise_variance) if args.noise_variance else float("inf")
        print(
            f"  {i}: freq={c.frequency:.5f} damping={c.damping:.4f} "
            f"A={c.amplitude:g} phase={c.phase:.3f}  SNR={snr:.1f} dB"
        )
    return 0


def _cmd_summary(args: argparse.Namespace) -> int:
    res = RJMCMCResult.load(args.result)
    sw = res.metadata.get("spectral_width") if args.hz else None
    if args.json:
        payload = {
            "k_map": res.k_map,
            "p_k": {int(k): float(p) for k, p in enumerate(res.k_posterior()) if p > 0},
            "components": [s.as_dict() for s in res.summarize()],
            "metadata": res.metadata,
        }
        json.dump(payload, sys.stdout, indent=2)
        print()
    else:
        print(res.report(sw, min_probability=args.min_probability))
    return 0


def _cmd_plot(args: argparse.Namespace) -> int:
    import matplotlib

    matplotlib.use("Agg")
    from .plotting import plot_diagnostics

    res = RJMCMCResult.load(args.result)
    freq_axis = None
    md = res.metadata
    if args.ppm and md.get("spectral_width") and md.get("observe_frequency"):
        sw, sfo1, o1 = md["spectral_width"], md["observe_frequency"], md.get("carrier_offset", 0.0)

        def freq_axis(w: np.ndarray) -> np.ndarray:
            return (w / (2 * np.pi) * sw + o1) / sfo1  # type: ignore[no-any-return]

    fig = plot_diagnostics(res, freq_axis=freq_axis, xlim=args.xlim)
    fig.savefig(args.out, dpi=args.dpi)
    print(f"saved {args.out}")
    return 0


def _cmd_baseline(args: argparse.Namespace) -> int:
    fid, _ = _load(args)
    order = args.order
    if order is None:
        est = estimate_order(fid.data, args.criterion, max_order=args.max_order)
        order = est.order
        print(f"{args.criterion.upper()} order estimate: {order}")
    comps = matrix_pencil(fid.data, order)
    print(f"{'#':>3} {'freq [rad]':>12} {'damping':>12} {'amplitude':>12} {'phase':>8}")
    for i, c in enumerate(comps, 1):
        print(f"{i:>3} {c.frequency:12.6f} {c.damping:12.6f} {c.amplitude:12.5g} {c.phase:8.3f}")
    return 0


def _cmd_deconvolve(args: argparse.Namespace) -> int:
    fid, info = _load(args)
    y = fid.data
    fid_for_axis = FID(y, fid.spectral_width, fid.observe_frequency, fid.carrier_offset)
    bands: list[tuple[float, float]] | None = None
    if args.bands:
        if len(args.bands) % 2:
            raise SystemExit("--bands needs an even number of edges (LO HI pairs)")
        pairs = list(zip(args.bands[::2], args.bands[1::2], strict=True))
        if args.ppm:
            conv = [
                tuple(sorted(float(fid_for_axis.ppm_to_frequency(e)) for e in p)) for p in pairs
            ]
            pairs = conv  # type: ignore[assignment]
        bands = [(float(lo), float(hi)) for lo, hi in pairs]
    else:
        bands = auto_bands(y, max_width=args.max_band_width)
    cfg = SamplerConfig(
        n_iterations=args.iterations,
        burn_in=args.burn_in,
        baseline=True,  # the 2010 pipeline always includes the flat baseline component
        amplitude_model="real" if args.real_amplitudes else "complex",
        max_damping=0.05 if args.max_damping is None else args.max_damping,
        update_kernel=args.update_kernel or "polar",
    )
    log.info("deconvolving %d points in %d bands", y.size, len(bands))
    t0 = time.perf_counter()
    result = deconvolve(
        y,
        bands,
        cfg,
        rolloff=args.rolloff,
        merge=not args.no_merge,
        refit=None if args.refit == "none" else args.refit,
        seed=args.seed,
        n_jobs=args.jobs,
    )
    info["runtime_s"] = time.perf_counter() - t0
    result.metadata.update(info)
    result.save_json(args.out)
    print(result.report(fid.spectral_width))
    print(f"\nsaved {args.out}  ({info['runtime_s']:.1f} s)")
    if args.plot:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from .plotting import plot_deconvolution

        freq_axis = None
        if args.ppm:

            def freq_axis(w: np.ndarray) -> np.ndarray:
                return np.asarray(fid_for_axis.frequency_to_ppm(w))

        fig, ax = plt.subplots(figsize=(12, 5), layout="constrained")
        lo = min(b[0] for b in bands) if bands else None
        hi = max(b[1] for b in bands) if bands else None
        xlim = None
        if lo is not None and hi is not None:
            xlim = (lo, hi) if freq_axis is None else tuple(freq_axis(np.array([lo, hi])))
        plot_deconvolution(result, ax=ax, freq_axis=freq_axis, xlim=xlim)
        fig.savefig(args.plot, dpi=150)
        print(f"saved {args.plot}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser."""
    parser = argparse.ArgumentParser(
        prog="rjnmr",
        description="Bayesian RJ-MCMC detection and estimation of damped sinusoids in NMR FIDs.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("-q", "--quiet", action="store_true", help="only print results")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("fit", help="run the RJ-MCMC sampler on an FID")
    _add_data_args(p)
    d = SamplerConfig()
    p.add_argument("-o", "--out", type=Path, default=Path("rjnmr_result.npz"))
    p.add_argument("-n", "--iterations", type=int, default=d.n_iterations)
    p.add_argument("-b", "--burn-in", type=int, default=d.burn_in)
    p.add_argument("--thin", type=int, default=d.thin)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--chains", type=int, default=1, help="independent chains to pool")
    p.add_argument("--jobs", type=int, default=None, help="parallel processes for chains")
    p.add_argument(
        "--init", choices=["empty", "random", "peaks", "matrix_pencil"], default="empty"
    )
    p.add_argument("--expected-components", type=float, default=d.expected_components)
    p.add_argument("--max-components", type=int, default=d.max_components)
    _add_model_args(p)
    p.add_argument(
        "--birth-proposal", choices=["uniform", "spectrum", "residual"], default=d.birth_proposal
    )
    p.add_argument("--delta2", type=float, default=None, help="fix δ² instead of sampling it")
    p.add_argument(
        "--frequency-range", type=float, nargs=2, metavar=("LO", "HI"), help="rad/sample"
    )
    p.add_argument("--log-every", type=int, default=1000)
    p.set_defaults(func=_cmd_fit)

    p = sub.add_parser("simulate", help="write the paper's 8-component test signal to .npy")
    p.add_argument("-o", "--out", type=Path, default=Path("simulated.npy"))
    p.add_argument("-n", type=int, default=1024, help="number of points")
    p.add_argument("--noise-variance", type=float, default=4.0)
    p.add_argument("--seed", type=int, default=None)
    p.set_defaults(func=_cmd_simulate)

    p = sub.add_parser("summary", help="summarise a saved result")
    p.add_argument("result", type=Path)
    p.add_argument("--json", action="store_true")
    p.add_argument("--hz", action="store_true", help="report frequencies in Hz if known")
    p.add_argument("--min-probability", type=float, default=0.0)
    p.set_defaults(func=_cmd_summary)

    p = sub.add_parser("plot", help="diagnostic figure for a saved result")
    p.add_argument("result", type=Path)
    p.add_argument("-o", "--out", type=Path, default=Path("rjnmr_result.png"))
    p.add_argument("--ppm", action="store_true", help="chemical-shift axis if parameters known")
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument(
        "--xlim", type=float, nargs=2, metavar=("LO", "HI"), help="spectral range to show"
    )
    p.set_defaults(func=_cmd_plot)

    p = sub.add_parser(
        "deconvolve", help="band-wise deconvolution of a full spectrum (2010 toolbox pipeline)"
    )
    _add_data_args(p)
    _add_model_args(p, baseline_flag=False)
    p.add_argument(
        "--bands",
        type=float,
        nargs="+",
        metavar="EDGE",
        help="band edges as LO HI pairs (rad/sample, or ppm with --ppm); default: automatic",
    )
    p.add_argument("--ppm", action="store_true", help="band edges and plot axis in ppm")
    p.add_argument("--max-band-width", type=float, default=0.2, help="rad/sample (auto bands)")
    p.add_argument("--rolloff", type=float, default=0.2)
    p.add_argument("-n", "--iterations", type=int, default=2000)
    p.add_argument("-b", "--burn-in", type=int, default=2000)
    p.add_argument("--refit", choices=["lstsq", "peak_weighted", "none"], default="lstsq")
    p.add_argument("--no-merge", action="store_true", help="keep near-duplicate components")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--jobs", type=int, default=1, help="parallel processes (bands)")
    p.add_argument("-o", "--out", type=Path, default=Path("deconvolution.json"))
    p.add_argument("--plot", type=Path, default=None, help="also save a figure")
    p.set_defaults(func=_cmd_deconvolve)

    p = sub.add_parser("baseline", help="matrix pencil estimate with AIC/MDL order selection")
    _add_data_args(p)
    p.add_argument("--order", type=int, default=None)
    p.add_argument("--criterion", choices=["mdl", "aic"], default="mdl")
    p.add_argument("--max-order", type=int, default=None)
    p.set_defaults(func=_cmd_baseline)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point of the ``rjnmr`` command."""
    args = build_parser().parse_args(argv)
    if not log.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(message)s", "%H:%M:%S"))
        log.addHandler(handler)
    log.setLevel(logging.WARNING if args.quiet else logging.INFO)
    result: int = args.func(args)
    return result


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
