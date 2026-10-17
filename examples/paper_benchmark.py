"""Reproduce the synthetic Monte Carlo study of Section 4.1 (Tables 2-6) of the paper.

For each of the three noise levels, ``--runs`` noise realisations are analysed with
the RJ-MCMC sampler, and with the matrix pencil method using AIC order selection.
The script prints the detection rate of every component, as in Table 2.

    uv run python examples/paper_benchmark.py --runs 5 --iterations 20000
"""

from __future__ import annotations

import argparse

import numpy as np

from rjnmr import RJMCMCSampler, SamplerConfig, paper_components, simulate_fid
from rjnmr.baselines import matrix_pencil
from rjnmr.simulate import PAPER_NOISE_VARIANCES


def detected(truth: list, found: list, tol: float) -> np.ndarray:
    freqs = np.array([c.frequency for c in found])
    if freqs.size == 0:
        return np.zeros(len(truth), bool)
    return np.array([np.min(np.abs(freqs - c.frequency)) < tol for c in truth])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=1024)
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--iterations", type=int, default=40_000)
    ap.add_argument("--burn-in", type=int, default=10_000)
    args = ap.parse_args()

    truth = paper_components(args.n)
    tol = np.pi / args.n  # half a Fourier bin
    cfg = SamplerConfig(n_iterations=args.iterations, burn_in=args.burn_in)
    print(f"N={args.n}, {args.runs} runs per noise level, tolerance ±{tol:.4f} rad")
    for var in PAPER_NOISE_VARIANCES:
        bayes = np.zeros(len(truth))
        aic = np.zeros(len(truth))
        k_hat = []
        for run in range(args.runs):
            y = simulate_fid(truth, args.n, var, seed=1000 * int(var) + run)
            res = RJMCMCSampler(y, cfg, seed=run).run()
            k_hat.append(res.k_map)
            bayes += detected(truth, res.map_components(), tol)
            aic += detected(truth, matrix_pencil(y, criterion="aic"), tol)
        print(f"\nnoise variance {var:g}: k_MAP over runs = {k_hat}")
        print(f"{'comp':>4} {'SNR dB':>7} {'Bayesian':>9} {'MP+AIC':>7}")
        for i, c in enumerate(truth, 1):
            print(
                f"{i:>4} {c.snr_db(var):7.1f} {bayes[i - 1] / args.runs:9.2f} "
                f"{aic[i - 1] / args.runs:7.2f}"
            )


if __name__ == "__main__":
    main()
