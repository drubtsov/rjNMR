# rjNMR

**Bayesian detection and estimation of damped sinusoids in NMR free induction decays,
using Reversible Jump MCMC.**

`rjnmr` fits the complex time-domain FID directly as a sum of exponentially damped
sinusoids:

$$y_t = \sum_{i=1}^{k} a_i\, z_i^{\,t} + e_t,\qquad z_i = e^{-\alpha_i + j\omega_i}.$$

It treats the **number of components `k` as unknown** and draws samples from the joint
posterior of `k`, the frequencies, dampings, amplitudes and phases. You get a full
posterior instead of a single least-squares fit, with no phase or baseline correction
and no manual peak picking.

It is a Python rewrite of the original Matlab implementation of

> **[1]** D. V. Rubtsov and J. L. Griffin, *Time-domain Bayesian detection and estimation of
> noisy damped sinusoidal signals applied to NMR spectroscopy*,
> J. Magn. Reson. **188** (2007) 367–379. [doi:10.1016/j.jmr.2007.08.008](https://doi.org/10.1016/j.jmr.2007.08.008)

together with the full-spectrum processing pipeline of the 2010 rjNMR toolbox:

> **[2]** D. V. Rubtsov, C. Waterman, R. A. Currie, C. Waterfield, J. D. Salazar, J. Wright and
> J. L. Griffin, *Application of a Bayesian deconvolution approach for high-resolution
> ¹H NMR spectra to assessing the metabolic effects of acute phenobarbital exposure in
> liver tissue*, Anal. Chem. **82** (2010) 4479–4485.

## Installation

The project is managed with [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/drubtsov/rjNMR.git
cd rjNMR
uv sync                 # creates .venv with the package and the dev tools
uv run rjnmr --help
```

To use it as a library in another uv project, run `uv add "rjnmr[plot] @ git+https://github.com/drubtsov/rjNMR"`.
The `plot` extra pulls in matplotlib.

## Quick start

```python
import rjnmr
from rjnmr import RJMCMCSampler, SamplerConfig, simulate_fid

# eight-component benchmark of the paper (Table 1), noise variance 4
y = simulate_fid(rjnmr.paper_components(1024), n=1024, noise_variance=4.0, seed=0)

result = RJMCMCSampler(y, SamplerConfig(n_iterations=40_000, burn_in=10_000), seed=1).run()

print(result.k_posterior())  # p(k | y)
print(result.report())  # MAP components + posterior probabilities
for c in result.summarize():  # label-switching-resolved summaries
    print(c.reference.frequency, c.probability, c.amplitude, c.amplitude_sd)

result.save("benchmark.npz")
```

Real data goes through the same steps. The paper drops the digital-filter group delay
and analyses 4096 points:

```python
from rjnmr.io import read_fid
from rjnmr.preprocess import trim_to_maximum, select_window, normalize
from rjnmr.init import initial_poles

fid = read_fid("experiment.jdx")  # JCAMP-DX, Bruker dir with acqus, or .npy
y, dropped = trim_to_maximum(fid.data)  # first point = maximal magnitude
y, scaling = normalize(select_window(y, 0, 4096))

cfg = SamplerConfig(n_iterations=50_000, burn_in=10_000, max_damping=0.5)
result = RJMCMCSampler(y, cfg, seed=0).run(initial_poles(y, "peaks"))
ppm = [fid.frequency_to_ppm(c.frequency) for c in result.map_components()]  # needs SFO1
```

### Full-spectrum pipeline (2010 toolbox)

For a whole high-resolution spectrum, the pipeline does the following:
1. phase-corrects the FID automatically;
2. splits the spectrum into raised-cosine bands;
3. fits each band with the sampler, using an exact band-limited likelihood with a
   baseline offset;
4. keeps the components inside each band's flat region;
5. merges near-duplicate lines and refits all amplitudes jointly.

```python
from rjnmr.io import read_fid
from rjnmr.phasing import auto_phase
from rjnmr.intervals import auto_bands, deconvolve

fid = read_fid("path/to/bruker/expno")  # experiment directory with fid + acqus
phased, correction, _ = auto_phase(fid.data)  # φ0 + first-order (group delay)
result = deconvolve(phased, auto_bands(phased), seed=0, n_jobs=4)
print(result.report())
```

Bands can also be given explicitly as `(low, high)` in rad/sample, or converted from ppm
with `fid.ppm_to_frequency`.

### Command line

```bash
rjnmr simulate -o sim.npy --noise-variance 16           # paper's synthetic signal
rjnmr fit sim.npy -n 40000 -b 10000 --seed 1 -o sim.npz # run the sampler
rjnmr fit path/to/expno --trim-to-max --length 4096 --normalize \
          --init peaks --max-damping 0.5 --chains 4 -o sample.npz
rjnmr summary sample.npz --hz                           # table (or --json)
rjnmr plot sample.npz --ppm -o sample.png               # 4-panel diagnostic figure
rjnmr baseline sim.npy --criterion mdl                  # matrix pencil + MDL/AIC
rjnmr deconvolve path/to/expno --phase \
          --jobs 4 --plot sample.png -o sample.json     # 2010 full-spectrum pipeline
```

## What is implemented

| Area | Module | Notes |
|---|---|---|
| Model | `rjnmr.model` | closed-form Gram matrix `DᴴD` and projections: O(N) per pole, so the `N × k` design matrix is never built during sampling |
| Sampler | `rjnmr.sampler` | birth / death / update moves, Gibbs draws of σ², amplitudes and δ² (eqs. 5, 6, 14); optional split / merge moves and multi-scale random walk (see theory notes); complex or real (zero-phase) amplitudes; band-limited pole prior; multiple chains in parallel processes |
| Likelihoods | `rjnmr.design` | the paper's time-domain likelihood, and the 2010 windowed frequency-band likelihood with closed-form columns; optional baseline offset integrated out exactly |
| Proposals | `rjnmr.proposals` | uniform-disk births (paper), plus births guided by the data or residual spectrum (from the Matlab code), with exact proposal densities |
| Post-processing | `rjnmr.results` | `p(k|y)`, MAP estimate (eq. 17), Hungarian-matching relabelling with per-component posterior probability, mean and SD; `.npz` persistence; chain pooling |
| Phasing | `rjnmr.phasing` | automatic zero- and first-order phase correction (`autoPhasing.m`) |
| Peaks | `rjnmr.peaks` | peak picking with three-point Lorentzian estimates of frequency, damping, amplitude and phase (`detectPeaks.m`) |
| Bands | `rjnmr.intervals` | raised-cosine windows, automatic band selection, band-wise `deconvolve` pipeline |
| Clean-up | `rjnmr.postprocess` | merging of near-duplicate components, joint least-squares or peak-weighted amplitude refit |
| Baselines | `rjnmr.baselines` | matrix pencil (Hua & Sarkar) with AIC / MDL order selection, the methods the paper compares against |
| I/O | `rjnmr.io` | JCAMP-DX `NTUPLES` FIDs (AFFN and compressed ASDF), JCAMP spectra, raw Bruker `fid` + `acqus` |
| Plots | `rjnmr.plotting` | figures of the papers: fit + residual, deconvolved components, `p(k|y)`, `k` trace, posterior draws, band-wise deconvolution |

The mathematics, and where this implementation deliberately departs from the paper,
are covered in [`docs/theory.md`](docs/theory.md). For instance, this implementation
corrects the birth/death acceptance ratio, and a test verifies the correction.
[`docs/matlab-port.md`](docs/matlab-port.md) maps every original Matlab function to its
Python counterpart. The original Matlab code is not part of this repository.

## Development

```bash
uv sync                                   # install everything
uv run pytest                             # full test suite (incl. slow statistical tests)
uv run pytest -m "not slow"               # fast subset
uv run ruff check . && uv run ruff format --check .
uv run mypy                               # strict typing
uv run pre-commit install                 # optional git hooks
```

The core correctness check is `tests/test_sampler.py::test_prior_recovery`. It runs the
sampler with a numerically flat likelihood and checks that the chain samples the prior,
`k ~ Poisson(Λ/(1+δ²))`. That validates the reversible-jump acceptance ratios for every
birth proposal.

## Citation

If you use this software, please cite [1], and [2] for the full-spectrum pipeline (see
[`CITATION.cff`](CITATION.cff)).

## Licence

BSD-3-Clause, the same terms as the 2010 rjNMR toolbox. See [LICENSE](LICENSE).
