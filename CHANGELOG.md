# Changelog

## Unreleased: 2010 toolbox port

- Windowed band-limited spectral likelihood (`rjnmr.design.WindowedSpectralDesign`)
  with exact closed-form columns; sampler accepts `window=`.
- Baseline offset component integrated out exactly (`baseline=True`), real (zero-phase)
  amplitude model, and a band-limited pole prior (`frequency_range`).
- Automatic zero/first-order phase correction (`rjnmr.phasing`), peak picking with
  Lorentzian estimates (`rjnmr.peaks`), raised-cosine bands and the band-wise
  `deconvolve` pipeline (`rjnmr.intervals`), component merging and amplitude refits
  (`rjnmr.postprocess`), `rjnmr deconvolve` CLI command and deconvolution plot.
- Relicensed from GPL-3.0-or-later to BSD-3-Clause, matching the 2010 toolbox.

## 0.1.0 — unreleased

First Python release; a rewrite of the original Matlab implementation.

- Complex-valued RJ-MCMC sampler of Rubtsov & Griffin (2007) with birth, death and
  update moves and Gibbs draws of σ², amplitudes and δ².
- Corrected birth/death acceptance ratio (no spurious `1/(k+1)` factor), verified
  by a prior-recovery test.
- Closed-form Gram matrix: O(N) work per pole, no `N × k` design matrix.
- Split/merge moves and a multi-scale random walk, so the chain no longer gets stuck
  with one line represented by two overlapping components.
- Uniform, data-spectrum and residual-spectrum birth proposals with exact densities;
  cartesian (paper) and polar random-walk updates; optional damping bound.
- MAP estimation, Hungarian-matching relabelling with per-component posterior
  probabilities, multi-chain pooling, `.npz` persistence.
- Matrix pencil with AIC/MDL order selection as baselines.
- JCAMP-DX (AFFN/ASDF) and raw Bruker readers; paper-style plots; `rjnmr` CLI.
