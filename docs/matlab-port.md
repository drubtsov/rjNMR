# Matlab → Python map

For users of the original Matlab code (the 2007 prototype that accompanied the
paper, and the 2010 rjNMR toolbox), these tables show where each piece now lives.
The Matlab sources themselves are not distributed with this package.

## 2007 prototype

| Matlab | Python | Notes |
|---|---|---|
| `nmrtest.m` (main script) | `rjnmr.sampler.RJMCMCSampler.run`, `rjnmr fit` CLI | session save/restore → `RJMCMCResult.save/load`; several chains → `run_chains` |
| `getD.m`, `getNewComp.m`, `getNewHarmonicComp.m`, `getNewDRComp.m` | `rjnmr.model.basis` | complex poles instead of separate cos/sin and decay matrices |
| `getNewDD.m`, `insertNewComp.m`, `removeComp.m`, `insertHarmonic.m`, `removeHarmonic.m`, `insertDR.m`, `removeDR.m` | `rjnmr.model.gram`, `gram_column`; Gram updates in `RJMCMCSampler._birth/_death/_update` | Gram matrix in closed form; `D` is never stored |
| `getYPY.m`, `probFreqAccept.m`, `probFreqPost.m`, `probNoiseCenterAccept.m` | `RJMCMCSampler._yPy`, `_log_lik`, `log_posterior` | complex-signal exponent `N + ν₀/2` |
| `probBirthDeath.m`, `probCompNumPrior.m` | `RJMCMCSampler._birth_death_probs` | deaths from `k = 1` allowed; prior ratio included in the acceptance |
| `drawSigmaPost.m`, `drawAmpPost.m`, `drawDeltaPost.m` | `RJMCMCSampler._gibbs` | Cholesky-based draws |
| `gengamma.m` | `numpy.random.Generator.gamma` | |
| `drawCenterPost.m` | – | DC offset: centre the data with `rjnmr.preprocess.normalize` |
| `drawFreqPrior.m`, `probFreqPrior.m`, `checkNewFreq.m` | `rjnmr.proposals.UniformDisk` | prior on poles in the unit disk |
| `drawDRProposal.m` | `update_kernel="polar"` | random walk on damping, reflected at 0; multi-scale steps via `rw_scales` |
| `fftindex.m`, `allotPointToInt.m`, `drawFreqFFTProp.m`, `probFreqFFTProp.m` | `rjnmr.proposals.SpectralDensity`, `PoleProposal` | `birth_proposal="spectrum"` / `"residual"`, with exact densities |
| `peakScan.m` | `rjnmr.init.peak_poles`, `initial_poles(..., "peaks")` | `ISCLEVER_INI` |
| `getdata.m` | `rjnmr.simulate.simulate_fid` | the paper's Table 1 signal: `paper_components` |
| `readFIDdata.m` | `rjnmr.io.read_jcamp_fid` | now reads the imaginary page too, plus compressed ASDF |
| `readFreqdata.m` | `rjnmr.io.read_jcamp_spectrum` | |
| – | `rjnmr.io.read_bruker_fid` | raw Bruker `fid` + `acqus` |
| `stat2arr.m` | `rjnmr.results.Trace` | ragged arrays with offsets |
| `plotSpectrum.m` | `rjnmr.plotting.plot_spectrum` | |
| `plotLorentz.m` | `rjnmr.plotting.plot_components` | |
| `plotFreqvItn.m` | `rjnmr.plotting.plot_frequency_trace` | |
| `plotFreqCompHist.m` | `RJMCMCResult.summarize` (per-component posterior probability) | |
| `playmovie.m` | `rjnmr.plotting.plot_posterior_spectra` | static overlay of posterior draws |

## 2010 toolbox (`rjNMR_Distribute`)

| Matlab | Python | Notes |
|---|---|---|
| `rjNMR_DEMO_script.m` | `rjnmr.intervals.deconvolve`, `rjnmr deconvolve` CLI | bands, per-band fits, flat-region filtering, merge and refit |
| `rjNMRMain.m`, `rwUpdate.m`, `getFFTAmps.m` | `RJMCMCSampler(..., window=...)` with `rjnmr.design.WindowedSpectralDesign` | exact band likelihood with amplitudes integrated out; global coordinates (no `length(iffy)/N` rescaling) |
| "flat" component `w=0, dr=-1e3` | `SamplerConfig(baseline=True)` | nuisance offset integrated out |
| `a_new(2:end) = abs(a_new(2:end))` | `SamplerConfig(amplitude_model="real")` | real (zero-phase) amplitudes |
| `probBirthDeath.m`, `poissonPDF.m` | `RJMCMCSampler._birth_death_probs` | |
| `autoPhasing.m`, `phCostFun.m` | `rjnmr.phasing.auto_phase`, `PhaseCorrection` | |
| `detectPeaks.m`, `getApproxPeakParams.m`, `getSecDer.m` | `rjnmr.peaks.detect_peaks`, `detect_spectrum_peaks`, `lorentzian_from_bins` | exact three-point Lorentzian fit |
| `rCosFilter.m` | `rjnmr.intervals.raised_cosine_window` | bands in rad/sample instead of bin indices |
| `compCleanup.m`, `mergeComps.m`, `recurrClust.m` | `rjnmr.postprocess.merge_close_components` | connected components via `scipy.sparse.csgraph` |
| `getPeakWeightAmps.m` | `rjnmr.postprocess.refit_amplitudes(..., method="peak_weighted")` | |
| `getDz.m` | `rjnmr.model.basis` | |
| `loadFidBV.m` | `rjnmr.io.read_bruker_fid` | Varian header skipping not ported |
| `plotComps.m` | `rjnmr.plotting.plot_deconvolution`, `plot_components` | |
| `drawAmpPost.m` (2010 copy) | `RJMCMCSampler._gibbs` | |
