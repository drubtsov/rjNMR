# rjNMR
## Bayesian NMR signal processing using Reversible Jump MCMC


The rjNMR toolbox contains Matlab files for Bayesian NMR signal processing described in [1].

The Bayesian deconvolution of the NMR signal is a parametric approach that seeks to estimate both a number of harmonics and their parameters from a complex-valued time-domain NMR signal. Its detailed description can be found in [2]. 

There is a number of auxiliary functions that are needed for pre- and postprocessing .
An example of a processing pipeline using different tools from the rjNMR toolbox is given in `rjNMR_script.m` .  Below the short description of main features is given.

### Phase correction 
Phase correction is formulated as an optimization procedure.  
In particular, a number of high-intensity peaks are identified using magnitude Fourier spectrum and approximate estimates for their frequency, decay rate, amplitude and phase are obtained.  The optimization procedure seeks to find phase correction terms that deliver a local minimum to a cost function.

### Initial peak detection. 
Choosing good starting values can significantly improve convergence of the sampling process and speed up the analysis. A simple peak-picking procedure is used to identify high-intensity peaks and obtain initial estimates of their parameters. These estimates are not very accurate and don't take into account possible effects of peak overlap or baseline distortions but they are easy to calculate and sufficient to provide a good starting point for the modelling. The more sophisticated methods such as linear prediction can be used to find better starting values. 

### Decomposition of the signal into a series of intervals. 
Estimation of the parameters in Bayesian models can be computationally intensive and time consuming due to the MCMC sampling process. One possible solution is to breakdown one global data model into a series of local models that can be processed more efficiently. In this case the frequency domain offers  better representation as decaying sinusoids with small damping factors are well localised in frequency. 

To achieve this, a series of bandwidth filters is applied to a frequency-domain signal. We use the raised-cosine filter so the interference of the peaks outside the filter band is smoothly weighed down to zero. 

The point estimates are then obtained using maximum a posteriory (MAP) criterion.  The estimated signal components that fell inside central non-downweighed regions of the filters are mapped back onto a global frequency axis. 

The amplitudes are then re-estimated on the global scale.

(c) Denis Rubtsov
Department of Biochemistry
Cambridge University


REFERENCES:

[1] Denis V. Rubtsov, Claire Waterman, Richard A. Currie, Catherine Waterfield, Jos Domingo Salazar, Jayne Wright and Julian L. Griffin , Application of a Bayesian Deconvolution Approach for High-Resolution 1H NMR Spectra to Assessing the Metabolic Effects of Acute Phenobarbital Exposure in Liver Tissue, Anal. Chem., 2010, 82 (11), pp 4479-4485

[2]  Denis V. Rubtsov and Julian L. Griffin,  Time-domain Bayesian detection and estimation of noisy
damped sinusoidal signals applied to NMR spectroscopy , J Magn Reson 2007, 188, pp 367-379.
