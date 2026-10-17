# Theory and implementation notes

This page restates the model of Rubtsov & Griffin (2007) in the notation of the code
and documents every place where the implementation deliberately departs from the
paper.

## Model

For `t = 0, …, N−1`

$$y_t = \sum_{i=1}^{k} a_i z_i^t + e_t, \qquad z_i = e^{-\alpha_i + j\omega_i},\ |z_i| < 1,$$

where $e_t$ is circular complex white Gaussian noise with variance $\sigma^2$ in each of
the real and imaginary parts. In matrix form, $y = D a + e$ with $D_{ti} = z_i^t$.

* **Frequency** $\omega \in [-\pi, \pi)$ is in radians per sample. With spectral width
  `SW` (Hz) it corresponds to $f = \omega\,SW / 2\pi$ Hz relative to the carrier.
* **Damping** $\alpha \ge 0$ is in nepers per sample. The Lorentzian FWHM is $\alpha\,SW/\pi$ Hz.
* **Amplitude** $a = A e^{j\phi}$.

## Priors

| Quantity | Prior | Default |
|---|---|---|
| `k` | Poisson($\Lambda$) truncated to $k \le k_{max}$ | $\Lambda = 20$, $k_{max} = 200$ |
| poles $z_i$ | i.i.d. uniform on the unit disk, density $1/\pi$ (optionally uniform on the annulus $e^{-\alpha_{max}} < \lvert z\rvert < 1$) | whole disk |
| amplitudes | $a \mid \sigma^2 \sim \mathcal{CN}(0, \sigma^2 \delta^2 (D^HD)^{-1})$ (g-prior) | – |
| $\sigma^2$ | $IG(\nu_0/2, \gamma_0/2)$ | $\nu_0=\gamma_0=0$ (Jeffreys) |
| $\delta^2$ | $IG(\alpha_\delta, \beta_\delta)$ | $2, 10$ |

## Marginal posterior

Integrating out $a$ and $\sigma^2$ gives

$$
p(k, z \mid y, \delta^2) \propto
(\gamma_0 + y^H P y)^{-(N + \nu_0/2)}\,
(1+\delta^2)^{-k}\, \frac{\Lambda^k}{k!}\, \pi^{-k},
\qquad
y^H P y = y^H y - \tfrac{\delta^2}{1+\delta^2}\, b^H G^{-1} b,
$$

with $G = D^H D$ and $b = D^H y$. Both have closed forms because the columns of $D$
are geometric sequences:

$$G_{il} = \sum_{t=0}^{N-1} (\bar z_i z_l)^t = \frac{\mathrm{expm1}(N u_{il})}{\mathrm{expm1}(u_{il})},\quad u_{il} = \bar s_i + s_l,\ s = \log z,$$

and $b_i = \sum_t \bar z_i^{\,t} y_t$. The exponent $N + \nu_0/2$ (not $(N+\nu_0)/2$)
reflects the $2N$ real observations of a complex signal.

The sampler keeps $G$, $b$ and the Cholesky factor of $G$ in its state. Changing one
pole costs O(N) for the new column and projection plus O(k³) for the factorisation,
and no `N × k` matrix is formed. A proposal whose $G$ is numerically singular (two
virtually identical poles) is rejected. Specifically, a configuration is rejected when
any Cholesky pivot falls below $10^{-10}$ of the corresponding diagonal entry, i.e. when
a column keeps less than that fraction of its energy after the others are projected
out. That amounts to a negligible restriction of the prior.

## Moves

Each iteration performs one of the following moves:

* **birth** with probability $b_k = c \min(1, p(k+1)/p(k))$,
* **death** with probability $d_k = c \min(1, p(k-1)/p(k))$,
* **split** / **merge** with probabilities $s_k, m_k$ of the same form, with constant
  `split_merge_scale` (an extension; set it to 0 for the paper's sampler),
* **update** with the remaining probability: a random walk on every pole in turn.

It then draws $\sigma^2$, $a$ and (unless fixed) $\delta^2$ from their full conditionals
(paper eqs. 5, 6, 14). The default $c = 0.2$ follows the paper.

### Update kernels

* `cartesian` (paper, eq. 12): $z' = z + \epsilon$, with $\epsilon$ complex Gaussian
  with variance `rw_variance` (default $1/(3N)$) in each part. The proposal is
  symmetric, and proposals outside the prior support are rejected, so the acceptance
  ratio is just the likelihood ratio.
* `polar`: independent Gaussian steps on $\omega$ (wrapped) and $\alpha$ (reflected
  at 0). The uniform disk prior has density $\propto e^{-2\alpha}$ in these
  coordinates, so the ratio includes $e^{-2(\alpha'-\alpha)}$. This kernel is often
  more efficient for narrow NMR lines, whose damping is orders of magnitude smaller
  than the frequency resolution.

> **Departure from the paper: multi-scale steps.** The posterior width of a pole
> shrinks with its SNR. For a strong line it can be a hundred times smaller than the
> paper's step $\sqrt{1/(3N)}$. With a single step size, such poles are never
> refined; the chain then compensates by adding a second, overlapping component,
> which a death move cannot undo. Each proposal therefore multiplies the step by a
> factor drawn uniformly from `rw_scales` (default `(1, 0.1, 0.01, 0.001)`). The
> factor does not depend on the state, so the kernel remains symmetric and the
> acceptance ratio is unchanged. `rw_scales=(1.0,)` restores the paper's kernel.

### Birth and death

A birth appends a new pole $z^\*$ drawn from a proposal $q$. A death removes a
uniformly chosen pole. The acceptance probability for a birth $k \to k+1$ is

$$
\alpha_{birth} = \min\left\{1,\
\left(\frac{\gamma_0 + y^H P_k y}{\gamma_0 + y^H P_{k+1} y}\right)^{N+\nu_0/2}
\frac{1}{1+\delta^2}\,
\frac{p(k+1)\,d_{k+1}}{p(k)\,b_k}\,
\frac{p(z^\*)}{q(z^\*)}\right\},
$$

and the death acceptance is its reciprocal, with $q$ evaluated at the removed pole for
the state *after* removal.

> **Departure from the paper.** Eqs. (15)–(16) of the paper, like Andrieu & Doucet
> (1999), contain an extra factor $1/(k+1)$ in $\alpha_{birth}$ (and $k+1$ in
> $\alpha_{death}$). The prior is exchangeable in the component labels, and the
> death move picks one of the $k+1$ components uniformly. The $1/(k+1)$ from that
> choice is therefore cancelled by the $(k+1)$ ways the newborn component can be
> labelled. Keeping the factor makes the chain target
> $p(k) \propto \mu^k/(k!)^2$ instead of the Poisson prior, which biases model
> selection towards fewer components. `test_prior_recovery` runs the sampler with a
> numerically flat likelihood. It confirms that the corrected ratio reproduces
> $k \sim \text{Poisson}(\Lambda/(1+\delta^2))$, and that the uncorrected one would
> not.

### Birth proposals

* `uniform`: $q = 1/\pi$ on the disk, the paper's choice. With a flat proposal, new
  components are rarely accepted on real data.
* `spectrum`: frequency from the magnitude spectrum of the data, damping from an
  exponential distribution with mean `damping_scale` (default $10/N$).
* `residual` (default): as `spectrum`, but the spectrum is that of the residual
  $y - D m$ at the posterior-mean amplitudes $m$ of the current state. This is the
  approach of the original Matlab code. Because the residual is a deterministic
  function of the state, the reverse density for a death is evaluated on the residual
  of the reduced state, so the move stays exactly reversible.

The two spectral proposals are mixed with the uniform one (weight `uniform_mixture`,
default 0.1). The mixture keeps the support equal to the whole disk, which
reversibility needs. Densities are with respect to area on the disk. A density in
$(\omega, \alpha)$ is converted with the Jacobian $|dz|^2 = e^{-2\alpha}\,d\omega\,d\alpha$.

### Split and merge

The paper (Section 5) suggests split/merge moves as an extension. A split chooses a
pole $z$ uniformly and replaces it by $z_{1,2} = z \pm \varepsilon$, with
$\varepsilon$ complex Gaussian with standard deviation `split_scale` (default $2/N$)
per part. A merge chooses a pole uniformly and merges it with its nearest
neighbour into the midpoint. The map $(z, \varepsilon) \mapsto (z_1, z_2)$ has
Jacobian 4. A split is admissible only if $z_1, z_2$ are nearest neighbours in at
least one direction ($c \in \{1, 2\}$ ways for the merge to pick the pair), and
$\pm\varepsilon$ give the same pair. Hence

$$
\alpha_{split} = \min\left\{1,\ L_r\,\frac{1}{1+\delta^2}\,
\frac{p(k+1)\,m_{k+1}}{p(k)\,s_k}\, p(z)\, \frac{4\,k\,c}{2\,q(\varepsilon)}\right\},
$$

and $\alpha_{merge}$ is its reciprocal. The prior-recovery test also covers these moves.

## Band-limited fits and other 2010 extensions

### Windowed spectral likelihood

The 2010 toolbox fits a full spectrum band by band. For a window $W_k \ge 0$ over the
DFT bins (a raised cosine, `rjnmr.intervals.raised_cosine_window`), the data are the
windowed bins $\tilde y_k = W_k Y_k / \sqrt N$ on the window's support. The model
columns are the windowed DFTs of the geometric sequences, which have the closed form

$$\tilde d_{k}(z) = \frac{W_k}{\sqrt N} \sum_{t=0}^{N-1} z^t e^{-2\pi i k t/N}
= \frac{W_k}{\sqrt N} \frac{\mathrm{expm1}(N u_k)}{\mathrm{expm1}(u_k)}, \qquad u_k = \log z - 2\pi i k/N .$$

`WindowedSpectralDesign` uses $\tilde y$ and $\tilde D$ in place of $y$ and $D$, with
$N$ replaced by the number $M$ of bins in the support. Everything else (marginal
likelihood, moves, Gibbs steps) is unchanged. With $W \equiv 1$ this reproduces the
time-domain likelihood exactly (Parseval), which a test checks.

There are three differences from the Matlab implementation (`rjNMRMain.m`), which
fitted the decimated inverse FFT of the band:
* the poles and amplitudes stay in global coordinates, so there is no rescaling
  step;
* the columns are exact rather than computed by FFT;
* the amplitudes are integrated out instead of being plugged in from least squares.

Inside the flat part of the window the likelihood is the exact Gaussian likelihood
of the band. In the tapers it is the same down-weighted residual as in the Matlab
code, which suppresses the tails of lines outside the band.

Unless `frequency_range` overrides it, the pole prior becomes uniform on the annular
sector $\{e^{-\alpha_{max}} < |z| < 1,\ \arg z \in \text{band}\}$. Its density is
$1/\text{area}$, which replaces $1/\pi$ in the acceptance ratios. The uniform part of
the birth proposal is restricted to the same sector, and the spectral part uses the
residual on the band's bins.

### Baseline offset

The Matlab sampler always carried a fixed "flat" component (`w = 0, dr = -1e3`, i.e.
a spike at $t = 0$, a constant spectral offset). Here it is a nuisance column $v$ with
a flat prior on its amplitude. Integrating that amplitude out projects $v$ out of the
data and the columns, and removes one complex observation:
$G \to G - (D^Hv)(v^HD)$, $b \to b - (D^Hv)(v^Hy)$, $y^Hy \to y^Hy - |v^Hy|^2$,
$N \to N - 1$. In the time domain $v = e_0$, so this is "subtract 1 from every Gram
entry and $y_0$ from $b$".

### Real amplitudes

For phase-corrected spectra, every line is pure absorption, so its amplitude is real.
The Matlab code imposes this after the fact with `abs(a)`. With
`amplitude_model="real"`, the amplitudes are real Gaussian under the g-prior
$a \sim N(0, \sigma^2\delta^2 (\operatorname{Re} G)^{-1})$, and the marginalisation
goes through with $\operatorname{Re} G$ and $\operatorname{Re} b$. The per-component
penalty becomes $(1+\delta^2)^{-1/2}$, and the $\delta^2$ full conditional becomes
$IG(\alpha_\delta + k/2, \beta_\delta + a^T \operatorname{Re}G\, a/2\sigma^2)$. The
flat-likelihood test checks
$k \sim \text{Poisson}(\Lambda/\sqrt{1+\delta^2})$.

The Occam penalty per component is therefore half as strong, while each pole still
has two free parameters. Broad, low-amplitude components become cheap, so the real
model should be combined with a damping bound (`max_damping`), as the pipeline does.

### Resolvability prior

Nearly coincident poles with opposite-sign amplitudes can model lineshape distortions,
for instance from imperfect shimming. Their columns are almost collinear, so the
amplitudes become enormous and cancel each other. They fit well but say nothing about
the spectrum. `min_independence` rejects any configuration in which a column keeps
less than that fraction of its energy after the others are projected out (a relative
Cholesky pivot of $G$). In effect, the prior is restricted to resolvable
configurations. The band pipeline uses $10^{-3}$. On a 600 MHz ¹H Bruker FID (the 2010 toolbox's demo data) this
reduces the largest fitted amplitude from $1.3\cdot10^6$ to $8\cdot10^4$ and does not
degrade the fit.

### Pre- and post-processing

* **Phasing** (`rjnmr.phasing.auto_phase`, from `autoPhasing.m`): the first-order slope
  is the median of blockwise fits to the unwrapped phase. The zero-order term comes
  from the best peak phase. Both are then refined, together with the pivot, by
  minimising the squared negative absorption in the peak neighbourhoods. On the
  2010 toolbox's demo FID, the phased spectrum matches the Matlab demo output with
  correlation > 0.999. The first-order term equals the Bruker DSP group delay
  (≈ 72 points).
* **Peaks** (`rjnmr.peaks`, from `detectPeaks.m`): maxima are found on a
  triangle-smoothed spectrum and bounded by inflection points. Each peak gets an exact
  three-point fit of $1/|Y|^2$, which is a parabola in $\omega$ for a Lorentzian.
* **Benchmark against the Matlab demo.** On the demo FID and the demo's five bands,
  the pipeline leaves an absorption residual of 5.0 / 5.6 / 6.3 / 1.0 / 8.9 % per band.
  The saved Matlab demo result leaves 16 / 13 / 18 / 4.1 / 23 %, and the
  overall figures are 1.7 % against 5.4 %. The line positions agree to within one or
  two DFT bins.
* **Clean-up** (`rjnmr.postprocess`, from `compCleanup.m`/`mergeComps.m`): components
  closer than 0.0009 rad in frequency and 0.004 in damping are merged. The merged pole
  is the $|a|^2$-weighted mean and the amplitude is the sum. Amplitudes are then refit
  jointly, by least squares or by the peak-weighted absorption regression of
  `getPeakWeightAmps.m`.

## Point estimates and label switching

* `RJMCMCResult.k_posterior()` returns the frequency of each `k` among kept draws.
* `RJMCMCResult.map_components()` returns the draw with the highest marginal posterior
  among those with $k = \hat k$ (paper eq. 17). Amplitudes are, by default, the
  conditional posterior means $m = \frac{\delta^2}{1+\delta^2} G^{-1} b$ for those
  poles.
* `RJMCMCResult.summarize()` relabels every draw against reference components (the
  MAP estimate by default). It solves a minimum-cost assignment on $|z - z_{ref}|$ and
  gates matches at one Fourier bin ($2\pi/N$). The result is the posterior probability
  that each component is present, plus posterior means and SDs of its parameters. This
  plays the role of the clustering post-processing described in Section 4.1 of the
  paper.

## Differences from the original Matlab code

The 2007 Matlab prototype does not implement the paper
exactly:

* It models a *real* signal $\sum (a^c \cos\omega t + a^s \sin \omega t)e^{-\alpha t}$
  plus a Gaussian DC offset. The paper and this package model the complex FID, which
  is what quadrature detection measures.
* It updates frequency and damping separately; the decay rate has a uniform `[0, 0.5]`
  birth proposal.
* Its birth/death ratios carry the extra `1/(k+1)` factor discussed above, and they
  build the residual spectrum from the amplitudes drawn at the previous iteration,
  which are not a function of the current state. So the moves are not exactly
  reversible.
* `probBirthDeath` forbids deaths from `k = 1`, so the chain can never reach `k = 0`.

The 2010 toolbox differs in its turn:

* It is complex-valued and band-limited (see above).
* It plugs in least-squares amplitudes (with `abs`) rather than integrating them out.
* It uses `1/(k+1)` birth and `k` death factors with no δ² penalty.
* Its random walk clips the damping at zero.
* It tracks the best state seen rather than sampling a posterior.

The Python version keeps the 2010 pipeline but runs it on the exact marginal posterior
above.

[`matlab-port.md`](matlab-port.md) lists how each Matlab function maps to the Python
code.
