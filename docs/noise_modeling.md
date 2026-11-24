# ARNoiseWhitener: Fast, Matrix-Free AR(q) Noise Modeling

`ARNoiseWhitener` is a high-performance, immutable data class designed to model stationary colored noise using an Autoregressive $\text{AR}(q)$ process and apply calibrated, matrix-free whitening to time-series pulses. It is optimized for microcalorimeter pulse records where the pulse length $p$ is much larger than the noise correlation length (e.g., $p = 8000$, $q \le 256$).

## Core Behavior

1. **Pooled Burg Estimation**: Fits an $\text{AR}(q)$ model by pooling reflection coefficients across multiple mean-subtracted training sequences.
2. **Automatic Order Selection**: Evaluates all orders up to `max_order` in a single pass and automatically truncates $q$ at the noise floor, ensuring the model does not overfit random sample fluctuations.
3. **Cross-Validation & Calibration**: Automatically performs K-fold cross-validation (if $N_{\text{seq}} \ge 2$) to calibrate the global variance scale ($\alpha$) and verify out-of-sample whitened residual autocorrelation.

## Matrix-Free Whitening

In standard Generalized Least Squares (GLS), whitening a pulse of length $p$ requires applying a dense $p \times p$ inverse Cholesky matrix $L^{-1}$, requiring $O(p^2)$ operations and massive memory overhead.

`ARNoiseWhitener` completely avoids dense $p \times p$ matrices. It exploits the banded Toeplitz structure of the $\text{AR}(q)$ covariance:
* **For the first $q$ samples:** It applies exact Levinson-Durbin step-down startup filters.
* **For all subsequent samples:** It applies a fast causal FIR filter (`scipy.signal.lfilter`).

This reduces the whitening complexity from $O(p^2)$ to **$O(pq)$**, dropping the memory footprint from hundreds of megabytes to a few kilobytes per model.

---

## ⚠️ Key Feature: `project_offset` (Constant Baseline Rejection)

By default, `ARNoiseWhitener` is instantiated with `project_offset=True`. This is a critical feature that mathematically marginalizes out unknown, drifting constant baselines.

### How it works
The whitener implicitly applies a prior of **infinite variance at DC**. It precomputes the whitened constant vector $\hat{w} = L^{-1}\mathbf{1}_p / \|L^{-1}\mathbf{1}_p\|$ and projects it out of the whitened data using a rank-1 update:

$$ W y = \frac{1}{\sqrt{\alpha}} \left(I - \hat{w}\hat{w}^T\right) L^{-1} y $$

### Practical Implications
1. **Absolute DC Invariance:** Whitening $y$ and whitening $y + 10^6$ will yield the exact same result (up to floating-point roundoff). You do not need to explicitly baseline-subtract individual test pulses before whitening.
2. **GLS Design Matrix Warning:** **Do not include a constant column of ones in your design matrix $X$.** Because $W \mathbf{1}_p = \mathbf{0}$, adding a constant column will make your whitened design matrix $WX$ rank-deficient, and the Fisher Information Matrix `WX.T @ WX` will become singular (non-invertible).
3. **Disabling the Feature:** If you require DC sensitivity, instantiate with `project_offset=False`.

---

## Capabilities and Methods

### Pulse Whitening and GLS
* **`whiten(y)`**: Whitens records of length $p$ using the exact $L^{-1}$ Cholesky factor and offset projection.
* **`fit_linear_model(y, X)`**: Computes Generalized Least Squares (GLS) parameter estimates $\hat{\beta}$ and the calibrated parameter covariance matrix. Yields a noise-weighted optimal parameter vector with $X\hat{\beta}\approx y$.
* **`predict_parameter_cov(X)`**: Predicts the calibrated error covariance $\widehat{\text{Cov}}(\hat{\beta}) = (W_X^T W_X)^{-1}$ directly from the design matrix.

### Continuous Stream Whitening
* **`whiten_long(y, mode="toeplitz")`**: Whitens arbitrary-length arrays ($M \ge p$).
    * `mode="toeplitz"` applies a lower-triangular Toeplitz causal FIR filter, anchoring to $y[0]$ to avoid step-transients.
    * `mode="exact_ar"` applies the exact finite-window startup rows before transitioning to the FIR filter.

### Diagnostics and Storage
* **`digest` attribute**: Contains full PSDs (parametric and Welch), autocorrelations, and scalar metrics like spectral dynamic range and innovation standard deviation.
* **`plot_digest()`**: Renders a 4-panel `matplotlib` figure for immediate visual validation of the fitted noise spectrum, ACF, order cutoff, and out-of-sample whiteness. See next section for details.
* **Parquet / Arrow IPC**: The model supports atomic single-row (`save`, `load`) and multi-row (`save_many`, `load_all`) serialization to disk, perfectly preserving the `digest` and operator state for later analysis.

## Diagnostic plots for noise models

`ARNoiseWhitener.plot_digest()` generates a 2×2 diagnostic figure summarizing the fitted $\text{AR}(q)$ model, its spectral and correlation accuracy, the automatic order cutoff, and out-of-sample whitening performance.

---

### Figure Header Summary

The banner at the top reports four scalar checks:

* **`AR(q)` and `p`:** The automatically chosen model order $q$ and target pulse window length $p$.


* **`RMS` ($\sqrt{\gamma[0]}$):** Total calibrated root-mean-square noise amplitude at the training sequence timescale.


* **`Innov STD` ($\sqrt{\alpha E_q}$):** Irreducible one-step white-noise standard deviation remaining after full $\text{AR}(q)$ whitening. Comparing `RMS` to `Innov STD` tells you how much total noise energy is correlated (removable by whitening) versus irreducible white floor.


* **`DynRange`:** Ratio in dB between the maximum and minimum bins of the one-sided $\text{AR}(q)$ PSD. (Note: Because the one-sided convention doubles interior bins while leaving DC and Nyquist un-doubled, even perfectly flat white noise has a baseline dynamic range of $10\log_{10}(2) \approx 3.01\text{ dB}$.)

---

### Panel 1 (Top-Left): One-Sided Power Spectral Density

**What is plotted:** The parametric `AR(q) Model PSD` (blue) overlaid on the empirical `Welch PSD (direct)` (gray, averaged across mean-subtracted training sequences with Hann windows of length $\min(N_{\text{seq}}, p)$ and 50% overlap), with a vertical red dashed line marking the pulse-window frequency $f_s / p$.

**Healthy fit:** The smooth blue $\text{AR}(q)$ curve tracks down the center of the noisy gray Welch trace across all decades—capturing low-frequency roll-offs, resonance peaks, and narrowband notches without chasing random periodogram fluctuations.

**Warning signs:**

* *Model misses sharp peaks or rounds off narrow notches:* $q$ is too small (or hit `max_order`), leaving insufficient poles to resolve steep spectral features.

* *Spiky ripples in the blue curve not present in the Welch trace:* Overfitting from too few total training samples ($N_{\text{seq}} \cdot L_{\text{seq}}$) relative to `max_order`.

* *Roll-off below $f_s / p$:* Expected artifact of per-sequence de-meaning; `project_offset=True` ensures that low-frequency mismatch does not bias length-$p$ pulse analyses.

---

### Panel 2 (Top-Right): Noise Autocorrelation (Lags 0 to $p - 1$)

**What is plotted:** The calibrated model autocorrelation $\rho[k] = \gamma[k] / \gamma[0]$ out to `max_acf_lag` (default $p - 1$), with a vertical orange dashed line at `Order cutoff q` whenever $q \lt p$. Lags $0 \dots q$ come directly from the fitted Burg reflection coefficients, while lags $k \gt q$ are extrapolated via the Yule–Walker recurrence.

**Healthy fit:** Starts at $\rho[0] = 1$ and decays smoothly (or with damped oscillations matching resonance frequencies) toward zero, transitioning seamlessly across the vertical `q` line with no kink or sudden change in decay rate.

* **Warning signs:**

* *$\rho[k]$ stays near $1$ across the entire window:* Training sequences are dominated by slow baseline drift over `sequence_length`.
* *Persistent undamped ringing past lag $q$:* A pole sits very close to the unit circle (such as an un-removed narrowband mains line).


---

### Panel 3 (Bottom-Left): Burg Reflection Coefficient Selection

**What is plotted:** The magnitude of the pooled reflection coefficients $\vert{}\kappa_m\vert{}$ across orders $m = 1 \dots \text{max\_order}$ (green, log scale), compared against the white-noise standard deviation floor $1/\sqrt{N_{\text{eff}}}$ (gray dotted line) and the $3.6\sigma$ spike threshold (red dashed line), with a vertical dash-dot orange line at `Selected q`.

**Healthy fit:** $\vert{}\kappa_m\vert{}$ starts well above the red line at low orders (capturing broadband slope and low-pass poles), decays downward—possibly with periodic lobes from narrowband notches—and settles into a flat band hovering around $1/\sqrt{N_{\text{eff}}}$, with `Selected q` placed right where the curve enters that noise floor.

**Warning signs:**

* *`Selected q` pegged at `max_order` (far right edge):* $\vert{}\kappa_m\vert{}$ has not yet decayed into the noise floor by $m = 256$, indicating either ultra-sharp spectral notches (ring-down longer than 256 samples) or transient glitches/outliers in the training sequences.

* *`Selected q = 1` on visibly colored noise:* Either the noise is genuinely white, or the training dataset is too short (making $1/\sqrt{N_{\text{eff}}}$ too high to detect weak high-order structure).

---

### Panel 4 (Bottom-Right): Out-of-Sample Whitened Residual ACF

**What is plotted:** The sample autocorrelation $\rho_{\text{white}}[k]$ ($k \ge 1$, purple) of held-out cross-validation windows after passing through `whiten()`, overlaid on the shaded green 95% white-noise confidence band $\pm 2 / \sqrt{M(p - 1)}$.

**Healthy fit:** This is the primary pass/fail check of the entire pipeline. $\rho_{\text{white}}[k]$ fluctuates randomly around zero with roughly 95% of lags inside the green shaded band and no coherent structure.

**Warning signs:**

* *Damped oscillations or spikes extending outside the green band:* Residual correlation left behind by an under-resolved peak or notch (check Panel 1 and Panel 3 to see if `max_order` capped `selected_order` too early).

* *Smooth non-zero curve across all lags:* Non-stationary baseline drift or steps inside individual length-$p$ windows that a single constant-offset projection (`project_offset=True`) cannot fully remove.
