"""
Algorithms to analyze noise data.
"""

from __future__ import annotations

import warnings
import json
import os
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pylab as plt
import scipy as sp
import tempfile
import uuid
import matplotlib.figure

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import datetime, UTC
from numpy import ndarray
from numpy.typing import NDArray, ArrayLike
from pathlib import Path
from scipy import signal
from typing import Any, Literal

import mass2
from mass2.mathstat.toeplitz import levinson_durbin


def calc_discontinuous_autocorrelation(data: ArrayLike, max_excursion: int = 1000) -> NDArray:
    """Calculate the autocorrelation of the input data, assuming the rows of the array are NOT
    continuous in time.

    Parameters
    ----------
    data : ArrayLike
        A 2D array of noise data. Shape is `(ntraces, nsamples)`.
    max_excursion : int, optional
        _description_, by default 1000

    Returns
    -------
    NDArray
        The mean autocorrelation of the rows ("traces") in the input `data`, from lags `[0, nsamples-1]`,
        corrected for the bias of removing each trace's mean. It is fixed only up to a constant on all lags,
        chosen to make it positive definite. If no constant does that, a warning is issued and the biased
        estimate is returned.
    """
    data = np.asarray(data)
    ntraces, nsamples = data.shape
    nfft = sp.fft.next_fast_len(2 * nsamples - 1, real=True)
    power = np.zeros(nfft // 2 + 1, dtype=float)

    traces_used = 0
    for i in range(0, ntraces, 100):
        pulses = data[i : i + 100] - data[i : i + 100].mean(axis=1, keepdims=True)
        pulses = pulses[~(np.abs(pulses).max(axis=1) > max_excursion)]
        power += (np.abs(sp.fft.rfft(pulses, n=nfft)) ** 2).sum(axis=0)
        traces_used += len(pulses)

    if traces_used == 0:
        raise ValueError("Apparently all noise traces had large excursions from baseline, so no autocorrelation was computed")
    return _unbiased_autocorrelation(sp.fft.irfft(power, n=nfft)[:nsamples] / traces_used, nsamples)


def _unbias_mean_removed_lag_sums(lag_sums: NDArray) -> NDArray:
    """Return the autocorrelation r whose expected lag sums, with each record's mean removed, are `lag_sums`.

    E[lag sum k] = sum_{i<N-k} (r_k - u_i - u_{i+k} + m), with u_i = (1/N) sum_j r_|i-j| and m = mean(u).
    This is linear in r and is solved for r by GMRES, each product costing O(N) with cumulative sums
    (u is symmetric, so both u sums are cumsum(u)[N-k]). A constant on all lags drops out, so r is fixed only
    up to a constant, which mass2's filters ignore (they fit a constant baseline). The term `weights * sum(r)`
    selects sum(r) = 0; `_unbiased_autocorrelation` then chooses the constant to make R positive definite.
    """
    n = len(lag_sums)
    npairs = n - np.arange(n)
    weights = np.where(npairs == n, 1.0, 2.0)

    def expected_lag_sums(r: NDArray) -> NDArray:
        r = np.ravel(r)
        cumsum_r = np.concatenate(([0.0], np.cumsum(r)))
        u = (cumsum_r[1:] + cumsum_r[n:0:-1] - r[0]) / n
        cumsum_u = np.concatenate(([0.0], np.cumsum(u)))
        m = cumsum_u[-1] / n
        return npairs * (r + m) - 2 * cumsum_u[npairs] + weights * r.sum()

    operator = sp.sparse.linalg.LinearOperator((n, n), matvec=expected_lag_sums, dtype=float)
    preconditioner = sp.sparse.linalg.LinearOperator((n, n), matvec=lambda x: np.ravel(x) / npairs, dtype=float)
    r, info = sp.sparse.linalg.gmres(
        operator, lag_sums, x0=lag_sums / npairs, M=preconditioner, rtol=1e-12, atol=0.0, restart=100, maxiter=20
    )
    if info != 0 or not np.linalg.norm(expected_lag_sums(r) - lag_sums) <= 1e-11 * np.linalg.norm(lag_sums):
        warnings.warn(f"Unbiasing the noise autocorrelation did not converge ({info=}); using the biased estimate", stacklevel=4)
        return lag_sums / n
    return r


def _unbiased_autocorrelation(lag_sums: NDArray, n_lags: int) -> NDArray:
    r = _unbias_mean_removed_lag_sums(lag_sums)[:n_lags]
    f, is_pd = levinson_durbin(r, check_pd=True)
    if is_pd:
        return r
    s = np.cumsum(f)
    inv_r_sum = (s @ s - (s[-1] - s) @ (s[-1] - s)) / f[0]
    if inv_r_sum < 0 and levinson_durbin(r - 2 / inv_r_sum, check_pd=True)[1]:
        return r - 2 / inv_r_sum
    warnings.warn("The unbiased noise autocorrelation is not positive definite; using the biased estimate", stacklevel=3)
    return lag_sums[:n_lags] / len(lag_sums)


def calc_continuous_autocorrelation(data: ArrayLike, n_lags: int, max_excursion: int = 1000, block_starts: ArrayLike = ()) -> NDArray:
    """Calculate the autocorrelation of the input data, assuming the entire array is continuous.

    Parameters
    ----------
    data : ArrayLike
        Data to be autocorrelated. Arrays of 2+ dimensions will be converted to a 1D array via `.ravel()`.
    n_lags : int
        Compute the autocorrelation for lags in the range `[0, n_lags-1]`.
    max_excursion : int, optional
        Chunks of data with max absolute excursion from the mean this large will be excluded from the calculation, by default 1000
    block_starts : ArrayLike, optional
        Indices into the raveled `data` where a new continuous block starts; no chunk spans two blocks,
        by default () (all data are one continuous block)

    Returns
    -------
    NDArray
        The autocorrelation array, corrected for the bias of removing each chunk's mean, as in
        `calc_discontinuous_autocorrelation`.

    Raises
    ------
    ValueError
        If the data are too short to provide the requested number of lags, or the data contain apparent pulses.
    """
    data = np.asarray(data).ravel()
    n_data = len(data)
    assert n_lags < n_data

    # When there are 10 million data points and only 10,000 lags wanted,
    # it's hugely inefficient to compute the full autocorrelation, especially
    # in memory.  Instead, compute it on chunks twice the length of the desired
    # correlation, and average.
    CHUNK_MULTIPLE = 2
    chunksize = CHUNK_MULTIPLE * n_lags
    blocks = np.split(data, np.asarray(block_starts, dtype=int))
    datachunks = np.concatenate([b[: len(b) // chunksize * chunksize].reshape(-1, chunksize) for b in blocks])
    if len(datachunks) == 0:
        msg = f"There are too few data values ({n_data=}) to compute at least {n_lags} lags."
        raise ValueError(msg)

    # Be sure to pad chunksize samples by AT LEAST chunksize - 1 zeros, to prevent
    # unwanted wraparound in the autocorrelation.
    # padded_data is what we do DFT/InvDFT on; ac is the unnormalized output.
    nfft = sp.fft.next_fast_len(2 * chunksize - 1)
    padded_data = np.zeros(nfft, dtype=float)

    ac = np.zeros(chunksize, dtype=float)

    entries = 0

    for chunk in datachunks:
        padded_data[:chunksize] = chunk - chunk.mean()
        if np.abs(padded_data).max() > max_excursion:
            continue

        ft = sp.fft.rfft(padded_data, n=nfft)
        ft[0] = 0  # this redundantly removes the mean of the data set
        power = (ft * ft.conj()).real
        acsum = sp.fft.irfft(power, n=nfft)
        ac += acsum[:chunksize]
        entries += 1

    if entries == 0:
        raise ValueError("Apparently all 'noise' chunks had large excursions from baseline, so no autocorrelation was computed")

    return _unbiased_autocorrelation(ac / entries, n_lags)


def calc_autocorrelation_times(n: int, dt: float) -> NDArray:
    """Compute the timesteps for an autocorrelation function

    Parameters
    ----------
    n : int
        Number of lags
    dt : float
        Sample time

    Returns
    -------
    NDArray
        The time delays for each lag
    """
    return np.arange(n) * dt


def noise_psd_periodogram(
    data: ndarray,
    dt: float,
    window: ndarray | str = "boxcar",
) -> NoiseResult:
    """Compute the noise power spectral density using scipy's periodogram function and the autocorrelation."""
    f, Pxx = sp.signal.periodogram(data - np.mean(data), fs=1 / dt, window=window, axis=-1, detrend=False)
    # len(f) = data.shape[1]//2+1
    # Pxx[i, j] is the PSD at frequency f[j] for the i‑th trace data[i, :]
    Pxx_mean = np.mean(Pxx, axis=0)
    # Pxx_mean[j] is the averaged PSD at frequency f[j] over all traces
    autocorr_vec = calc_discontinuous_autocorrelation(data)
    return NoiseResult(psd=Pxx_mean, autocorr_vec=autocorr_vec, frequencies=f)


def calc_noise_result(
    data: ArrayLike,
    dt: float,
    continuous: bool,
    window: Callable | None = None,
    skip_autocorr_if_length_over: int = 100_000,
    block_starts: ArrayLike = (),
) -> NoiseResult:
    """Analyze the noise as Mass has always done.

    * Compute autocorrelation with a lower noise at longer lags when data are known to be continuous
    * Subtract the mean before computing the power spectrum

    Parameters
    ----------
    data : ArrayLike
        A 2d array of noise data, of shape `(npulses, nsamples)`
    dt : float
        Periodic sampling time, in seconds
    continuous : bool
        Whether the "pulses" in the `data` array are continuous in time
    window : callable, optional
        A function to compute a data window (or if None, no windowing), by default None
    block_starts : ArrayLike, optional
        Indices of the records that start a new continuous block (e.g. after dropped records); when `continuous`,
        records of different blocks are never joined, by default () (all records are one continuous block)

    Returns
    -------
    NoiseResult
        The derived noise spectrum and autocorrelation
    """
    data = np.asarray(data)
    data_zeromean = data - np.mean(data)
    (n_pulses, nsamples) = data_zeromean.shape
    blocks = np.split(data_zeromean, np.asarray(block_starts, dtype=int))
    if continuous:
        m = nsamples // 2
        spec = mass2.mathstat.power_spectrum.PowerSpectrum(m, dt=dt)
        for block in blocks:
            for i in range(2 * len(block) - 1):
                spec.addDataSegment(block.ravel()[i * m : (i + 2) * m], window=window)
        f_mass, psd_mass = spec.frequencies(), spec.spectrum()
    else:
        spec = mass2.mathstat.power_spectrum.PowerSpectrum(nsamples // 2, dt=dt)
        for record in data_zeromean:
            spec.addDataSegment(record[: 2 * (nsamples // 2)], window=window)
        f_mass, psd_mass = spec.frequencies(), spec.spectrum()
    if nsamples <= skip_autocorr_if_length_over:
        if continuous and max(len(block) for block in blocks) >= 2:
            starts = np.asarray(block_starts, dtype=int) * nsamples
            autocorr_vec = calc_continuous_autocorrelation(data_zeromean.ravel(), n_lags=nsamples, block_starts=starts)
        else:
            autocorr_vec = calc_discontinuous_autocorrelation(data_zeromean)
    else:
        print(
            """warning: noise_psd_mass skipping autocorrelation calculation for long traces,
            use skip_autocorr_if_length_over argument to override this"""
        )
        autocorr_vec = None
    return NoiseResult(psd=psd_mass, autocorr_vec=autocorr_vec, frequencies=f_mass)


@dataclass
class NoiseResult:
    """A dataclass to hold the results of noise analysis, both power-spectral density and (optionally) autocorrelation."""

    psd: np.ndarray
    autocorr_vec: np.ndarray | None
    frequencies: np.ndarray

    def plot(
        self,
        axis: plt.Axes | None = None,
        arb_to_unit_scale_and_label: tuple[int, str] = (1, "arb"),
        sqrt_psd: bool = True,
        loglog: bool = True,
        **plotkwarg: Any,
    ) -> None:
        """Plot the power spectral density."""
        if axis is None:
            plt.figure()
            axis = plt.gca()
        arb_to_unit_scale, unit_label = arb_to_unit_scale_and_label
        psd = self.psd[1:] * (arb_to_unit_scale**2)
        freq = self.frequencies[1:]
        if sqrt_psd:
            axis.plot(freq, np.sqrt(psd), **plotkwarg)
            axis.set_ylabel(f"Amplitude Spectral Density ({unit_label}$/\\sqrt{{Hz}}$)")
        else:
            axis.plot(freq, psd, **plotkwarg)
            axis.set_ylabel(f"Power Spectral Density ({unit_label}$^2$ Hz$^{{-1}}$)")
        if loglog:
            axis.loglog()
        axis.grid()
        axis.set_xlabel("Frequency (Hz)")
        axis.set_title(f"noise from records of length {len(self.frequencies) * 2 - 2}")

    def plot_log_rebinned(
        self,
        bins_per_decade: int = 10,
        axis: plt.Axes | None = None,
        arb_to_unit_scale_and_label: tuple[int, str] = (1, "arb"),
        sqrt_psd: bool = True,
        **plotkwarg: Any,
    ) -> None:
        """Plot PSD rebinned into logarithmically spaced frequency bins."""
        if axis is None:
            plt.figure()
            axis = plt.gca()

        arb_to_unit_scale, unit_label = arb_to_unit_scale_and_label
        psd = self.psd[1:] * (arb_to_unit_scale**2)
        freq = self.frequencies[1:]

        # define logarithmically spaced bin edges
        fmin, fmax = freq[0], freq[-1]
        n_decades = np.log10(fmax / fmin)
        n_bins = int(bins_per_decade * n_decades)
        bin_edges = np.logspace(np.log10(fmin), np.log10(fmax), n_bins + 1)

        # digitize frequencies into bins
        inds = np.digitize(freq, bin_edges)

        # average PSD per bin
        binned_freqs = np.zeros(n_bins, dtype=float)
        binned_psd = np.zeros(n_bins, dtype=float)
        for i in range(1, len(bin_edges)):
            mask = inds == i
            if np.any(mask):
                binned_freqs[i - 1] = np.exp(np.mean(np.log(freq[mask])))  # geometric mean
                binned_psd[i - 1] = np.mean(psd[mask])

        if sqrt_psd:
            axis.plot(binned_freqs, np.sqrt(binned_psd), **plotkwarg)
            axis.set_ylabel(f"Amplitude Spectral Density ({unit_label}$/\\sqrt{{Hz}}$)")
        else:
            axis.plot(binned_freqs, binned_psd, **plotkwarg)
            axis.set_ylabel(f"Power Spectral Density ({unit_label}$^2$ Hz$^{{-1}}$)")

        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.grid(True, which="both")
        axis.set_xlabel("Frequency (Hz)")
        axis.set_title(f"Log-rebinned noise from {len(self.frequencies) * 2 - 2} samples")


class ARNoiseWhitener:
    """Data-driven AR(q) noise modeler and matrix-free whitener for pulses of length `p >> q`.

    * De-means `n_sequences` training sequences (each of length `>= p`) and fits a pooled Burg AR model
      in `O(n_sequences * seq_length * log(max_order) + n_sequences * max_order^2 + max_order^3)` time.
    * Automatically selects model order `q <= max_order` (`<= 256`) where higher orders only fit noise.
    * Provides exact `O(p * q)` matrix-free Cholesky + offset-invariant GLS whitening for length-`p`
      pulses, as well as calibrated linear parameter uncertainty predictions.
    * Provides causal lower-triangular Toeplitz FIR (and exact AR) whitening for vectors of length `> p`.
    * Builds a comprehensive noise digest (autocorrelation to lag `p - 1`, AR(q) PSD, direct Welch PSD,
      and out-of-sample whitened residual autocorrelation) with 4-panel Matplotlib diagnostics.
    * Supports atomic single-row and multi-row persistence in Parquet and Apache Arrow IPC files.

    Attributes
    ----------
    pulse_length : int
        Target pulse record length `p`.
    selected_order : int
        Automatically chosen AR model order `q` (`1 <= q <= max_order`).
    max_order : int
        Maximum candidate AR model order evaluated during Burg fitting.
    sample_rate : float
        Sampling frequency in Hz.
    project_offset : bool
        Whether length-`p` whitening projects out the constant-offset subspace.
    n_sequences : int
        Number of training noise sequences passed to the constructor.
    sequence_length : int
        Sample length of each training noise sequence.
    label : str
        Optional human-readable identifier for this noise model.
    model_id : str
        Unique string identifier (UUID4 by default) for archiving and lookup.
    created_at : str
        ISO-8601 UTC timestamp recorded when the model was constructed.
    metadata : dict[str, Any]
        Arbitrary JSON-serializable metadata dictionary associated with this model.
    reflection_coeffs : np.ndarray
        Pooled Burg reflection coefficients `kappa[0 .. max_order]` (`kappa[0] == 0.0`).
    innovation_vars_by_order : np.ndarray
        One-step prediction error variance `E[0 .. max_order]` across candidate orders.
    noise_floor_kappa : float
        Theoretical white-noise standard deviation `1 / sqrt(N_eff)` of the reflection coefficients.
    ar_coeffs : np.ndarray
        Fitted AR(q) polynomial coefficients `[1.0, a_1, ..., a_q]` of shape `(q + 1,)`.
    innovation_var : float
        Uncalibrated one-step innovation variance `E_q` at the selected order `q`.
    calibration_alpha : float
        Out-of-sample cross-validated scale calibration factor applied during whitening.
    w_hat : np.ndarray | None
        Unit-norm whitened constant vector `L^-1 1 / ||L^-1 1||` of shape `(p,)` when
        `project_offset=True`, or `None` when `project_offset=False`.
    digest : dict[str, Any]
        Summary statistics, calibrated autocovariance/autocorrelation out to lag `p - 1`,
        one-sided AR(q) and Welch PSD arrays, and whitened residual autocorrelation diagnostics.
    """

    def __init__(  # noqa: PLR0917
        self,
        sequences: np.ndarray,
        pulse_length: int,
        max_order: int = 256,
        sample_rate: float = 1.0,
        project_offset: bool = True,
        calibrate: bool = True,
        n_folds: int = 5,
        n_psd_bins: int = 8192,
        label: str = "",
        model_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Fit an AR(q) noise model from de-meaned training sequences and build whitening operators.

        Parameters
        ----------
        sequences : np.ndarray
            Noise training data of shape `(n_sequences, seq_length)` or 1D `(seq_length,)`, where
            `seq_length >= pulse_length`. Each sequence is de-meaned individually before fitting.
        pulse_length : int
            Target pulse record length `p` (`p >= 16` and `p > max_order`).
        max_order : int, optional
            Maximum candidate AR order `q_max` (`1 <= max_order <= 256`), by default 256
        sample_rate : float, optional
            Sampling frequency in Hz used for PSD frequency axes and scaling, by default 1.0
        project_offset : bool, optional
            If `True`, project out the constant-offset direction in whitened space so pulse
            whitening and parameter estimates are invariant to additive DC shifts, by default True
        calibrate : bool, optional
            If `True` and `n_sequences >= 2`, run K-fold cross-validation across sequences to
            calibrate the global variance scale `calibration_alpha`, by default True
        n_folds : int, optional
            Number of cross-validation folds used when `calibrate=True`, by default 5
        n_psd_bins : int, optional
            FFT length used to evaluate the one-sided AR(q) PSD in `self.digest`, by default 8192
        label : str, optional
            Optional human-readable label stored with the model and shown in plots, by default ""
        model_id : str | None, optional
            Explicit unique identifier for persistence; auto-generated UUID4 if `None`, by default None
        metadata : dict[str, Any] | None, optional
            Optional JSON-serializable metadata dictionary stored alongside the model, by default None

        Raises
        ------
        ValueError
            If `sequences` is not 1D or 2D, `pulse_length < 16`, `seq_length < pulse_length`,
            `max_order` is outside `[1, 256]`, `pulse_length <= max_order`, or input sequences have
            zero variance after de-meaning.
        """
        seqs = np.asarray(sequences, dtype=np.float64)
        if seqs.ndim == 1:
            seqs = seqs[None, :]
        if seqs.ndim != 2:
            raise ValueError("`sequences` must be a 2D array of shape (n_sequences, seq_length).")

        n_sequences, seq_length = seqs.shape
        if pulse_length < 16:
            raise ValueError("`pulse_length` (p) must be at least 16.")
        if seq_length < pulse_length:
            raise ValueError(f"Each noise sequence must have length >= pulse_length ({seq_length} < {pulse_length}).")
        if not (1 <= max_order <= 256):
            raise ValueError("`max_order` must be between 1 and 256.")
        if pulse_length <= max_order:
            raise ValueError(f"`pulse_length` ({pulse_length}) must be greater than `max_order` ({max_order}).")

        self.pulse_length = int(pulse_length)
        self.max_order = int(max_order)
        self.sample_rate = float(sample_rate)
        self.project_offset = bool(project_offset)
        self.n_sequences = int(n_sequences)
        self.sequence_length = int(seq_length)
        self.label = str(label)
        self.model_id = str(model_id) if model_id is not None else str(uuid.uuid4())
        self.created_at = datetime.now(UTC).isoformat()
        self.metadata: dict[str, Any] = dict(metadata) if metadata is not None else {}

        # De-mean each sequence individually
        demeaned = seqs - np.mean(seqs, axis=1, keepdims=True)

        # 1. Fit pooled Burg up to max_order and select q where higher lags only fit noise
        a_hist, errs_by_order, kappas, noise_floor_kappa = self._fit_pooled_burg_all_orders(demeaned, self.max_order)
        self.reflection_coeffs = kappas
        self.innovation_vars_by_order = errs_by_order
        self.noise_floor_kappa = float(noise_floor_kappa)

        self.selected_order = self._select_model_order(kappas, noise_floor_kappa)
        self.ar_coeffs = a_hist[self.selected_order, : self.selected_order + 1].copy()
        self.innovation_var = float(errs_by_order[self.selected_order])

        # 2. Build matrix-free step-down startup filters, steady-state FIR filter, and w_hat
        self.calibration_alpha = 1.0
        self._build_operator_state()

        # 3. Out-of-sample cross-validation for scale calibration & whitened residual check
        if calibrate and self.n_sequences >= 2:
            self.calibration_alpha, whitened_acf, whitened_acf_bound = self._cross_validate(
                demeaned, n_folds=min(n_folds, self.n_sequences)
            )
        else:
            whitened_acf, whitened_acf_bound = self._evaluate_whitened_acf(demeaned)

        # 4. Compute and store comprehensive noise digest
        self.digest = self._build_digest(
            demeaned=demeaned,
            n_psd_bins=n_psd_bins,
            whitened_acf=whitened_acf,
            whitened_acf_bound=whitened_acf_bound,
        )

    # ------------------------------------------------------------------
    # Internal estimation & order selection
    # ------------------------------------------------------------------

    @staticmethod
    def _fit_pooled_burg_all_orders(demeaned: np.ndarray, max_order: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:  # noqa: PLR0914
        """Fit pooled Burg reflection coefficients and AR polynomials for orders `1 .. max_order`.

        Combines overlap-save block-FFT autocorrelation with exact boundary Gram matrix
        corrections so the result matches standard sample-domain Burg recursion to machine
        precision in `O(S * N * log(q_max) + S * q_max^2 + q_max^3)` time.

        Parameters
        ----------
        demeaned : np.ndarray
            Zero-mean training noise array of shape `(n_sequences, seq_length)`.
        max_order : int
            Maximum AR model order `q_max` to compute.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray, float]
            - `a_hist`: Lower-triangular array of shape `(q_max + 1, q_max + 1)` where row `m`
              holds `[1.0, a_1^(m), ..., a_m^(m), 0, ...]`.
            - `errs`: One-step prediction error variances `E[0 .. q_max]` of shape `(q_max + 1,)`.
            - `kappas`: Pooled reflection coefficients `kappa[0 .. q_max]` of shape `(q_max + 1,)`.
            - `noise_floor_kappa`: Expected white-noise standard deviation `1 / sqrt(N_eff)` of
              individual reflection coefficients.

        Raises
        ------
        ValueError
            If the pooled zero-lag variance `errs[0]` is not strictly positive.
        """
        S, N = demeaned.shape
        q_max = min(max_order, N - 1)

        # Block-FFT pooled autocorrelation R_full[0 .. q_max]
        step = 1 << int(np.ceil(np.log2(max(4 * q_max, 256))))
        nfft = 1 << int(np.ceil(np.log2(step + 2 * q_max)))
        P_spec = np.zeros(nfft // 2 + 1, dtype=np.complex128)

        for start in range(0, N, step):
            xl = demeaned[:, start : min(start + step, N)]
            xr = demeaned[:, start : min(start + step + q_max, N)]
            Fl = np.fft.rfft(xl, n=nfft, axis=1)
            Fr = np.fft.rfft(xr, n=nfft, axis=1)
            P_spec += np.sum(np.conj(Fl) * Fr, axis=0)

        R_full = np.fft.irfft(P_spec, n=nfft)[: q_max + 1]

        # Boundary Gram matrices from first q_max and last q_max samples
        head = demeaned[:, :q_max]
        tail = demeaned[:, N - q_max :]
        G_head = head.T @ head
        G_tail = tail.T @ tail

        head_csum = np.zeros((q_max + 1, q_max + 1), dtype=np.float64)
        tail_csum = np.zeros((q_max + 1, q_max + 1), dtype=np.float64)
        for k in range(q_max + 1):
            head_csum[k, 1 : q_max - k + 1] = np.cumsum(np.diag(G_head, k))
            tail_csum[k, 1 : q_max - k + 1] = np.cumsum(np.diag(G_tail, k))

        errs = np.zeros(q_max + 1, dtype=np.float64)
        kappas = np.zeros(q_max + 1, dtype=np.float64)
        errs[0] = float(R_full[0] / (S * N))
        if not errs[0] > 0.0:
            raise ValueError("Input noise sequences have zero variance.")

        a_hist = np.zeros((q_max + 1, q_max + 1), dtype=np.float64)
        a_hist[:, 0] = 1.0
        a = np.zeros(q_max + 1, dtype=np.float64)
        a[0] = 1.0

        for m in range(1, q_max + 1):
            idx = np.arange(m + 1)
            ii, jj = np.meshgrid(idx, idx, indexing="ij")
            k_mat = np.abs(jj - ii)
            val = R_full[k_mat].copy()
            val -= head_csum[k_mat, np.maximum(0, m - jj)]
            val -= tail_csum[k_mat, np.maximum(0, q_max - k_mat)] - tail_csum[k_mat, np.maximum(0, q_max - jj)]
            C = np.where(jj >= ii, val, val.T)

            f_vec = np.zeros(m + 1, dtype=np.float64)
            f_vec[:m] = a[:m]
            b_vec = np.zeros(m + 1, dtype=np.float64)
            b_vec[1:] = a[:m][::-1]

            Cf = C @ f_vec
            Cb = C @ b_vec
            kap = float(-2.0 * (f_vec @ Cb) / ((f_vec @ Cf) + (b_vec @ Cb)))

            kappas[m] = kap
            a[1:m] += kap * a[m - 1 : 0 : -1]
            a[m] = kap
            a_hist[m, : m + 1] = a[: m + 1]
            errs[m] = errs[m - 1] * (1.0 - kap * kap)

        n_eff = S * max(1.0, N - 0.5 * q_max)
        noise_floor_kappa = float(1.0 / np.sqrt(n_eff))
        return a_hist, errs, kappas, noise_floor_kappa

    @staticmethod
    def _select_model_order(
        kappas: np.ndarray,
        noise_floor_kappa: float,
        window: int = 16,
        single_lag_z: float = 3.6,
    ) -> int:
        """Select the AR order `q <= max_order` where higher-order reflection coefficients only fit noise.

        Finds the largest lag `m` where either `|kappa[m]|` exceeds `single_lag_z * noise_floor_kappa`
        or the forward `window`-lag block RMS of `kappa` exceeds its white-noise fluctuation threshold.

        Parameters
        ----------
        kappas : np.ndarray
            Reflection coefficients `kappa[0 .. q_max]` of shape `(q_max + 1,)`.
        noise_floor_kappa : float
            Asymptotic white-noise standard deviation `1 / sqrt(N_eff)` of `kappa[m]`.
        window : int, optional
            Forward lookahead block length used to detect sustained exponential tails, by default 16
        single_lag_z : float, optional
            Z-score threshold for isolated reflection coefficient spikes, by default 3.6

        Returns
        -------
        int
            Selected model order `q` (`1 <= q <= len(kappas) - 1`).
        """
        q_max = len(kappas) - 1
        selected_q = 1
        for m in range(1, q_max + 1):
            hi = min(q_max + 1, m + window)
            block = kappas[m:hi]
            dof = len(block)
            block_rms = float(np.sqrt(np.mean(block * block)))
            block_thresh = noise_floor_kappa * np.sqrt(1.0 + 3.0 / np.sqrt(dof))
            if abs(kappas[m]) > single_lag_z * noise_floor_kappa or (dof >= 4 and block_rms > block_thresh):
                selected_q = m
        return selected_q

    def _build_operator_state(self) -> None:
        """Reconstruct Levinson-Durbin step-down startup filters, steady-state FIR taps, and `w_hat`."""
        q = self.selected_order
        self._startup_filters: list[np.ndarray] = [np.empty(0, dtype=np.float64) for _ in range(q + 1)]
        self._startup_errs = np.zeros(q + 1, dtype=np.float64)

        a_cur = self.ar_coeffs.copy()
        e_cur = self.innovation_var
        for m in range(q, 0, -1):
            self._startup_filters[m] = a_cur[1 : m + 1].copy()
            self._startup_errs[m] = e_cur
            kap = a_cur[m]
            a_cur[1:m] = (a_cur[1:m] - kap * a_cur[m - 1 : 0 : -1]) / (1.0 - kap * kap)
            e_cur /= 1.0 - kap * kap
        self._startup_errs[0] = e_cur

        # Uncalibrated causal FIR tap vector h = [1, a_1, ..., a_q] / sqrt(E_q)
        self._h_fir_uncal = self.ar_coeffs / np.sqrt(self._startup_errs[q])

        if self.project_offset:
            w = self._apply_uncal_L_inv(np.ones((1, self.pulse_length), dtype=np.float64))[0]
            self.w_hat: np.ndarray | None = w / np.linalg.norm(w)
        else:
            self.w_hat = None

    def _apply_uncal_L_inv(self, Y: np.ndarray) -> np.ndarray:
        """Apply the uncalibrated lower-triangular inverse Cholesky factor `L^-1` to a 2D batch.

        Parameters
        ----------
        Y : np.ndarray
            2D array of shape `(batch_size, length)` with `length >= selected_order + 1`.

        Returns
        -------
        np.ndarray
            Uncalibrated whitened array `Z0 = Y @ L^-T` of shape `(batch_size, length)`.
        """
        _, L = Y.shape
        q = self.selected_order
        Z = np.zeros_like(Y, dtype=np.float64)
        Z[:, 0] = Y[:, 0] / np.sqrt(self._startup_errs[0])

        for m in range(1, min(q + 1, L)):
            Z[:, m] = (Y[:, m] + Y[:, :m] @ self._startup_filters[m][::-1]) / np.sqrt(self._startup_errs[m])

        if L > q + 1:
            filtered = signal.lfilter(self._h_fir_uncal, [1.0], Y, axis=1)
            assert isinstance(filtered, np.ndarray)
            Z[:, q + 1 :] = filtered[:, q + 1 :]
        return Z

    def _cross_validate(self, demeaned: np.ndarray, n_folds: int) -> tuple[float, np.ndarray, float]:  # noqa: PLR0914
        """Run K-fold cross-validation across sequences to calibrate scale and check residual whiteness.

        Parameters
        ----------
        demeaned : np.ndarray
            Zero-mean training noise array of shape `(n_sequences, seq_length)`.
        n_folds : int
            Number of folds (`2 <= n_folds <= n_sequences`) to split sequences across.

        Returns
        -------
        tuple[float, np.ndarray, float]
            - `alpha`: Empirical out-of-sample variance scale factor `E[||z||^2] / dof`.
            - `acf`: Sample autocorrelation of whitened validation windows up to `min(p - 1, 512)`.
            - `bound`: Approximate 95% white-noise confidence bound `2 / sqrt(M * (p - 1))`.
        """
        p = self.pulse_length
        folds = np.array_split(demeaned, n_folds, axis=0)
        sq_norms = []
        val_whitened = []

        for i in range(n_folds):
            tr = np.concatenate([folds[j] for j in range(n_folds) if j != i], axis=0)
            va = folds[i]

            a_hist_f, errs_f, kappas_f, nf_f = self._fit_pooled_burg_all_orders(tr, self.max_order)
            q_f = self._select_model_order(kappas_f, nf_f)
            a_f = a_hist_f[q_f, : q_f + 1]
            e_f = float(errs_f[q_f])

            # Temporary lightweight fold whitener
            fold_model = object.__new__(ARNoiseWhitener)
            fold_model.pulse_length = p
            fold_model.selected_order = q_f
            fold_model.ar_coeffs = a_f
            fold_model.innovation_var = e_f
            fold_model.project_offset = self.project_offset
            fold_model.calibration_alpha = 1.0
            fold_model._build_operator_state()

            n_win = va.shape[1] // p
            wins = va[:, : n_win * p].reshape(-1, p)
            z = fold_model.whiten(wins)
            sq_norms.append(np.sum(z * z, axis=1))
            val_whitened.append(z)

        dof_per_win = (p - 1) if self.project_offset else p
        alpha = float(np.mean(np.concatenate(sq_norms)) / dof_per_win)

        Z_all = np.concatenate(val_whitened, axis=0) / np.sqrt(alpha)
        acf, bound = self._compute_batch_acf(Z_all, max_lag=min(p - 1, 512))
        return alpha, acf, bound

    def _evaluate_whitened_acf(self, demeaned: np.ndarray) -> tuple[np.ndarray, float]:
        """Compute in-sample whitened residual autocorrelation when cross-validation is skipped.

        Parameters
        ----------
        demeaned : np.ndarray
            Zero-mean training noise array of shape `(n_sequences, seq_length)`.

        Returns
        -------
        tuple[np.ndarray, float]
            - `acf`: Sample autocorrelation of whitened length-`p` windows.
            - `bound`: Approximate 95% white-noise confidence bound `2 / sqrt(M * (p - 1))`.
        """
        p = self.pulse_length
        n_win = demeaned.shape[1] // p
        wins = demeaned[:, : n_win * p].reshape(-1, p)
        Z = self.whiten(wins)
        return self._compute_batch_acf(Z, max_lag=min(p - 1, 512))

    @staticmethod
    def _compute_batch_acf(Z: np.ndarray, max_lag: int) -> tuple[np.ndarray, float]:
        """Compute the pooled sample autocorrelation across rows of a whitened batch `Z`.

        Parameters
        ----------
        Z : np.ndarray
            2D array of whitened windows of shape `(n_windows, pulse_length)`.
        max_lag : int
            Maximum lag `k` to evaluate (`0 <= k <= pulse_length - 1`).

        Returns
        -------
        tuple[np.ndarray, float]
            - `acf`: Normalized sample autocorrelation vector of shape `(max_lag + 1,)`.
            - `bound`: Approximate 95% white-noise confidence bound `2 / sqrt(M * (p - 1))`.
        """
        M, p = Z.shape
        Z_c = Z - np.mean(Z, axis=1, keepdims=True)
        var0 = float(np.mean(Z_c * Z_c))
        acf = np.ones(max_lag + 1, dtype=np.float64)
        for k in range(1, max_lag + 1):
            acf[k] = float(np.mean(Z_c[:, k:] * Z_c[:, :-k]) / var0)
        bound = float(2.0 / np.sqrt(M * max(1, p - 1)))
        return acf, bound

    # ------------------------------------------------------------------
    # (a) Exact whitening for length-p pulses & linear uncertainty prediction
    # ------------------------------------------------------------------

    def whiten(self, y: np.ndarray) -> np.ndarray:
        """Apply calibrated, offset-invariant AR(q) Cholesky whitening to vectors of length `p`.

        Executes in `O(p * q)` time per record without materializing a `p x p` matrix.

        Parameters
        ----------
        y : np.ndarray
            Input pulse or noise record(s) of shape `(pulse_length,)` or
            `(batch_size, pulse_length)`.

        Returns
        -------
        np.ndarray
            Whitened array with the exact same shape as `y`.

        Raises
        ------
        ValueError
            If `y` is not 1D or 2D, or its trailing dimension does not equal `self.pulse_length`.
        """
        arr = np.asarray(y, dtype=np.float64)
        is_1d = arr.ndim == 1
        Y_2d = arr[None, :] if is_1d else arr
        if Y_2d.ndim != 2 or Y_2d.shape[1] != self.pulse_length:
            raise ValueError(f"Expected vectors of length p={self.pulse_length}, got shape {arr.shape}.")

        Z0 = self._apply_uncal_L_inv(Y_2d)
        if self.project_offset and self.w_hat is not None:
            Z0 -= np.outer(Z0 @ self.w_hat, self.w_hat)
        Z = Z0 / np.sqrt(self.calibration_alpha)
        return Z[0] if is_1d else Z

    def fit_linear_model(self, y: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Estimate linear model parameters and calibrated covariance via Generalized Least Squares.

        Solves `min_beta ||W (y - X beta)||_2^2` where `W` is the calibrated, offset-invariant
        AR(q) whitening operator.

        Parameters
        ----------
        y : np.ndarray
            Observed record of shape `(pulse_length,)`.
        X : np.ndarray
            Linear design matrix of shape `(pulse_length, n_params)` whose columns are model
            templates (e.g., pulse shapes or sinusoids).

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            - `beta_hat`: Optimal GLS parameter estimates of shape `(n_params,)`.
            - `cov_beta`: Predicted calibrated parameter covariance `(W_X^T W_X)^-1` of shape
              `(n_params, n_params)`.

        Raises
        ------
        ValueError
            If `X` is not 2D with first dimension equal to `self.pulse_length`, or `y` does not
            have length `self.pulse_length`.
        """
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim != 2 or X_arr.shape[0] != self.pulse_length:
            raise ValueError(f"Design matrix X must have shape ({self.pulse_length}, n_params).")
        z = self.whiten(y)
        WX = self.whiten(X_arr.T).T
        cov_beta = np.linalg.inv(WX.T @ WX)
        beta_hat = cov_beta @ (WX.T @ z)
        return beta_hat, cov_beta

    def predict_parameter_cov(self, X: np.ndarray) -> np.ndarray:
        """Predict the calibrated GLS parameter error covariance matrix for design matrix `X`.

        Parameters
        ----------
        X : np.ndarray
            Linear design matrix of shape `(pulse_length, n_params)`.

        Returns
        -------
        np.ndarray
            Predicted parameter covariance matrix `(W_X^T W_X)^-1` of shape `(n_params, n_params)`.

        Raises
        ------
        ValueError
            If `X` is not 2D with first dimension equal to `self.pulse_length`.
        """
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim != 2 or X_arr.shape[0] != self.pulse_length:
            raise ValueError(f"Design matrix X must have shape ({self.pulse_length}, n_params).")
        WX = self.whiten(X_arr.T).T
        return np.linalg.inv(WX.T @ WX)

    # ------------------------------------------------------------------
    # (b) Approximate Toeplitz (and exact AR) whitening for length > p
    # ------------------------------------------------------------------

    @property
    def toeplitz_fir_taps(self) -> np.ndarray:
        """Return the calibrated causal FIR tap vector defining the lower-triangular Toeplitz whitener.

        Returns
        -------
        np.ndarray
            1D tap array `h = [1, a_1, ..., a_q] / sqrt(alpha * E_q)` of shape `(selected_order + 1,)`
            such that `(T_h y)_t = sum_{k=0}^q h_k y_{t-k}`.
        """
        return self._h_fir_uncal / np.sqrt(self.calibration_alpha)

    def whiten_long(
        self,
        y: np.ndarray,
        mode: Literal["toeplitz", "exact_ar"] = "toeplitz",
        anchor_first: bool = True,
    ) -> np.ndarray:
        """Whiten vectors of arbitrary length `M >= p` using Toeplitz FIR or exact AR Cholesky filtering.

        Parameters
        ----------
        y : np.ndarray
            Input record(s) of shape `(M,)` or `(batch_size, M)` with `M >= self.pulse_length`.
        mode : {"toeplitz", "exact_ar"}, optional
            Whitening operator mode, by default "toeplitz":
            - `"toeplitz"`: Applies the calibrated lower-triangular Toeplitz causal FIR filter
              `T_h` in `O(M * q)` time.
            - `"exact_ar"`: Applies the exact AR(q) startup Cholesky rows for `t <= q`, continues
              with the causal FIR filter for `t > q`, and applies a rank-1 constant-offset
              projection over length `M` when `self.project_offset=True`.
        anchor_first : bool, optional
            When `mode="toeplitz"` and `self.project_offset=True`, subtract `y[..., 0]` from each
            record before filtering so large constant pedestals do not trigger a `t = 0` step-jump
            transient, by default True

        Returns
        -------
        np.ndarray
            Whitened array with the exact same shape as `y`.

        Raises
        ------
        ValueError
            If `y` has length `< self.pulse_length`, is not 1D or 2D, or `mode` is unrecognized.
        """
        arr = np.asarray(y, dtype=np.float64)
        is_1d = arr.ndim == 1
        Y_2d = arr[None, :] if is_1d else arr
        if Y_2d.ndim != 2 or Y_2d.shape[1] < self.pulse_length:
            raise ValueError(f"Expected vectors of length >= p={self.pulse_length}, got shape {arr.shape}.")

        if mode == "toeplitz":
            work = Y_2d - Y_2d[:, :1] if (anchor_first and self.project_offset) else Y_2d
            Z = signal.lfilter(self.toeplitz_fir_taps, [1.0], work, axis=1)
            assert isinstance(Z, np.ndarray)
        elif mode == "exact_ar":
            Z0 = self._apply_uncal_L_inv(Y_2d)
            if self.project_offset:
                w_long = self._apply_uncal_L_inv(np.ones((1, Y_2d.shape[1]), dtype=np.float64))[0]
                w_long /= np.linalg.norm(w_long)
                Z0 -= np.outer(Z0 @ w_long, w_long)
            Z = Z0 / np.sqrt(self.calibration_alpha)
        else:
            raise ValueError(f"Unknown mode '{mode}'. Use 'toeplitz' or 'exact_ar'.")

        return Z[0] if is_1d else Z

    # ------------------------------------------------------------------
    # (c) Digest of the noise analysis
    # ------------------------------------------------------------------

    def _build_digest(
        self,
        demeaned: np.ndarray,
        n_psd_bins: int,
        whitened_acf: np.ndarray,
        whitened_acf_bound: float,
    ) -> dict[str, Any]:
        """Compute the noise analysis digest (autocorrelation to length `p`, AR(q) PSD, and Welch PSD).

        Parameters
        ----------
        demeaned : np.ndarray
            Zero-mean training noise array of shape `(n_sequences, seq_length)`.
        n_psd_bins : int
            FFT length used to evaluate the parametric AR(q) PSD.
        whitened_acf : np.ndarray
            Whitened residual autocorrelation array from cross-validation or training evaluation.
        whitened_acf_bound : float
            Approximate 95% white-noise confidence bound for `whitened_acf`.

        Returns
        -------
        dict[str, Any]
            Dictionary containing scalar summary metrics (`rms_noise`, `innovation_std`,
            `spectral_dynamic_range_db`, etc.) and 1D arrays (`lags`, `autocov`, `autocorr`,
            `freqs_ar`, `psd_ar`, `freqs_welch`, `psd_welch`, `whitened_acf`).
        """
        p = self.pulse_length
        q = self.selected_order
        fs = self.sample_rate

        # 1. Calibrated autocovariance & autocorrelation out to length p (lags 0 .. p-1)
        gamma = np.zeros(p, dtype=np.float64)
        gamma[0] = self._startup_errs[0]
        a_up = np.zeros(q + 1, dtype=np.float64)

        for m in range(1, q + 1):
            kap = float(self._startup_filters[m][-1])
            e_prev = float(self._startup_errs[m - 1])
            gamma[m] = -kap * e_prev - float(a_up[1:m] @ gamma[m - 1 : 0 : -1])
            a_up[1:m] += kap * a_up[m - 1 : 0 : -1]
            a_up[m] = kap

        if p > q + 1:
            ar_rev = self.ar_coeffs[1:][::-1]
            for k in range(q + 1, p):
                gamma[k] = -float(ar_rev @ gamma[k - q : k])

        gamma *= self.calibration_alpha
        acf = gamma / gamma[0]

        # 2. One-sided calibrated AR(q) PSD
        freqs_ar = np.fft.rfftfreq(n_psd_bins, d=1.0 / fs)
        A_freq = np.fft.rfft(self.ar_coeffs, n=n_psd_bins)
        innov_var_cal = self.calibration_alpha * self.innovation_var
        psd_ar = (innov_var_cal / fs) / (np.abs(A_freq) ** 2)
        psd_ar[1:-1] *= 2.0  # one-sided convention

        # 3. Direct Welch-windowed PSD averaged across all de-meaned training sequences
        nperseg = min(self.sequence_length, p)
        freqs_welch, psd_welch_all = signal.welch(
            demeaned,
            fs=fs,
            window="hann",
            nperseg=nperseg,
            noverlap=nperseg // 2,
            detrend="constant",
            return_onesided=True,
            scaling="density",
            axis=1,
        )
        psd_welch = np.mean(psd_welch_all, axis=0)

        dyn_range_db = float(10.0 * np.log10(np.max(psd_ar) / np.min(psd_ar)))

        return {
            "selected_order": q,
            "max_order": self.max_order,
            "pulse_length": p,
            "sample_rate": fs,
            "calibration_alpha": float(self.calibration_alpha),
            "rms_noise": float(np.sqrt(gamma[0])),
            "innovation_std": float(np.sqrt(innov_var_cal)),
            "spectral_dynamic_range_db": dyn_range_db,
            "noise_floor_kappa": float(self.noise_floor_kappa),
            "lags": np.arange(p, dtype=np.int64),
            "autocov": gamma,
            "autocorr": acf,
            "freqs_ar": freqs_ar,
            "psd_ar": psd_ar,
            "freqs_welch": freqs_welch,
            "psd_welch": psd_welch,
            "whitened_acf": whitened_acf,
            "whitened_acf_bound": float(whitened_acf_bound),
        }

    # ------------------------------------------------------------------
    # (d) Matplotlib diagnostic plots from the digest
    # ------------------------------------------------------------------

    def plot_digest(  # noqa: PLR0914
        self,
        figsize: tuple[float, float] = (13.0, 8.5),
        max_acf_lag: int | None = None,
    ) -> tuple[matplotlib.figure.Figure, np.ndarray]:
        """Generate a 4-panel Matplotlib diagnostic figure from `self.digest`.

        Panels:
          1. Direct Welch PSD vs. parametric AR(q) model PSD.
          2. Model autocorrelation `rho[k]` out to `max_acf_lag` (or `pulse_length`).
          3. Burg reflection coefficient magnitudes `|kappa_m|` and automatic order cutoff `q`.
          4. Out-of-sample whitened residual autocorrelation with 95% confidence bounds.

        Parameters
        ----------
        figsize : tuple[float, float], optional
            Figure width and height in inches, by default (13.0, 8.5)
        max_acf_lag : int | None, optional
            Maximum lag shown in the model autocorrelation panel; defaults to `self.pulse_length`
            when `None`, by default None

        Returns
        -------
        tuple[matplotlib.figure.Figure, np.ndarray]
            - `fig`: Created Matplotlib `Figure` instance.
            - `axes`: 2x2 NumPy object array of Matplotlib `Axes`.
        """
        d = self.digest
        q = d["selected_order"]
        p = d["pulse_length"]
        fs = d["sample_rate"]
        lag_lim = p if max_acf_lag is None else min(p, max_acf_lag)

        fig, axes = plt.subplots(2, 2, figsize=figsize, constrained_layout=True)
        title_suffix = f" ({self.label})" if self.label else ""
        fig.suptitle(
            f"AR({q}) Noise Model Digest{title_suffix} | "
            f"p={p}, RMS={d['rms_noise']:.3g}, Innov STD={d['innovation_std']:.3g}, "
            f"DynRange={d['spectral_dynamic_range_db']:.1f} dB",
            fontsize=12,
            fontweight="bold",
        )

        # Panel 1: Welch PSD vs AR(q) PSD
        ax_psd = axes[0, 0]
        f_w = d["freqs_welch"][1:]  # skip DC for log-x compatibility
        p_w = d["psd_welch"][1:]
        f_a = d["freqs_ar"][1:]
        p_a = d["psd_ar"][1:]
        ax_psd.loglog(f_w, p_w, color="0.55", alpha=0.8, lw=1.0, label="Welch PSD (direct)")
        ax_psd.loglog(f_a, p_a, color="C0", lw=1.8, label=f"AR({q}) Model PSD")
        ax_psd.axvline(fs / p, color="C3", ls="--", lw=1.0, alpha=0.8, label=f"Pulse rate fs/p ({fs / p:.3g} Hz)")
        ax_psd.set_title("One-Sided Power Spectral Density")
        ax_psd.set_xlabel("Frequency [Hz]")
        ax_psd.set_ylabel("PSD [units² / Hz]")
        ax_psd.grid(True, which="both", alpha=0.25)
        ax_psd.legend(loc="best", fontsize=8.5)

        # Panel 2: Model Autocorrelation out to length p
        ax_acf = axes[0, 1]
        lags = d["lags"][:lag_lim]
        acf = d["autocorr"][:lag_lim]
        ax_acf.plot(lags, acf, color="C0", lw=1.4, label="AR(q) Model ACF ρ[k]")
        if q < lag_lim:
            ax_acf.axvline(q, color="C1", ls="--", lw=1.2, label=f"Order cutoff q={q} (YW extrap. > q)")
        ax_acf.axhline(0.0, color="0.3", lw=0.8)
        ax_acf.set_title(f"Noise Autocorrelation (Lags 0 to {lag_lim - 1})")
        ax_acf.set_xlabel("Lag k [samples]")
        ax_acf.set_ylabel("Autocorrelation ρ[k]")
        ax_acf.grid(True, alpha=0.25)
        ax_acf.legend(loc="best", fontsize=8.5)

        # Panel 3: Reflection Coefficient Decay & Order Selection
        ax_kap = axes[1, 0]
        m_idx = np.arange(1, len(self.reflection_coeffs))
        abs_kap = np.abs(self.reflection_coeffs[1:])
        ax_kap.semilogy(m_idx, np.maximum(abs_kap, 1e-12), color="C2", lw=1.2, label="|κ_m|")
        ax_kap.axhline(
            d["noise_floor_kappa"],
            color="0.4",
            ls=":",
            lw=1.1,
            label=f"1/√N_eff ({d['noise_floor_kappa']:.2e})",
        )
        ax_kap.axhline(
            3.6 * d["noise_floor_kappa"],
            color="C3",
            ls="--",
            lw=1.0,
            label="3.6σ spike threshold",
        )
        ax_kap.axvline(q, color="C1", ls="-.", lw=1.3, label=f"Selected q = {q}")
        ax_kap.set_title("Burg Reflection Coefficient Selection")
        ax_kap.set_xlabel("Model Order m")
        ax_kap.set_ylabel("|κ_m|")
        ax_kap.grid(True, which="both", alpha=0.25)
        ax_kap.legend(loc="best", fontsize=8.5)

        # Panel 4: Whitened Residual Autocorrelation
        ax_res = axes[1, 1]
        w_acf = d["whitened_acf"][1:]
        w_lags = np.arange(1, len(d["whitened_acf"]))
        bound = d["whitened_acf_bound"]
        ax_res.plot(w_lags, w_acf, color="C4", lw=1.0, label="Whitened residual ACF")
        ax_res.axhspan(-bound, bound, color="C2", alpha=0.18, label=f"95% white bound (±{bound:.3f})")
        ax_res.axhline(0.0, color="0.3", lw=0.8)
        ax_res.set_title("Out-of-Sample Whitened Residual ACF")
        ax_res.set_xlabel("Lag k [samples]")
        ax_res.set_ylabel("Residual ρ_white[k]")
        ax_res.grid(True, alpha=0.25)
        ax_res.legend(loc="best", fontsize=8.5)

        return fig, axes

    # ------------------------------------------------------------------
    # (e) & (f) Multi-row Parquet & Arrow IPC persistence
    # ------------------------------------------------------------------

    @staticmethod
    def _arrow_schema() -> pa.Schema:
        """Return the Apache Arrow schema used for Parquet and IPC multi-row serialization.

        Returns
        -------
        pa.Schema
            Arrow schema containing scalar metadata/summary columns and `list<float64>` /
            `list<int64>` array columns for model parameters and digest curves.
        """
        return pa.schema([
            ("model_id", pa.string()),
            ("label", pa.string()),
            ("created_at", pa.string()),
            ("metadata_json", pa.string()),
            ("pulse_length", pa.int64()),
            ("selected_order", pa.int64()),
            ("max_order", pa.int64()),
            ("sample_rate", pa.float64()),
            ("n_sequences", pa.int64()),
            ("sequence_length", pa.int64()),
            ("project_offset", pa.bool_()),
            ("calibration_alpha", pa.float64()),
            ("rms_noise", pa.float64()),
            ("innovation_std", pa.float64()),
            ("spectral_dynamic_range_db", pa.float64()),
            ("noise_floor_kappa", pa.float64()),
            ("ar_coeffs", pa.list_(pa.float64())),
            ("innovation_var", pa.float64()),
            ("reflection_coeffs", pa.list_(pa.float64())),
            ("innovation_vars_by_order", pa.list_(pa.float64())),
            ("w_hat", pa.list_(pa.float64())),
            ("digest_lags", pa.list_(pa.int64())),
            ("digest_autocov", pa.list_(pa.float64())),
            ("digest_autocorr", pa.list_(pa.float64())),
            ("digest_freqs_ar", pa.list_(pa.float64())),
            ("digest_psd_ar", pa.list_(pa.float64())),
            ("digest_freqs_welch", pa.list_(pa.float64())),
            ("digest_psd_welch", pa.list_(pa.float64())),
            ("digest_whitened_acf", pa.list_(pa.float64())),
            ("digest_whitened_acf_bound", pa.float64()),
        ])

    def to_row_dict(self) -> dict[str, Any]:
        """Serialize this whitener and its full digest into a single Arrow-compatible row dict.

        Returns
        -------
        dict[str, Any]
            Dictionary mapping each column name in `_arrow_schema()` to Python primitives or
            lists ready for `pa.Table.from_pylist`.
        """
        d = self.digest
        return {
            "model_id": self.model_id,
            "label": self.label,
            "created_at": self.created_at,
            "metadata_json": json.dumps(self.metadata),
            "pulse_length": self.pulse_length,
            "selected_order": self.selected_order,
            "max_order": self.max_order,
            "sample_rate": self.sample_rate,
            "n_sequences": self.n_sequences,
            "sequence_length": self.sequence_length,
            "project_offset": self.project_offset,
            "calibration_alpha": float(self.calibration_alpha),
            "rms_noise": float(d["rms_noise"]),
            "innovation_std": float(d["innovation_std"]),
            "spectral_dynamic_range_db": float(d["spectral_dynamic_range_db"]),
            "noise_floor_kappa": float(d["noise_floor_kappa"]),
            "ar_coeffs": self.ar_coeffs.tolist(),
            "innovation_var": float(self.innovation_var),
            "reflection_coeffs": self.reflection_coeffs.tolist(),
            "innovation_vars_by_order": self.innovation_vars_by_order.tolist(),
            "w_hat": self.w_hat.tolist() if self.w_hat is not None else [],
            "digest_lags": d["lags"].tolist(),
            "digest_autocov": d["autocov"].tolist(),
            "digest_autocorr": d["autocorr"].tolist(),
            "digest_freqs_ar": d["freqs_ar"].tolist(),
            "digest_psd_ar": d["psd_ar"].tolist(),
            "digest_freqs_welch": d["freqs_welch"].tolist(),
            "digest_psd_welch": d["psd_welch"].tolist(),
            "digest_whitened_acf": d["whitened_acf"].tolist(),
            "digest_whitened_acf_bound": float(d["whitened_acf_bound"]),
        }

    @classmethod
    def from_row_dict(cls, row: dict[str, Any]) -> ARNoiseWhitener:
        """Reconstruct an `ARNoiseWhitener` instance from a deserialized row dictionary.

        Parameters
        ----------
        row : dict[str, Any]
            Row dictionary matching the schema returned by `to_row_dict()`.

        Returns
        -------
        ARNoiseWhitener
            Reconstructed whitener with operator filters and `digest` restored.
        """
        obj = object.__new__(cls)
        obj.model_id = str(row["model_id"])
        obj.label = str(row["label"])
        obj.created_at = str(row["created_at"])
        obj.metadata = json.loads(row["metadata_json"]) if row.get("metadata_json") else {}
        obj.pulse_length = int(row["pulse_length"])
        obj.selected_order = int(row["selected_order"])
        obj.max_order = int(row["max_order"])
        obj.sample_rate = float(row["sample_rate"])
        obj.n_sequences = int(row["n_sequences"])
        obj.sequence_length = int(row["sequence_length"])
        obj.project_offset = bool(row["project_offset"])
        obj.calibration_alpha = float(row["calibration_alpha"])
        obj.noise_floor_kappa = float(row["noise_floor_kappa"])

        obj.ar_coeffs = np.asarray(row["ar_coeffs"], dtype=np.float64)
        obj.innovation_var = float(row["innovation_var"])
        obj.reflection_coeffs = np.asarray(row["reflection_coeffs"], dtype=np.float64)
        obj.innovation_vars_by_order = np.asarray(row["innovation_vars_by_order"], dtype=np.float64)

        # Rebuild step-down startup filters and causal FIR taps
        obj._build_operator_state()
        if obj.project_offset and len(row["w_hat"]) == obj.pulse_length:
            obj.w_hat = np.asarray(row["w_hat"], dtype=np.float64)

        obj.digest = {
            "selected_order": obj.selected_order,
            "max_order": obj.max_order,
            "pulse_length": obj.pulse_length,
            "sample_rate": obj.sample_rate,
            "calibration_alpha": obj.calibration_alpha,
            "rms_noise": float(row["rms_noise"]),
            "innovation_std": float(row["innovation_std"]),
            "spectral_dynamic_range_db": float(row["spectral_dynamic_range_db"]),
            "noise_floor_kappa": obj.noise_floor_kappa,
            "lags": np.asarray(row["digest_lags"], dtype=np.int64),
            "autocov": np.asarray(row["digest_autocov"], dtype=np.float64),
            "autocorr": np.asarray(row["digest_autocorr"], dtype=np.float64),
            "freqs_ar": np.asarray(row["digest_freqs_ar"], dtype=np.float64),
            "psd_ar": np.asarray(row["digest_psd_ar"], dtype=np.float64),
            "freqs_welch": np.asarray(row["digest_freqs_welch"], dtype=np.float64),
            "psd_welch": np.asarray(row["digest_psd_welch"], dtype=np.float64),
            "whitened_acf": np.asarray(row["digest_whitened_acf"], dtype=np.float64),
            "whitened_acf_bound": float(row["digest_whitened_acf_bound"]),
        }
        return obj

    @staticmethod
    def _infer_format(path: Path, fmt: Literal["parquet", "ipc"] | None) -> Literal["parquet", "ipc"]:
        """Resolve the storage format (`"parquet"` or `"ipc"`) from an explicit flag or file suffix.

        Parameters
        ----------
        path : Path
            Target file path.
        fmt : {"parquet", "ipc"} | None
            Explicit format override, or `None` to infer from `path.suffix`.

        Returns
        -------
        Literal["parquet", "ipc"]
            Resolved serialization format.
        """
        if fmt is not None:
            return fmt
        suffix = path.suffix.lower()
        if suffix in {".arrow", ".ipc", ".feather"}:
            return "ipc"
        return "parquet"

    @classmethod
    def _read_table(cls, path: Path, fmt: Literal["parquet", "ipc"]) -> pa.Table:
        """Read a `pyarrow.Table` from a Parquet or Arrow IPC file on disk.

        Parameters
        ----------
        path : Path
            Path to the existing archive file.
        fmt : {"parquet", "ipc"}
            Storage format of the file.

        Returns
        -------
        pa.Table
            Loaded Arrow table containing one or more serialized model rows.
        """
        if fmt == "parquet":
            return pq.read_table(path)
        with pa.OSFile(str(path), "rb") as source:
            with pa.ipc.open_file(source) as reader:
                return reader.read_all()

    @classmethod
    def _write_table_atomic(cls, table: pa.Table, path: Path, fmt: Literal["parquet", "ipc"]) -> None:
        """Atomically write a `pyarrow.Table` to `path` via a temporary file and `os.replace`.

        Parameters
        ----------
        table : pa.Table
            Arrow table to persist.
        path : Path
            Destination file path.
        fmt : {"parquet", "ipc"}
            Target serialization format (`"parquet"` with zstd compression or `"ipc"`).
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
        os.close(fd)
        tmp_path = Path(tmp_name)
        try:  # ruff: ignore[too-many-statements-in-try-clause]
            if fmt == "parquet":
                pq.write_table(table, tmp_path, compression="zstd")
            else:
                with pa.OSFile(str(tmp_path), "wb") as sink:
                    with pa.ipc.new_file(sink, table.schema) as writer:
                        writer.write_table(table)
            os.replace(tmp_path, path)
        except Exception:
            if tmp_path.exists():
                tmp_path.unlink()
            raise

    def save(
        self,
        path: str | Path,
        fmt: Literal["parquet", "ipc"] | None = None,
        append: bool = True,
    ) -> None:
        """Store this model as a row in a Parquet or Arrow IPC file.

        Parameters
        ----------
        path : str | Path
            Destination file path.
        fmt : {"parquet", "ipc"} | None, optional
            Storage format (`"parquet"` or `"ipc"`), or `None` to infer from the file extension
            (`.arrow`, `.ipc`, `.feather` -> `"ipc"`; otherwise `"parquet"`), by default None
        append : bool, optional
            If `True` and `path` already exists, concatenate this row onto existing rows; if
            `False`, overwrite `path`, by default True
        """
        self.save_many([self], path=path, fmt=fmt, append=append)

    @classmethod
    def save_many(
        cls,
        models: Iterable[ARNoiseWhitener],
        path: str | Path,
        fmt: Literal["parquet", "ipc"] | None = None,
        append: bool = False,
    ) -> None:
        """Store multiple `ARNoiseWhitener` instances as rows in a single Parquet or Arrow IPC file.

        Parameters
        ----------
        models : Iterable[ARNoiseWhitener]
            Iterable of `ARNoiseWhitener` objects to serialize. If empty, no file is written.
        path : str | Path
            Destination file path.
        fmt : {"parquet", "ipc"} | None, optional
            Storage format (`"parquet"` or `"ipc"`), or `None` to infer from `path`, by default None
        append : bool, optional
            If `True` and `path` already exists, append new rows to existing rows; if `False`,
            overwrite `path`, by default False
        """
        target = Path(path)
        resolved_fmt = cls._infer_format(target, fmt)
        rows = [m.to_row_dict() for m in models]
        if not rows:
            return

        new_table = pa.Table.from_pylist(rows, schema=cls._arrow_schema())
        if append and target.exists():
            existing = cls._read_table(target, resolved_fmt)
            combined = pa.concat_tables([existing, new_table])
        else:
            combined = new_table

        cls._write_table_atomic(combined, target, resolved_fmt)

    @classmethod
    def load(
        cls,
        path: str | Path,
        row: int = -1,
        model_id: str | None = None,
        label: str | None = None,
        fmt: Literal["parquet", "ipc"] | None = None,
    ) -> ARNoiseWhitener:
        """Recover a single `ARNoiseWhitener` from a Parquet or Arrow IPC file.

        Lookup precedence: `model_id` (exact match) -> `label` (most recent match) -> `row` index.

        Parameters
        ----------
        path : str | Path
            Path to the Parquet or Arrow IPC file.
        row : int, optional
            Zero-based row index (supports negative indexing) used when neither `model_id` nor
            `label` is given, by default -1
        model_id : str | None, optional
            Exact `model_id` UUID string to select, by default None
        label : str | None, optional
            Exact `label` string to select (returns the last matching row if multiple rows share
            the same label), by default None
        fmt : {"parquet", "ipc"} | None, optional
            Storage format override, or `None` to infer from `path`, by default None

        Returns
        -------
        ARNoiseWhitener
            Reconstructed `ARNoiseWhitener` instance.

        Raises
        ------
        KeyError
            If `model_id` or `label` is specified and no matching row exists in the file.
        IndexError
            If `row` is out of bounds for the number of rows in the file.
        """
        target = Path(path)
        resolved_fmt = cls._infer_format(target, fmt)
        table = cls._read_table(target, resolved_fmt)

        if model_id is not None:
            ids: list[str] = table["model_id"].to_pylist()
            try:
                idx = ids.index(model_id)
            except ValueError as exc:
                raise KeyError(f"No model found with model_id='{model_id}'.") from exc
            selected = table.slice(idx, 1).to_pylist()[0]
        elif label is not None:
            labels: list[str] = table["label"].to_pylist()
            matches = [i for i, val in enumerate(labels) if val == label]
            if not matches:
                raise KeyError(f"No model found with label='{label}'.")
            selected = table.slice(matches[-1], 1).to_pylist()[0]
        else:
            n = table.num_rows
            idx = row if row >= 0 else n + row
            if not (0 <= idx < n):
                raise IndexError(f"Row index {row} out of range for file with {n} rows.")
            selected = table.slice(idx, 1).to_pylist()[0]

        return cls.from_row_dict(selected)

    @classmethod
    def load_all(
        cls,
        path: str | Path,
        fmt: Literal["parquet", "ipc"] | None = None,
    ) -> list[ARNoiseWhitener]:
        """Recover all stored `ARNoiseWhitener` rows from a Parquet or Arrow IPC file.

        Parameters
        ----------
        path : str | Path
            Path to the Parquet or Arrow IPC file.
        fmt : {"parquet", "ipc"} | None, optional
            Storage format override, or `None` to infer from `path`, by default None

        Returns
        -------
        list[ARNoiseWhitener]
            List of reconstructed `ARNoiseWhitener` instances in row order.
        """
        target = Path(path)
        resolved_fmt = cls._infer_format(target, fmt)
        table = cls._read_table(target, resolved_fmt)
        return [cls.from_row_dict(r) for r in table.to_pylist()]
