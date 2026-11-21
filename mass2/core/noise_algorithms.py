"""
Algorithms to analyze noise data.
"""

import warnings
import numpy as np
import scipy as sp
import pylab as plt  # type: ignore
import mass2
from dataclasses import dataclass
from numpy import ndarray
from numpy.typing import NDArray, ArrayLike
from collections.abc import Callable
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
) -> "NoiseResult":
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
) -> "NoiseResult":
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
        **plotkwarg: dict,
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
            plt.loglog()
        axis.grid()
        axis.set_xlabel("Frequency (Hz)")
        plt.title(f"noise from records of length {len(self.frequencies) * 2 - 2}")

    def plot_log_rebinned(
        self,
        bins_per_decade: int = 10,
        axis: plt.Axes | None = None,
        arb_to_unit_scale_and_label: tuple[int, str] = (1, "arb"),
        sqrt_psd: bool = True,
        **plotkwarg: dict,
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
