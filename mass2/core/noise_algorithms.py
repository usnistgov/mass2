import numpy as np
import scipy as sp
import pylab as plt  # type: ignore
import mass2
from dataclasses import dataclass
from numpy import ndarray
from numpy.typing import NDArray, ArrayLike
from collections.abc import Callable


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
        The mean autocorrelation of the rows ("traces") in the input `data`, from lags `[0, nsamples-1]`.
    """
    data = np.asarray(data)
    ntraces, nsamples = data.shape
    ac = np.zeros(nsamples, dtype=float)

    traces_used = 0
    for i in range(ntraces):
        pulse = data[i, :] - data[i, :].mean()
        if np.abs(pulse).max() > max_excursion:
            continue
        ac += np.correlate(pulse, pulse, "full")[nsamples - 1 :]
        traces_used += 1

    ac /= traces_used
    ac /= nsamples - np.arange(nsamples, dtype=float)
    return ac


def calc_continuous_autocorrelation(data: ArrayLike, n_lags: int, max_excursion: int = 1000) -> NDArray:
    """Calculate the autocorrelation of the input data, assuming the entire array is continuous.

    Parameters
    ----------
    data : ArrayLike
        Data to be autocorrelated. Arrays of 2+ dimensions will be converted to a 1D array via `.ravel()`.
    n_lags : int
        Compute the autocorrelation for lags in the range `[0, n_lags-1]`.
    max_excursion : int, optional
        Chunks of data with max absolute excursion from the mean this large will be excluded from the calculation, by default 1000

    Returns
    -------
    NDArray
        The autocorrelation array

    Raises
    ------
    ValueError
        If the data are too short to provide the requested number of lags, or the data contain apparent pulses.
    """
    data = np.asarray(data).ravel()
    n_data = len(data)
    assert n_lags < n_data

    def padded_length(n):
        """Return a sensible number in the range [n, 2n] which is not too
        much larger than n, yet is good for FFTs.

        Returns:
            A number: (1, 3, or 5)*(a power of two), whichever is smallest.
        """
        pow2 = np.round(2 ** np.ceil(np.log2(n)))
        if n == pow2:
            return int(n)
        elif n > 0.75 * pow2:
            return int(pow2)
        elif n > 0.625 * pow2:
            return int(np.round(0.75 * pow2))
        else:
            return int(np.round(0.625 * pow2))

    # When there are 10 million data points and only 10,000 lags wanted,
    # it's hugely inefficient to compute the full autocorrelation, especially
    # in memory.  Instead, compute it on chunks several times the length of the desired
    # correlation, and average.
    CHUNK_MULTIPLE = 15
    if n_data < CHUNK_MULTIPLE * n_lags:
        msg = f"There are too few data values ({n_data=}) to compute at least {n_lags} lags."
        raise ValueError(msg)

    # Be sure to pad chunksize samples by AT LEAST n_lags zeros, to prevent
    # unwanted wraparound in the autocorrelation.
    # padded_data is what we do DFT/InvDFT on; ac is the unnormalized output.
    chunksize = CHUNK_MULTIPLE * n_lags
    padsize = n_lags
    padded_data = np.zeros(padded_length(padsize + chunksize), dtype=float)

    ac = np.zeros(n_lags, dtype=float)

    entries = 0

    Nchunks = n_data // chunksize
    datachunks = data[: Nchunks * chunksize].reshape(Nchunks, chunksize)
    for data in datachunks:
        padded_data[:chunksize] = data - np.asarray(data).mean()
        if np.abs(padded_data).max() > max_excursion:
            continue

        ft = np.fft.rfft(padded_data)
        ft[0] = 0  # this redundantly removes the mean of the data set
        power = (ft * ft.conj()).real
        acsum = np.fft.irfft(power)
        ac += acsum[:n_lags]
        entries += 1

    if entries == 0:
        raise ValueError("Apparently all 'noise' chunks had large excursions from baseline, so no autocorrelation was computed")

    ac /= entries
    ac /= np.arange(chunksize, chunksize - n_lags + 0.5, -1.0, dtype=float)
    return ac


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


def psd_2d(Nt: ndarray, dt: float) -> ndarray:
    # Nt is size (n,m) with m records of length n
    (n, _m) = Nt.shape
    df = 1 / n / dt  # the frequency bin spacing of the rfft
    # take the absolute value of the rfft of each record, then average all records
    Nabs = np.mean(np.abs(np.fft.rfft(Nt, axis=0)), axis=1)
    # PSD = 2*Nabs^2/n/df
    # the 2 accounts for the power that would be in the negative frequency bins, due to use of rfft
    # n comes from parseval's theorm
    # df normalizes binsize since rfft doesn't know the bin size
    psd = 2 * Nabs**2 / n / df
    # Handle the DC component and Nyquist frequency differently (no factor of 2)
    psd[0] /= 2  # DC component
    if n % 2 == 0:  # If even number of samples
        psd[-1] /= 2  # Nyquist frequency
    return psd


def calc_psd_frequencies(nbins: int, dt: float) -> ndarray:
    return np.arange(nbins, dtype=float) / (2 * dt * nbins)


def noise_psd_periodogram(data: ndarray, dt: float, window="boxcar", detrend=False) -> "NoiseResult":
    f, Pxx = sp.signal.periodogram(data, fs=1 / dt, window=window, axis=-1, detrend=detrend)
    # len(f) = data.shape[1]//2+1
    # Pxx[i, j] is the PSD at frequency f[j] for the i‑th trace data[i, :]
    Pxx_mean = np.mean(Pxx, axis=0)
    # Pxx_mean[j] is the averaged PSD at frequency f[j] over all traces
    autocorr_vec = calc_discontinuous_autocorrelation(data)
    return NoiseResult(psd=Pxx_mean, autocorr_vec=autocorr_vec, frequencies=f)


def calc_noise_result(
    data: ArrayLike, dt: float, continuous: bool, window: Callable | None = None, skip_autocorr_if_length_over: int = 10000
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

    Returns
    -------
    NoiseResult
        The derived noise spectrum and autocorrelation
    """
    data = np.asarray(data)
    data_zeromean = data - np.mean(data)
    (n_pulses, nsamples) = data_zeromean.shape
    # see test_ravel_behavior to be sure this is written correctly
    f_mass, psd_mass = mass2.mathstat.power_spectrum.computeSpectrum(data_zeromean.ravel(), segfactor=n_pulses, dt=dt, window=window)
    if nsamples <= skip_autocorr_if_length_over:
        if continuous:
            autocorr_vec = calc_continuous_autocorrelation(data_zeromean.ravel(), n_lags=nsamples)
        else:
            autocorr_vec = calc_discontinuous_autocorrelation(data_zeromean)
    else:
        print(
            """warning: noise_psd_mass skipping autocorrelation calculation for long traces,
            use skip_autocorr_if_length_over argument to override this"""
        )
        autocorr_vec = None
    # nbins = len(psd_mass)
    # frequencies = calc_psd_frequencies(nbins, dt)
    return NoiseResult(psd=psd_mass, autocorr_vec=autocorr_vec, frequencies=f_mass)


@dataclass
class NoiseResult:
    psd: np.ndarray
    autocorr_vec: np.ndarray | None
    frequencies: np.ndarray

    def plot(
        self,
        axis: plt.Axes | None = None,
        arb_to_unit_scale_and_label: tuple[int, str] = (1, "arb"),
        sqrt_psd: bool = True,
        loglog: bool = True,
        **plotkwarg,
    ):
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
        axis.figure.tight_layout()

    def plot_log_rebinned(
        self,
        bins_per_decade: int = 10,
        axis: plt.Axes | None = None,
        arb_to_unit_scale_and_label: tuple[int, str] = (1, "arb"),
        sqrt_psd: bool = True,
        **plotkwarg,
    ):
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
        binned_freqs = []
        binned_psd = []
        for i in range(1, len(bin_edges)):
            mask = inds == i
            if np.any(mask):
                binned_freqs.append(np.exp(np.mean(np.log(freq[mask]))))  # geometric mean
                binned_psd.append(np.mean(psd[mask]))

        binned_freqs = np.array(binned_freqs)
        binned_psd = np.array(binned_psd)

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
        axis.figure.tight_layout()
