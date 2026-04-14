"""
mass2.core.analysis_algorithms - main algorithms used in data analysis

Designed to abstract certain key algorithms out of the class `MicrocalDataSet`
and be able to run them fast.

Created on Jun 9, 2014

@author: fowlerj
"""

import numpy as np
import polars as pl
import scipy as sp

from collections.abc import Callable
from dataclasses import dataclass
from numpy.typing import NDArray, ArrayLike
from numba import njit
from typing import Any

from mass2.mathstat.entropy import laplace_entropy
from mass2.mathstat.interpolate import CubicSpline
import logging

LOG = logging.getLogger("mass")


########################################################################################
# Pulse summary quantities


@njit
def estimateRiseTime(pulse_data: ArrayLike, timebase: float, nPretrig: int) -> NDArray:
    """Computes the rise time of timeseries <pulse_data>, where the time steps are <timebase>.
    <pulse_data> can be a 2D array where each row is a different pulse record, in which case
    the return value will be an array last long as the number of rows in <pulse_data>.

    If nPretrig >= 4, then the samples pulse_data[:nPretrig] are averaged to estimate
    the baseline.  Otherwise, the minimum of pulse_data is assumed to be the baseline.

    Specifically, take the first and last of the rising points in the range of
    10% to 90% of the peak value, interpolate a line between the two, and use its
    slope to find the time to rise from 0 to the peak.

    Args:
        pulse_data: An np.ndarray of dimension 1 (a single pulse record) or 2 (an
            array with each row being a pulse record).
        timebase: The sampling time.
        nPretrig: The number of samples that are recorded before the trigger.

    Returns:
        An ndarray of dimension 1, giving the rise times.
    """
    MINTHRESH, MAXTHRESH = 0.1, 0.9

    # If pulse_data is a 1D array, turn it into 2
    pulse_data = np.asarray(pulse_data)
    ndim = len(pulse_data.shape)
    if ndim > 2 or ndim < 1:
        raise ValueError("input pulse_data should be a 1d or 2d array.")
    if ndim == 1:
        pulse_data.shape = (1, pulse_data.shape[0])

    # The following requires a lot of numpy foo to read. Sorry!
    if nPretrig >= 4:
        baseline_value = pulse_data[:, 0:nPretrig].mean(axis=1)
    else:
        baseline_value = pulse_data.min(axis=1)
        nPretrig = 0
    value_at_peak = pulse_data.max(axis=1) - baseline_value
    idx_last_pk = pulse_data.argmax(axis=1).max()

    npulses = pulse_data.shape[0]
    try:
        rising_data = (pulse_data[:, nPretrig : idx_last_pk + 1] - baseline_value[:, np.newaxis]) / value_at_peak[:, np.newaxis]
        # Find the last and first indices at which the data are in (0.1, 0.9] times the
        # peak value. Then make sure last is at least 1 past first.
        last_idx = (rising_data > MAXTHRESH).argmax(axis=1) - 1
        first_idx = (rising_data > MINTHRESH).argmax(axis=1)
        last_idx[last_idx < first_idx] = first_idx[last_idx < first_idx] + 1
        last_idx[last_idx == rising_data.shape[1]] = rising_data.shape[1] - 1

        pulsenum = np.arange(npulses)
        y_diff = np.asarray(rising_data[pulsenum, last_idx] - rising_data[pulsenum, first_idx], dtype=float)
        y_diff[y_diff < timebase] = timebase
        time_diff = timebase * (last_idx - first_idx)
        rise_time = time_diff / y_diff
        rise_time[y_diff <= 0] = -9.9e-6
        return rise_time

    except ValueError:
        return -9.9e-6 + np.zeros(npulses, dtype=float)


def compute_max_deriv(
    pulse_data: ArrayLike, ignore_leading: int, spike_reject: bool = True, kernel: ArrayLike | str | None = None
) -> NDArray:
    """Computes the maximum derivative in timeseries <pulse_data>.
    <pulse_data> can be a 2D array where each row is a different pulse record, in which case
    the return value will be an array last long as the number of rows in <pulse_data>.

    Args:
        pulse_data:
        ignore_leading:
        spike_reject: (default True)
        kernel: the linear filter against which the signals will be convolved
            (CONVOLED, not correlated, so reverse the filter as needed). If None,
            then the default kernel of [+.2 +.1 0 -.1 -.2] will be used. If
            "SG", then the cubic 5-point Savitzky-Golay filter will be used (see
            below). Otherwise, kernel needs to be a (short) array which will
            be converted to a 1xN 2-dimensional np.ndarray. (default None)

    Returns:
        An np.ndarray, dimension 1: the value of the maximum derivative (units of <pulse_data units> per sample).

    When kernel=="SG", then we estimate the derivative by Savitzky-Golay filtering
    (with 1 point before/3 points after the point in question and fitting polynomial
    of order 3).  Find the right general area by first doing a simple difference.
    """

    # If pulse_data is a 1D array, turn it into 2
    pulse_data = np.asarray(pulse_data)
    ndim = len(pulse_data.shape)
    if ndim > 2 or ndim < 1:
        raise ValueError("input pulse_data should be a 1d or 2d array.")
    if ndim == 1:
        pulse_data.shape = (1, pulse_data.shape[0])
    pulse_view = pulse_data[:, ignore_leading:]
    NPulse = pulse_view.shape[0]
    NSamp = pulse_view.shape[1]

    # The default filter:
    filter_coef = np.array([+0.2, +0.1, 0, -0.1, -0.2])
    if kernel == "SG":
        # This filter is the Savitzky-Golay filter of n_L=1, n_R=3 and M=3, to use the
        # language of Numerical Recipes 3rd edition.  It amounts to least-squares fitting
        # of an M=3rd order polynomial to the five points [-1,+3] and
        # finding the slope of the polynomial at 0.
        # Note that we reverse the order of coefficients because convolution will re-reverse
        filter_coef = np.array([-0.45238, -0.02381, 0.28571, 0.30952, -0.11905])[::-1]

    elif kernel is not None:
        filter_coef = np.array(kernel).ravel()

    f0, f1, f2, f3, f4 = filter_coef

    max_deriv = np.zeros(NPulse, dtype=np.float64)

    if spike_reject:
        for i in range(NPulse):
            pulses = pulse_view[i]
            t0 = f4 * pulses[0] + f3 * pulses[1] + f2 * pulses[2] + f1 * pulses[3] + f0 * pulses[4]
            t1 = f4 * pulses[1] + f3 * pulses[2] + f2 * pulses[3] + f1 * pulses[4] + f0 * pulses[5]
            t2 = f4 * pulses[2] + f3 * pulses[3] + f2 * pulses[4] + f1 * pulses[5] + f0 * pulses[6]
            t_max_deriv = t2 if t2 < t0 else t0

            for j in range(7, NSamp):
                t3 = f4 * pulses[j - 4] + f3 * pulses[j - 3] + f2 * pulses[j - 2] + f1 * pulses[j - 1] + f0 * pulses[j]
                t4 = t3 if t3 < t1 else t1
                t_max_deriv = max(t4, t_max_deriv)

                t0, t1, t2 = t1, t2, t3

            max_deriv[i] = t_max_deriv
    else:
        for i in range(NPulse):
            pulses = pulse_view[i]
            t0 = f4 * pulses[0] + f3 * pulses[1] + f2 * pulses[2] + f1 * pulses[3] + f0 * pulses[4]
            t_max_deriv = t0

            for j in range(5, NSamp):
                t0 = f4 * pulses[j - 4] + f3 * pulses[j - 3] + f2 * pulses[j - 2] + f1 * pulses[j - 1] + f0 * pulses[j]
                t_max_deriv = max(t0, t_max_deriv)
            max_deriv[i] = t_max_deriv

    return np.asarray(max_deriv, dtype=np.float32)


########################################################################################
# Drift correction and related algorithms


class HistogramSmoother:
    """Object that can repeatedly smooth histograms with the same bin count and
    width to the same Gaussian width.  By pre-computing the smoothing kernel for
    that histogram, we can smooth multiple histograms with the same geometry.
    """

    def __init__(self, smooth_sigma: float, limits: ArrayLike):
        """Give the smoothing Gaussian's width as <smooth_sigma> and the
        [lower,upper] histogram limits as <limits>."""

        self.limits = tuple(np.asarray(limits, dtype=float))
        self.smooth_sigma = smooth_sigma

        # Choose a reasonable # of bins, at least 1024 and a power of 2
        stepsize = 0.4 * smooth_sigma
        dlimits = self.limits[1] - self.limits[0]
        nbins_guess = int(dlimits / stepsize + 0.5)
        min_nbins = 1024
        max_nbins = 32768  # 32k bins, 2**15

        # Clamp nbins_guess to at least min_nbins
        clamped_nbins = np.clip(nbins_guess, min_nbins, max_nbins)
        nbins_forced_to_power_of_2 = int(2 ** np.ceil(np.log2(clamped_nbins)))
        # if nbins_forced_to_power_of_2 == max_nbins:
        #     print(f"Warning: HistogramSmoother (for drift correct) Limiting histogram bins to {max_nbins} (requested {nbins_guess})")
        self.nbins = nbins_forced_to_power_of_2
        self.stepsize = dlimits / self.nbins

        # Compute the Fourier-space smoothing kernel
        kernel = np.exp(-0.5 * (np.arange(self.nbins) * self.stepsize / self.smooth_sigma) ** 2)
        kernel[1:] += kernel[-1:0:-1]  # Handle the negative frequencies
        kernel /= kernel.sum()
        self.kernel_ft = np.fft.rfft(kernel)

    def __call__(self, values: ArrayLike) -> NDArray:
        """Return a smoothed histogram of the data vector <values>"""
        contents, _ = np.histogram(values, self.nbins, self.limits)
        ftc = np.fft.rfft(contents)
        csmooth = np.fft.irfft(self.kernel_ft * ftc)
        csmooth[csmooth < 0] = 0
        return csmooth


@njit
def make_smooth_histogram(values: ArrayLike, smooth_sigma: float, limit: float, upper_limit: float | None = None) -> NDArray:
    """Convert a vector of arbitrary <values> info a smoothed histogram by
    histogramming it and smoothing.

    This is a convenience function using the HistogramSmoother class.

    Args:
        values: The vector of data to be histogrammed.
        smooth_sigma: The smoothing Gaussian's width (FWHM)
        limit, upper_limit: The histogram limits are [limit,upper_limit] or
            [0,limit] if upper_limit is None.

    Returns:
        The smoothed histogram as an array.
    """
    if upper_limit is None:
        limit, upper_limit = 0, limit
    return HistogramSmoother(smooth_sigma, [limit, upper_limit])(values)


def drift_correct(indicator: ArrayLike, uncorrected: ArrayLike, limit: float | None = None) -> tuple[float, dict]:
    """Compute a drift correction that minimizes the spectral entropy.

    Args:
        indicator: The "x-axis", which indicates the size of the correction.
        uncorrected: A filtered pulse height vector. Same length as indicator.
            Assumed to have some gain that is linearly related to indicator.
        limit: The upper limit of uncorrected values over which entropy is
            computed (default None).

    Generally indicator will be the pretrigger mean of the pulses, but you can
    experiment with other choices.

    The entropy will be computed on corrected values only in the range
    [0, limit], so limit should be set to a characteristic large value of
    uncorrected. If limit is None (the default), then in will be compute as
    25% larger than the 99%ile point of uncorrected.

    The model is that the filtered pulse height PH should be scaled by (1 +
    a*PTM) where a is an arbitrary parameter computed here, and PTM is the
    difference between each record's pretrigger mean and the median value of all
    pretrigger means. (Or replace "pretrigger mean" with whatever quantity you
    passed in as <indicator>.)
    """
    uncorrected = np.asarray(uncorrected)
    indicator = np.array(indicator)  # make a copy
    ptm_offset = np.median(indicator)
    indicator -= ptm_offset

    if limit is None:
        pct99 = np.percentile(uncorrected, 99)
        limit = 1.25 * pct99

    smoother = HistogramSmoother(0.5, [0, limit])
    assert smoother.nbins < 1e6, "will be crazy slow, should not be possible"

    def entropy(param: NDArray, indicator: NDArray, uncorrected: NDArray, smoother: HistogramSmoother) -> float:
        """Return the entropy of the drift-corrected values"""
        corrected = uncorrected * (1 + indicator * param)
        hsmooth = smoother(corrected)
        w = hsmooth > 0
        return -(np.log(hsmooth[w]) * hsmooth[w]).sum()

    drift_corr_param = sp.optimize.brent(entropy, (indicator, uncorrected, smoother), brack=[0, 0.001])

    drift_correct_info = {"type": "ptmean_gain", "slope": drift_corr_param, "median_pretrig_mean": ptm_offset}
    return drift_corr_param, drift_correct_info


@dataclass(frozen=True)
class ExtTriggerControl:
    """Parameters to control what columns are added when we incorporate external trigger timing info"""

    ms_nearest_trig: bool = True
    ms_last_trig: bool = False
    ms_next_trig: bool = False
    sf_last_trig: bool = False
    sf_next_trig: bool = False
    absolute_sfs: bool = False

    @property
    def require_last(self) -> bool:
        return self.ms_last_trig or self.sf_last_trig or self.ms_nearest_trig or self.absolute_sfs

    @property
    def require_next(self) -> bool:
        return self.ms_next_trig or self.sf_next_trig or self.ms_nearest_trig or self.absolute_sfs


def add_external_trigger(
    df: pl.DataFrame, df_ext: pl.DataFrame, output_control: ExtTriggerControl, subframediv: int, frametime_s: float
) -> pl.DataFrame:
    """Add external trigger info to a pulse dataframe, with columns of user's choice.

    Args:
        df (pl.DataFrame): The dataframe where each row is a pulse. Contains a "subframecount" column
        df_ext (pl.DataFrame): The dataframe containing external trigger times in a "subframecount" column
        output_control (ExtTriggerControl): Several fields that choose what columns to compute
        subframediv (int): How many subframe divisions there are per frame
        frametime_s (float): The DAQ system frame time in seconds.

    Returns:
        pl.DataFrame: The input frame `df` enhanced with new columns
    """
    assert output_control.require_last or output_control.require_next
    subframe_time_ms = frametime_s * 1000.0 / subframediv

    df_subframe = df.select("subframecount")
    if output_control.require_last:
        df_subframe = df_subframe.join_asof(df_ext, on="subframecount", strategy="backward", coalesce=False, suffix="_prev_ext_trig")
        delta = pl.col("subframecount").cast(pl.Int64) - pl.col("subframecount_prev_ext_trig")
        df_subframe = df_subframe.with_columns(dsf_last_ext_trig=delta)
    if output_control.require_next:
        df_subframe = df_subframe.join_asof(df_ext, on="subframecount", strategy="forward", coalesce=False, suffix="_next_ext_trig")
        delta = pl.col("subframecount_next_ext_trig") - pl.col("subframecount").cast(pl.Int64)
        df_subframe = df_subframe.with_columns(dsf_next_ext_trig=delta)

    if output_control.ms_last_trig:
        df_subframe = df_subframe.with_columns(ms_last_ext_trig=(pl.col("dsf_last_ext_trig") * subframe_time_ms))
    if output_control.ms_next_trig:
        df_subframe = df_subframe.with_columns(ms_next_ext_trig=(pl.col("dsf_next_ext_trig") * subframe_time_ms))
    if output_control.ms_nearest_trig:
        df_subframe = df_subframe.with_columns(
            pl.when(pl.col("dsf_last_ext_trig") < pl.col("dsf_next_ext_trig"))
            .then(pl.col("dsf_last_ext_trig") * subframe_time_ms)
            .otherwise(-pl.col("dsf_next_ext_trig") * subframe_time_ms)
            .alias("ms_nearest_ext_trig")
        )

    # Drop columns that the user doesn't want
    if not output_control.absolute_sfs:
        df_subframe = df_subframe.drop("subframecount_prev_ext_trig", "subframecount_next_ext_trig")
    if not output_control.sf_last_trig:
        df_subframe = df_subframe.drop("dsf_last_ext_trig")
    if not output_control.sf_next_trig:
        df_subframe = df_subframe.drop("dsf_next_ext_trig")

    return df.with_columns(df_subframe)


@njit
def filter_signal_lowpass(sig: NDArray, fs: float, fcut: float) -> NDArray:
    """Tophat lowpass filter using an FFT

    Args:
        sig - the signal to be filtered
        fs - the sampling frequency of the signal
        fcut - the frequency at which to cutoff the signal

    Returns:
        the filtered signal
    """
    N = sig.shape[0]
    SIG = np.fft.fft(sig)
    freqs = (fs / N) * np.concatenate((np.arange(0, N / 2 + 1), np.arange(N / 2 - 1, 0, -1)))
    filt = np.zeros_like(SIG)
    filt[freqs < fcut] = 1.0
    sig_filt = np.fft.ifft(SIG * filt)
    return sig_filt


def correct_flux_jumps_original(vals: ArrayLike, mask: ArrayLike, flux_quant: float) -> NDArray:
    """Remove 'flux' jumps' from pretrigger mean.

    When using umux readout, if a pulse is recorded that has a very fast rising
    edge (e.g. a cosmic ray), the readout system will "slip" an integer number
    of flux quanta. This means that the baseline level returned to after the
    pulse will different from the pretrigger value by an integer number of flux
    quanta. This causes that pretrigger mean summary quantity to jump around in
    a way that causes trouble for the rest of MASS. This function attempts to
    correct these jumps.

    Arguments:
    vals -- array of values to correct
    mask -- mask indentifying "good" pulses
    flux_quant -- size of 1 flux quanta

    Returns:
    Array with values corrected
    """
    # The naive thing is to simply replace each value with its value mod
    # the flux quantum. But of the baseline value turns out to fluctuate
    # about an integer number of flux quanta, this will introduce new
    # jumps. I don't know the best way to handle this in general. For now,
    # if there are still jumps after the mod, I add 1/4 of a flux quanta
    # before modding, then mod, then subtract the 1/4 flux quantum and then
    # *add* a single flux quantum so that the values never go negative.
    #
    # To determine whether there are "still jumps after the mod" I look at the
    # difference between the largest and smallest values for "good" pulses. If
    # you don't exclude "bad" pulses, this check can be tricked in cases where
    # the pretrigger section contains a (sufficiently large) tail.
    vals = np.asarray(vals)
    mask = np.asarray(mask)
    if (np.amax(vals) - np.amin(vals)) >= flux_quant:
        corrected = vals % flux_quant
        if (np.amax(corrected[mask]) - np.amin(corrected[mask])) > 0.75 * flux_quant:
            corrected = (vals + flux_quant / 4) % (flux_quant)
            corrected = corrected - flux_quant / 4 + flux_quant
        corrected -= corrected[0] - vals[0]
        return corrected
    else:
        return vals


def correct_flux_jumps(vals: ArrayLike, mask: ArrayLike, flux_quant: float) -> NDArray:
    """Remove 'flux' jumps' from pretrigger mean.

    When using umux readout, if a pulse is recorded that has a very fast rising
    edge (e.g. a cosmic ray), the readout system will "slip" an integer number
    of flux quanta. This means that the baseline level returned to after the
    pulse will different from the pretrigger value by an integer number of flux
    quanta. This causes that pretrigger mean summary quantity to jump around in
    a way that causes trouble for the rest of MASS. This function attempts to
    correct these jumps.

    Arguments:
    vals -- array of values to correct
    mask -- mask indentifying "good" pulses
    flux_quant -- size of 1 flux quanta

    Returns:
    Array with values corrected
    """
    return unwrap_n(vals, flux_quant, mask)


@njit
def unwrap_n(data: NDArray[np.uint16], period: float, mask: ArrayLike, n: int = 3) -> NDArray:
    """Unwrap data that has been restricted to a given period.

    The algorithm iterates through each data point and compares
    it to the average of the previous n data points. It then
    offsets the data point by the multiple of the period that
    will minimize the difference from that n-point running average.

    For the first n data points, there are not enough preceding
    points to average n of them, so the algorithm will average
    fewer points.

    This code was written by Thomas Baker; integrated into MASS by Dan
    Becker. Sped up 300x by @njit.

    Parameters
    ----------
    data : array of data values
    period : the range over which the data loops
    n : how many preceding points to average
    mask: mask indentifying "good" pulses
    """
    mask = np.asarray(mask)
    udata = np.copy(data)  # make a copy for output
    if n <= 0:
        return udata

    # Iterate through each data point and offset it by
    # an amount that will minimize the difference from the
    # rolling average
    nprior = 0
    firstgoodidx = np.argmax(mask)
    priorvalues = np.full(n, udata[firstgoodidx])
    for i in range(len(data)):
        # Take the average of the previous n data points (only those with mask[i]==True).
        # Offset the data point by the most reasonable multiple of period (make this point closest to the running average).
        if mask[i]:
            avg = np.mean(priorvalues)
            if nprior == 0:
                avg = float(priorvalues[0])
            elif nprior < n:
                avg = np.mean(priorvalues[:nprior])
        udata[i] -= np.round((udata[i] - avg) / period) * period
        if mask[i]:
            priorvalues[nprior % n] = udata[i]
            nprior += 1
    return udata


def time_drift_correct(  # noqa: PLR0914
    time: ArrayLike,
    uncorrected: ArrayLike,
    w: float,
    sec_per_degree: float = 2000,
    pulses_per_degree: int = 2000,
    max_degrees: int = 20,
    ndeg: int | None = None,
    limit: tuple[float, float] | None = None,
) -> dict[str, Any]:
    """Compute a time-based drift correction that minimizes the spectral entropy.

    Args:
        time: The "time-axis". Correction will be a low-order polynomial in this.
        uncorrected: A filtered pulse height vector. Same length as indicator.
            Assumed to have some gain that is linearly related to indicator.
        w: the kernel width for the Laplace KDE density estimator
        sec_per_degree: assign as many as one polynomial degree per this many seconds
        pulses_per_degree: assign as many as one polynomial degree per this many pulses
        max_degrees: never use more than this many degrees of Legendre polynomial.
        n_deg: If not None, use this many degrees, regardless of the values of
               sec_per_degree, pulses_per_degree, and max_degress. In this case, never downsample.
        limit: The [lower,upper] limit of uncorrected values over which entropy is
            computed (default None).

    The entropy will be computed on corrected values only in the range
    [limit[0], limit[1]], so limit should be set to a characteristic large value
    of uncorrected. If limit is None (the default), then it will be computed as
    25%% larger than the 99%%ile point of uncorrected.

    Possible improvements in the future:
    * Use Numba to speed up.
    * Allow the parameters to be function arguments with defaults: photons per
      degree of freedom, seconds per degree of freedom, and max degrees of freedom.
    * Figure out how to span the available time with more than one set of legendre
      polynomials, so that we can have more than 20 d.o.f. eventually, for long runs.
    """
    time = np.asarray(time)
    uncorrected = np.asarray(uncorrected)
    if limit is None:
        pct99 = np.percentile(uncorrected, 99)
        limit = (0.0, 1.25 * pct99)

    use = np.logical_and(uncorrected > limit[0], uncorrected < limit[1])
    time = np.asarray(time[use])
    uncorrected = np.asarray(uncorrected[use])

    tmin, tmax = np.min(time), np.max(time)

    def normalize(t: NDArray) -> NDArray:
        """Rescale time to the range [-1,1]"""
        return (t - tmin) / (tmax - tmin) * 2 - 1

    info = {
        "tmin": tmin,
        "tmax": tmax,
        "normalize": normalize,
    }

    dtime = tmax - tmin
    N = len(time)
    if ndeg is None:
        ndeg = int(np.minimum(dtime / sec_per_degree, N / pulses_per_degree))
        ndeg = min(ndeg, max_degrees)
        ndeg = max(ndeg, 1)
        phot_per_degree = N / float(ndeg)
        if phot_per_degree >= 2 * pulses_per_degree:
            downsample = int(phot_per_degree / pulses_per_degree)
            time = time[::downsample]
            uncorrected = uncorrected[::downsample]
            N = len(time)
        else:
            downsample = 1
    else:
        downsample = 1

    LOG.info("Using %2d degrees for %6d photons (after %d downsample)", ndeg, N, downsample)
    LOG.info("That's %6.1f photons per degree, and %6.1f seconds per degree.", N / float(ndeg), dtime / ndeg)

    def model1(param_i: NDArray, i: int, param: NDArray, basis: NDArray) -> NDArray:
        "The model function, with one parameter pi varied, others fixed."
        pcopy = np.array(param)
        pcopy[i] = param_i
        return 1 + np.dot(basis.T, pcopy)

    def cost1(pi: NDArray, i: int, param: NDArray, y: NDArray, w: float, basis: NDArray) -> float:
        "The cost function (spectral entropy), with one parameter pi varied, others fixed."
        return laplace_entropy(y * model1(pi, i, param, basis), w=w)

    param = np.zeros(ndeg, dtype=float)
    xnorm = np.asarray(normalize(time), dtype=float)
    basis = np.vstack([sp.special.legendre(i + 1)(xnorm) for i in range(ndeg)])

    fc = 0
    model: Callable = np.poly1d([0])
    info["coefficients"] = np.zeros(ndeg, dtype=float)
    for i in range(ndeg):
        result, _fval, _iter, funcalls = sp.optimize.brent(
            cost1, (i, param, uncorrected, w, basis), [-0.001, 0.001], tol=1e-5, full_output=True
        )
        param[i] = result
        fc += funcalls
        model += sp.special.legendre(i + 1) * result
        info["coefficients"][i] = result
    info["funccalls"] = fc

    xk = np.linspace(-1, 1, 1 + 2 * ndeg)
    model2 = CubicSpline(xk, model(xk))
    H1 = laplace_entropy(uncorrected, w=w)
    H2 = laplace_entropy(uncorrected * (1 + model(xnorm)), w=w)
    H3 = laplace_entropy(uncorrected * (1 + model2(xnorm)), w=w)
    if H2 <= 0 or H2 - H1 > 0.0:
        model = np.poly1d([0])
    elif H3 <= 0 or H3 - H2 > 0.00001:
        model = model2

    def safe_model(x: NDArray) -> NDArray:
        "Return the TDC model, but enforcing no adjustment for times outside the range we learned on"
        y = model(x)
        y[x < -1] = 0
        y[x > +1] = 0
        return y

    info["entropies"] = (H1, H2, H3)
    info["model"] = safe_model
    return info


@njit
def resample_pulses(pulses: NDArray, shifts: ArrayLike | float | int) -> NDArray:
    """Resample multiple pulses (in place). The data will be modified and returned.

    Shifts can be integer or not; if shifts have a fractional part, linear interpolation is used.

    Positive values of `shifts` delay the values in a pulse record, and the initial value or values
    are padded by copying the first value of the pulse record. Negative values shift the pulse earlier
    in the record, and the final value is used to pad the end of the new record.

    Parameters
    ----------
    pulses : NDArray
        Array of pulses to resample _in place_. Size (N,M) for N pulses, each of length M.
    shifts : ArrayLike | float | int
        The number of samples to shift each pulse. Array of size `N`, or a scalar. If scalar,
        apply the same shift to all `N` pulses. Values can be non-integer, producing a linear
        interpolation of the data.

    Returns
    -------
    NDArray
        The `pulses` input array, which is modified by this operation.
    """
    # Positive shift means delay the record and pad the start.
    # Negative shift means rewind the record and pad the end.
    assert len(pulses.shape) == 2
    Npulses, Nsamples = pulses.shape
    if np.isscalar(shifts):
        shift_vec = np.zeros(Npulses, dtype=float)
        shift_vec.fill(shifts)
        return resample_pulses(pulses, shift_vec)

    shifts = np.asarray(shifts)
    assert len(shifts) == Npulses
    assert np.abs(shifts).max() < Nsamples

    for pulse, shift in zip(pulses, shifts):
        resample_one_pulse(pulse, shift)
    return pulses


@njit
def resample_one_pulse(pulse: NDArray, shift: float | int) -> NDArray:
    """Resample one pulses (in place). The data will be modified and returned.

    Shift can be integer or not; if shift has a fractional part, linear interpolation is used.

    Positive values of `shift` delay the values in a pulse record, and the initial value or values
    are padded by copying the first value of the pulse record. Negative values shift the pulse earlier
    in the record, and the final value is used to pad the end of the new record.

    Parameters
    ----------
    pulse : NDArray
        Pulse to resample _in place_.
    shift : float | int
        The number of samples to shift each pulse. Value can be non-integer, producing a linear
        interpolation of the data.

    Returns
    -------
    NDArray
        The `pulse` input vector, which is modified by this operation.
    """
    Nsamples = len(pulse)
    if shift > 0.0:
        fullshift = int(shift)
        fracshift = shift - fullshift
        # integer shift
        if fullshift > 0:
            pulse[fullshift:] = pulse[:-fullshift]
        # linear interpolation for the fraction
        pulse[fullshift + 1 :] = np.rint((1.0 - fracshift) * pulse[fullshift + 1 :] + fracshift * pulse[fullshift:-1])
        # fill the initial values
        pulse[:fullshift] = pulse[fullshift]

    elif shift < 0.0:
        fullshift = -int(-shift)
        fracshift = fullshift - shift
        # integer shift
        if fullshift < 0:
            pulse[:fullshift] = pulse[-fullshift:]
        # linear interpolation for the fraction
        pulse[: Nsamples + fullshift - 1] = np.rint(
            (1 - fracshift) * pulse[: Nsamples + fullshift - 1] + fracshift * pulse[1 : Nsamples + fullshift]
        )
        # fill the final values
        pulse[Nsamples + fullshift :] = pulse[-1]
    # Edge case of shift == 0.0 is a no-op and can be ignored.
    return pulse
