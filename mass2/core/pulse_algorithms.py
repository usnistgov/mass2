"""
Pulse summarizing algorithms.
"""

import numpy as np
from numba import njit
from numpy.typing import NDArray, ArrayLike
import lmfit
from ..calibration.line_models import LineModelResult

# Define the dtype for the structured array
result_dtype = np.dtype([
    ("pretrig_mean", np.float32),  # mean ADC value of pretrigger samples (baseline)
    ("pretrig_rms", np.float32),  # RMS noise of pretrigger samples
    ("pulse_average", np.float32),  # mean baseline-subtracted ADC value over the post-trigger region
    ("pulse_rms", np.float32),  # RMS of the baseline-subtracted post-trigger region
    ("promptness", np.float32),  # fraction of pulse energy in the prompt window (samples nPresamples+2..+7)
    ("rise_time", np.float32),  # 10%→90% rise time on the leading edge (seconds)
    ("postpeak_deriv", np.float32),  # maximum post-peak derivative (0.1 × max 5-point slope), proxy for ringing
    ("peak_index", np.uint16),  # sample index of the maximum ADC value
    ("peak_value", np.uint16),  # baseline-subtracted peak ADC value
    ("min_value", np.uint16),  # minimum raw ADC value in the trace
    ("shift1", np.uint16),  # 1 if the prompt window was shifted 1 sample earlier due to early pulse onset
])

result_full_dtype = np.dtype(
    result_dtype.descr
    + np.dtype([
        ("pretrig_slope", np.float32),  # linear slope of the pretrigger baseline (ADC/sample)
        ("pretrig_range", np.float32),  # peak-to-peak spread of pretrigger samples (max − min)
        ("pretrig_delta_max", np.uint16),  # maximum absolute difference between consecutive pretrigger samples
        ("pretrig_delta_rms", np.float32),  # RMS of consecutive differences in the pretrigger (AC noise)
        ("pulse_slope", np.float32),  # linear slope over the full trace (ADC/sample)
        ("pulse_range", np.float32),  # raw peak value minus minimum value in the trace
        ("pulse_delta_max", np.uint16),  # maximum absolute difference between consecutive samples in the full trace
        ("pulse_delta_rms", np.float32),  # RMS of consecutive differences over the full trace
        ("pulse_area", np.float32),  # baseline-subtracted sum over the full trace (pulse integral in ADC·samples)
        ("pulse_fwhm", np.float32),  # full width at half maximum (seconds)
        ("pulse_centroid", np.float32),  # time-weighted centroid of the baseline-subtracted pulse (samples)
        ("pulse_duration", np.uint16),  # number of samples above the 10% threshold
        ("tail_average", np.float32),  # mean baseline-subtracted ADC value in the tail region
        ("tail_rms", np.float32),  # RMS of tail samples around baseline
        ("tail_slope", np.float32),  # linear slope of the tail region (ADC/sample)
        ("tail_range", np.float32),  # peak-to-peak spread of tail samples (max − min)
        ("tail_delta_max", np.uint16),  # maximum absolute difference between consecutive tail samples
        ("tail_delta_rms", np.float32),  # RMS of consecutive differences in the tail
        ("tail_area", np.float32),  # baseline-subtracted sum over the tail region (ADC·samples)
        ("decay_timescale", np.float32),  # pulse_area / peak_above_baseline: effective exponential decay time (samples)
        ("fall_time", np.float32),  # 90%→10% fall time on the trailing edge (seconds)
        ("peaks", np.uint16),  # number of local maxima above the 10% threshold (ideal pulse = 1)
        ("peak_index_interp", np.float32),  # sub-sample peak position from parabolic interpolation (samples)
        ("peak_value_interp", np.float32),  # sub-sample peak ADC value from parabolic interpolation (raw, not baseline-subtracted)
        ("min_index", np.uint16),  # sample index of the minimum ADC value
        ("is_clipped", np.bool_),  # True if raw peak ≥ max_adc (ADC saturation)
        ("is_traceless", np.bool_),  # True if the trace is empty (nSamples == 0)
        ("pulse_onset", np.int32,),  # index where pulse first exceeds sigma_onset×pretrig_rms for onset_samples consecutive samples
        ("pulse_sign", np.int8),  # sign of the pulse: +1 positive, −1 negative, 0 if amplitude < sigma_sign×pretrig_rms
    ]).descr
)

# Create a type alias for the structured array
ResultArrayType = NDArray


# this cache code works, but it's not clear it's faster than just running the function
# from joblib import Memory
# cache_dir = Path.cwd()/"_summarize_data_cache"
# memory = Memory(cache_dir, mmap_mode="r", verbose=0)


# @memory.cache
@njit
def summarize_data_numba(  # noqa: PLR0914
    rawdata: NDArray[np.uint16],
    timebase: float,
    peak_samplenumber: int,
    pretrigger_ignore_samples: int,
    nPresamples: int,
) -> ResultArrayType:
    """Summarize one segment of the data file, loading it into cache."""
    nPulses = rawdata.shape[0]
    nSamples = rawdata.shape[1]

    e_nPresamples = nPresamples - pretrigger_ignore_samples

    # Create the structured array for results
    results = np.zeros(nPulses, dtype=result_dtype)

    for j in range(nPulses):
        pulse = rawdata[j, :]
        pretrig_sum = 0.0
        pretrig_rms_sum = 0.0
        pulse_sum = 0.0
        pulse_rms_sum = 0.0
        promptness_sum = 0.0
        peak_value = 0
        peak_index = 0
        min_value = np.iinfo(np.uint16).max
        s_prompt = nPresamples + 2
        e_prompt = nPresamples + 8

        for k in range(nSamples):
            signal = pulse[k]

            if signal > peak_value:
                peak_value = signal
                peak_index = k
            min_value = min(min_value, signal)

            if k < e_nPresamples:
                pretrig_sum += signal
                pretrig_rms_sum += signal**2

            if s_prompt <= k < e_prompt:
                promptness_sum += signal

            if k == nPresamples - 1:
                ptm = pretrig_sum / e_nPresamples
                ptrms = np.sqrt(pretrig_rms_sum / e_nPresamples - ptm**2)
                if signal - ptm > 4.3 * ptrms:
                    e_prompt -= 1
                    s_prompt -= 1
                    results["shift1"][j] = 1
                else:
                    results["shift1"][j] = 0

            if k >= nPresamples - 1:
                pulse_sum += signal
                pulse_rms_sum += signal**2

        results["pretrig_mean"][j] = ptm
        results["pretrig_rms"][j] = ptrms
        if ptm < peak_value:
            peak_value -= int(ptm + 0.5)
            results["promptness"][j] = (promptness_sum / 6.0 - ptm) / peak_value
            results["peak_value"][j] = peak_value
            results["peak_index"][j] = peak_index
        else:
            results["promptness"][j] = 0.0
            results["peak_value"][j] = 0
            results["peak_index"][j] = 0
        results["min_value"][j] = min_value
        pulse_avg = pulse_sum / (nSamples - nPresamples + 1) - ptm
        results["pulse_average"][j] = pulse_avg
        results["pulse_rms"][j] = np.sqrt(pulse_rms_sum / (nSamples - nPresamples + 1) - ptm * pulse_avg * 2 - ptm**2)

        low_th = int(0.1 * peak_value + ptm)
        high_th = int(0.9 * peak_value + ptm)

        k = nPresamples
        low_value = pulse[k]
        low_idx = k
        while k < nSamples:
            signal = pulse[k]
            if signal > low_th:
                low_idx = k
                low_value = signal
                break
            k += 1

        high_value = low_value
        high_idx = low_idx

        while k < nSamples:
            signal = pulse[k]
            if signal > high_th:
                high_idx = k - 1
                high_value = pulse[high_idx]
                break
            k += 1

        if high_value > low_value:
            results["rise_time"][j] = timebase * (high_idx - low_idx) * peak_value / (high_value - low_value)
        else:
            results["rise_time"][j] = timebase

        # The following is quite confusing, but it appears to be equivalent to
        # slope = -2 * pulse[peak_samplenumber:-4]
        # slope -= pulse[peak_samplenumber+1:-3]
        # slope += pulse[peak_samplenumber+3:-1]
        # slope += 2*pulse[peak_samplenumber+4:]
        # slope = np.minimum(slope[2:], slope[:-2])
        # results["postpeak_deriv"][j] = 0.1 * np.max(slope)
        # TODO: consider replacing, if the above is not slower?

        f0, f1, f3, f4 = 2, 1, -1, -2
        s0, s1, s2, s3 = (
            pulse[peak_samplenumber],
            pulse[peak_samplenumber + 1],
            pulse[peak_samplenumber + 2],
            pulse[peak_samplenumber + 3],
        )
        s4 = pulse[peak_samplenumber + 4]
        t0 = f4 * s0 + f3 * s1 + f1 * s3 + f0 * s4
        s0, s1, s2, s3 = s1, s2, s3, s4
        s4 = pulse[peak_samplenumber + 5]
        t1 = f4 * s0 + f3 * s1 + f1 * s3 + f0 * s4
        t_max_deriv = np.iinfo(np.int32).min

        for k in range(peak_samplenumber + 6, nSamples):
            s0, s1, s2, s3 = s1, s2, s3, s4
            s4 = pulse[k]
            t2 = f4 * s0 + f3 * s1 + f1 * s3 + f0 * s4

            t3 = min(t2, t0)
            t_max_deriv = max(t_max_deriv, t3)

            t0, t1 = t1, t2

        results["postpeak_deriv"][j] = 0.1 * t_max_deriv

    return results


@njit
def summarize_data_numba_full(  # noqa: PLR0914, PLR0917
    rawdata: NDArray[np.uint16],
    timebase: float,
    peak_samplenumber: int,
    pretrigger_ignore_samples: int,
    nPresamples: int,
    max_adc: int = 16383,
    nTailSamples: int = 0,
    sigma_sign: float = 3.0,
    sigma_onset: float = 3.0,
    onset_samples: int = 3,
) -> NDArray:
    """
    Extended summary: computes all fields in result_full_dtype, including pulse_sign
    and pulse_onset. nTailSamples: samples from the end of the trace used for tail
    statistics (0 = same as e_nPresamples).
    """
    nPulses = rawdata.shape[0]
    nSamples = rawdata.shape[1]

    e_nPresamples = nPresamples - pretrigger_ignore_samples
    e_nTailSamples = nTailSamples if nTailSamples > 0 else e_nPresamples
    e_nTailSamples = min(e_nTailSamples, nSamples)

    results = np.zeros(nPulses, dtype=result_full_dtype)

    if nSamples == 0:
        results["is_traceless"][:] = True
        return results

    n_full = nSamples
    sum_x_full = n_full * (n_full - 1) / 2.0
    denom_full = n_full**2 * (n_full**2 - 1) / 12.0

    n_pre = e_nPresamples
    sum_x_pre = n_pre * (n_pre - 1) / 2.0
    denom_pre = n_pre**2 * (n_pre**2 - 1) / 12.0

    n_tail = e_nTailSamples
    tail_start = nSamples - n_tail
    sum_x_tail = n_tail * (n_tail - 1) / 2.0
    denom_tail = n_tail**2 * (n_tail**2 - 1) / 12.0

    for j in range(nPulses):
        pulse = rawdata[j, :]

        # ---- shared / original accumulators ----
        pretrig_sum = 0.0
        pretrig_rms_sum = 0.0
        pulse_sum = 0.0
        pulse_rms_sum = 0.0
        promptness_sum = 0.0
        peak_value = 0
        peak_index = 0
        min_value = np.iinfo(np.uint16).max
        s_prompt = nPresamples + 2
        e_prompt = nPresamples + 8

        # ---- extra accumulators ----
        sum_y_full = 0.0
        sum_xy_full = 0.0
        sum_xy_pre = 0.0

        min_idx = 0
        max_pre_val = 0
        min_pre_val = np.iinfo(np.uint16).max
        max_pre_delta = 0
        prev_pre_sample = pulse[0]
        pretrig_diffsq_sum = 0.0

        max_adj_delta = 0
        prev_sample = pulse[0]
        pulse_diffsq_sum = 0.0

        # ---- tail accumulators ----
        tail_sum = 0.0
        tail_rms_sum = 0.0
        sum_xy_tail = 0.0
        max_tail_val = 0
        min_tail_val = np.iinfo(np.uint16).max
        max_tail_delta = 0
        prev_tail_sample = pulse[tail_start]
        tail_diffsq_sum = 0.0

        for k in range(nSamples):
            signal = pulse[k]

            if signal > peak_value:
                peak_value = signal
                peak_index = k

            if signal < min_value:
                min_value = signal
                min_idx = k

            sum_y_full += signal
            sum_xy_full += k * signal

            # ---- pretrigger region ----
            if k < e_nPresamples:
                pretrig_sum += signal
                pretrig_rms_sum += signal**2
                sum_xy_pre += k * signal
                max_pre_val = max(max_pre_val, signal)
                min_pre_val = min(min_pre_val, signal)
                delta = int(abs(np.int32(signal) - np.int32(prev_pre_sample)))
                max_pre_delta = max(max_pre_delta, delta)
                pretrig_diffsq_sum += delta**2
                prev_pre_sample = signal

            # ---- glitch detector over full trace ----
            delta = int(abs(np.int32(signal) - np.int32(prev_sample)))
            max_adj_delta = max(max_adj_delta, delta)
            pulse_diffsq_sum += delta**2
            prev_sample = signal

            # ---- tail region ----
            if k >= tail_start:
                tail_sum += signal
                tail_rms_sum += signal**2
                sum_xy_tail += (k - tail_start) * signal
                max_tail_val = max(max_tail_val, signal)
                min_tail_val = min(min_tail_val, signal)
                delta_t = int(abs(np.int32(signal) - np.int32(prev_tail_sample)))
                max_tail_delta = max(max_tail_delta, delta_t)
                tail_diffsq_sum += delta_t**2
                prev_tail_sample = signal

            if s_prompt <= k < e_prompt:
                promptness_sum += signal

            if k == nPresamples - 1:
                ptm = pretrig_sum / e_nPresamples
                ptrms = np.sqrt(pretrig_rms_sum / e_nPresamples - ptm**2)
                if signal - ptm > 4.3 * ptrms:
                    e_prompt -= 1
                    s_prompt -= 1
                    results["shift1"][j] = 1
                else:
                    results["shift1"][j] = 0

            if k >= nPresamples - 1:
                pulse_sum += signal
                pulse_rms_sum += signal**2

        # ============================================================
        # original result_dtype fields
        # ============================================================
        peak_value_raw = peak_value

        results["pretrig_mean"][j] = ptm
        results["pretrig_rms"][j] = ptrms
        if ptm < peak_value:
            peak_value -= int(ptm + 0.5)
            results["promptness"][j] = (promptness_sum / 6.0 - ptm) / peak_value
            results["peak_value"][j] = peak_value
            results["peak_index"][j] = peak_index
        else:
            results["promptness"][j] = 0.0
            results["peak_value"][j] = 0
            results["peak_index"][j] = 0
        results["min_value"][j] = min_value

        pulse_avg = pulse_sum / (nSamples - nPresamples + 1) - ptm
        results["pulse_average"][j] = pulse_avg
        results["pulse_rms"][j] = np.sqrt(pulse_rms_sum / (nSamples - nPresamples + 1) - ptm * pulse_avg * 2 - ptm**2)

        # ---- thresholds (ADC units, baseline included) ----
        low_th = int(0.1 * peak_value + ptm)
        high_th = int(0.9 * peak_value + ptm)
        half_th = int(0.5 * peak_value + ptm)

        # ---- rising edge: find 10%, 50%, 90% crossings ----
        k = nPresamples
        low_value = pulse[k]
        low_idx = k

        while k < nSamples:
            signal = pulse[k]
            if signal > low_th:
                low_idx = k
                low_value = signal
                break
            k += 1

        high_value = low_value
        high_idx = low_idx
        found_half_rise = False
        fwhm_rise_idx = low_idx

        while k < nSamples:
            signal = pulse[k]
            if not found_half_rise and signal > half_th:
                fwhm_rise_idx = k
                found_half_rise = True
            if signal > high_th:
                high_idx = k - 1
                high_value = pulse[high_idx]
                break
            k += 1

        if high_value > low_value:
            results["rise_time"][j] = timebase * (high_idx - low_idx) * peak_value / (high_value - low_value)
        else:
            results["rise_time"][j] = timebase

        # ---- falling edge: find 90%, 50%, 10% crossings ----
        fall_90_idx = nSamples - 1
        fall_90_val = float(pulse[nSamples - 1])
        fall_10_idx = nSamples - 1
        fall_10_val = float(pulse[nSamples - 1])
        fwhm_fall_idx = nSamples - 1
        found_fall_90 = False
        found_fwhm_fall = False

        k = peak_index + 1
        while k < nSamples:
            signal = pulse[k]
            if not found_fall_90 and signal <= high_th:
                fall_90_idx = k
                fall_90_val = float(signal)
                found_fall_90 = True
            if found_fall_90 and not found_fwhm_fall and signal <= half_th:
                fwhm_fall_idx = k
                found_fwhm_fall = True
            if found_fall_90 and signal <= low_th:
                fall_10_idx = k
                fall_10_val = float(signal)
                break
            k += 1

        if fall_90_val > fall_10_val:
            results["fall_time"][j] = timebase * (fall_10_idx - fall_90_idx) * peak_value / (fall_90_val - fall_10_val)
        else:
            results["fall_time"][j] = timebase

        if found_half_rise and found_fwhm_fall:
            results["pulse_fwhm"][j] = timebase * (fwhm_fall_idx - fwhm_rise_idx)

        # ---- pulse_duration and peaks (local maxima above 10% threshold) ----
        pulse_duration = 0
        peaks = 0
        k = nPresamples
        while k < nSamples:
            if pulse[k] > low_th:
                pulse_duration += 1
            if nPresamples < k < nSamples - 1 and pulse[k] > low_th and pulse[k] > pulse[k - 1] and pulse[k] > pulse[k + 1]:
                peaks += 1
            k += 1
        results["pulse_duration"][j] = pulse_duration
        results["peaks"][j] = peaks

        # ---- postpeak_deriv (unchanged from original) ----
        f0, f1, f3, f4 = 2, 1, -1, -2
        s0, s1, s2, s3 = (
            pulse[peak_samplenumber],
            pulse[peak_samplenumber + 1],
            pulse[peak_samplenumber + 2],
            pulse[peak_samplenumber + 3],
        )
        s4 = pulse[peak_samplenumber + 4]
        t0 = f4 * s0 + f3 * s1 + f1 * s3 + f0 * s4
        s0, s1, s2, s3 = s1, s2, s3, s4
        s4 = pulse[peak_samplenumber + 5]
        t1 = f4 * s0 + f3 * s1 + f1 * s3 + f0 * s4
        t_max_deriv = np.iinfo(np.int32).min

        for k in range(peak_samplenumber + 6, nSamples):
            s0, s1, s2, s3 = s1, s2, s3, s4
            s4 = pulse[k]
            t2 = f4 * s0 + f3 * s1 + f1 * s3 + f0 * s4
            t3 = min(t2, t0)
            t_max_deriv = max(t_max_deriv, t3)
            t0, t1 = t1, t2

        results["postpeak_deriv"][j] = 0.1 * t_max_deriv

        # ============================================================
        # full (result_full_dtype) fields
        # ============================================================

        # ---- pretrig ----
        results["pretrig_range"][j] = max_pre_val - min_pre_val
        results["pretrig_delta_max"][j] = max_pre_delta
        if n_pre > 1:
            results["pretrig_slope"][j] = (n_pre * sum_xy_pre - sum_x_pre * pretrig_sum) / denom_pre
            results["pretrig_delta_rms"][j] = np.sqrt(pretrig_diffsq_sum / (n_pre - 1))

        # ---- pulse (full-trace) ----
        results["pulse_range"][j] = peak_value_raw - min_value
        results["pulse_delta_max"][j] = max_adj_delta
        results["min_index"][j] = min_idx
        results["is_clipped"][j] = peak_value_raw >= max_adc
        results["pulse_slope"][j] = (n_full * sum_xy_full - sum_x_full * sum_y_full) / denom_full if n_full >= 2 else 0.0
        if n_full > 1:
            results["pulse_delta_rms"][j] = np.sqrt(pulse_diffsq_sum / (n_full - 1))

        baseline_subtracted_sum = sum_y_full - n_full * ptm
        results["pulse_area"][j] = baseline_subtracted_sum
        results["pulse_centroid"][j] = (
            (sum_xy_full - ptm * sum_x_full) / baseline_subtracted_sum if baseline_subtracted_sum != 0 else 0.0
        )

        peak_above_baseline = peak_value_raw - ptm
        results["decay_timescale"][j] = baseline_subtracted_sum / peak_above_baseline if peak_above_baseline > 0 else 0.0

        # ---- tail ----
        tail_avg = tail_sum / n_tail - ptm
        results["tail_average"][j] = tail_avg
        results["tail_rms"][j] = np.sqrt(tail_rms_sum / n_tail - ptm * tail_avg * 2 - ptm**2)
        results["tail_range"][j] = max_tail_val - min_tail_val
        results["tail_delta_max"][j] = max_tail_delta
        tail_area_val = tail_sum - n_tail * ptm
        results["tail_area"][j] = tail_area_val
        if n_tail > 1:
            results["tail_slope"][j] = (n_tail * sum_xy_tail - sum_x_tail * tail_sum) / denom_tail
            results["tail_delta_rms"][j] = np.sqrt(tail_diffsq_sum / (n_tail - 1))

        # ---- sub-sample peak interpolation (parabolic) ----
        if peak_index > 0 and peak_index < nSamples - 1:
            p_left = float(pulse[peak_index - 1])
            p_mid = float(pulse[peak_index])
            p_right = float(pulse[peak_index + 1])
            denom2 = p_left - 2.0 * p_mid + p_right
            if denom2 < 0:
                results["peak_index_interp"][j] = peak_index + 0.5 * (p_left - p_right) / denom2
                results["peak_value_interp"][j] = p_mid - (p_right - p_left) ** 2 / (8.0 * denom2)
            else:
                results["peak_index_interp"][j] = float(peak_index)
                results["peak_value_interp"][j] = p_mid
        else:
            results["peak_index_interp"][j] = float(peak_index)
            results["peak_value_interp"][j] = float(pulse[peak_index])

        # ---- pulse_sign: compare peak and min amplitude against noise threshold ----
        results["pulse_onset"][j] = -1
        sign = 0
        if ptrms > 0:
            if float(peak_value_raw) - ptm > sigma_sign * ptrms:
                sign = 1
            elif ptm - float(min_value) > sigma_sign * ptrms:
                sign = -1
        results["pulse_sign"][j] = sign

        # ---- pulse_onset: first run of onset_samples samples exceeding threshold ----
        if ptrms > 0:
            threshold = sigma_onset * ptrms
            run = 0
            for k in range(e_nPresamples, nSamples):
                val = float(pulse[k]) - ptm
                if sign == 1:
                    exceeds = val >= threshold
                elif sign == -1:
                    exceeds = -val >= threshold
                else:
                    exceeds = abs(val) >= threshold
                if exceeds:
                    run += 1
                    if run >= onset_samples:
                        results["pulse_onset"][j] = k - onset_samples + 1
                        break
                else:
                    run = 0

    return results


def pulse_2exp_with_tail(
    t: ArrayLike, t0: float, a_tail: float, tau_tail: float, a: float, tau_rise: float, tau_fall_factor: float, baseline: float
) -> NDArray:
    """Create a pulse shape from two exponentials plus an exponential tail."""
    tt = np.asarray(t) - t0
    tau_fall = tau_rise * tau_fall_factor
    assert tau_fall_factor >= 1

    if tau_fall_factor > 1:
        # location of peak
        t_peak = (tau_rise * tau_fall) / (tau_fall - tau_rise) * np.log(tau_fall / tau_rise)
        # value at peak
        max_val = np.exp(-t_peak / tau_fall) - np.exp(-t_peak / tau_rise)
    else:  # tau_fall == tau_rise
        max_val = np.exp(-1)

    return (
        a_tail * np.exp(-tt / tau_tail) / np.exp(-tt[0] / tau_tail)  # normalized tail
        + a * (np.exp(-tt / tau_fall) - np.exp(-tt / tau_rise)) * np.greater(tt, 0) / max_val
        + baseline
    )


def fit_pulse_2exp_with_tail(data: ArrayLike, npre: int, dt: float = 1, guess_tau: float | None = None) -> LineModelResult:
    """Fit a pulse shape to data using two exponentials plus an exponential tail."""
    data = np.asarray(data)
    if guess_tau is None:
        guess_tau = dt * len(data) / 5
    model = lmfit.Model(pulse_2exp_with_tail)
    baseline = np.amin(data)
    params = model.make_params(
        t0=npre * dt,
        a_tail=data[0] - baseline,
        baseline=baseline,
        a=np.amax(data) - baseline,
        tau_tail=guess_tau,
        tau_rise=guess_tau,
        tau_fall_factor=2.0,
    )
    params["a_tail"].set(min=0)
    params["a"].set(min=0)
    params["tau_tail"].set(min=dt / 5)
    params["tau_rise"].set(min=dt / 5)
    params["tau_fall_factor"].set(min=1)
    params.add("tau_fall", expr="tau_rise*tau_fall_factor")

    result = model.fit(data, params, t=np.arange(len(data)) * dt)

    return result
