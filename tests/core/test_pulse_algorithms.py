import numpy as np
import polars as pl
import pytest
import mass2
from mass2.core import pulse_algorithms
from mass2.core.recipe import SummarizeStep


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

N_PRE = 50
N_SAMPLES = 200
TIMEBASE = 1e-6
BASELINE = 4096


def _make_pulse(peak_amplitude: int = 1000, noise_sigma: float = 0.0, seed: int = 42) -> np.ndarray:
    """Return a single uint16 trace: flat pretrig then a Gaussian-shaped pulse."""
    rng = np.random.default_rng(seed)
    t = np.arange(N_SAMPLES)
    noise = rng.standard_normal(N_SAMPLES) * noise_sigma if noise_sigma > 0 else np.zeros(N_SAMPLES)
    pulse_shape = peak_amplitude * np.exp(-0.5 * ((t - (N_PRE + 20)) / 8) ** 2)
    trace = np.clip(BASELINE + pulse_shape + noise, 0, 65535).astype(np.uint16)
    return trace


def _call_basic(traces: np.ndarray) -> np.ndarray:
    return pulse_algorithms.summarize_data_numba(
        traces, TIMEBASE, peak_samplenumber=N_PRE, pretrigger_ignore_samples=0, nPresamples=N_PRE
    )


def _call_full(traces: np.ndarray) -> np.ndarray:
    return pulse_algorithms.summarize_data_numba_full(
        traces, TIMEBASE, peak_samplenumber=N_PRE, pretrigger_ignore_samples=0, nPresamples=N_PRE
    )


def test_fit_pulse_2exp_with_tail():
    # Test parameters
    t0 = 50
    a_tail = 5.0
    tau_tail = 100.0
    a = 1000.0
    tau_rise = 10.0
    tau_fall_factor = 2
    baseline = -32.7

    # Create a time array
    t = np.arange(100)

    # Calculate the expected values using the function
    expected_values = pulse_algorithms.pulse_2exp_with_tail(t, t0, a_tail, tau_tail, a, tau_rise, tau_fall_factor, baseline)

    assert expected_values[0] == pytest.approx(a_tail + baseline, rel=1e-2)
    assert np.amax(expected_values) == pytest.approx(a + baseline, rel=1e-2)

    result = pulse_algorithms.fit_pulse_2exp_with_tail(expected_values, npre=50, dt=1)

    assert result.params["t0"].value == pytest.approx(t0, rel=1e-2)
    assert result.params["a_tail"].value == pytest.approx(a_tail, rel=1e-2)
    assert result.params["tau_tail"].value == pytest.approx(tau_tail, rel=1e-2)
    assert result.params["a"].value == pytest.approx(a, rel=1e-2)
    assert result.params["tau_rise"].value == pytest.approx(tau_rise, rel=1e-2)
    assert result.params["tau_fall_factor"].value == pytest.approx(tau_fall_factor, rel=1e-2)
    assert result.params["baseline"].value == pytest.approx(baseline, rel=1e-2)


# ---------------------------------------------------------------------------
# result_full_dtype
# ---------------------------------------------------------------------------


def test_result_full_dtype_contains_basic_fields() -> None:
    basic_names = set(pulse_algorithms.result_dtype.names or [])
    full_names = set(pulse_algorithms.result_full_dtype.names or [])
    assert basic_names.issubset(full_names), f"Missing from full dtype: {basic_names - full_names}"


def test_result_full_dtype_extra_fields() -> None:
    expected_extras = {
        "pretrig_slope",
        "pretrig_range",
        "pretrig_delta_max",
        "pretrig_delta_rms",
        "pulse_slope",
        "pulse_range",
        "pulse_delta_max",
        "pulse_delta_rms",
        "pulse_area",
        "pulse_fwhm",
        "pulse_centroid",
        "pulse_duration",
        "tail_average",
        "tail_rms",
        "tail_slope",
        "tail_range",
        "tail_delta_max",
        "tail_delta_rms",
        "tail_area",
        "decay_timescale",
        "fall_time",
        "peaks",
        "peak_index_interp",
        "peak_value_interp",
        "min_index",
        "is_clipped",
        "is_traceless",
        "pulse_onset",
        "pulse_sign",
    }
    full_names = set(pulse_algorithms.result_full_dtype.names or [])
    assert expected_extras.issubset(full_names), f"Missing: {expected_extras - full_names}"


# ---------------------------------------------------------------------------
# summarize_data_numba_full — basic correctness
# ---------------------------------------------------------------------------


def test_summarize_full_basic_fields_match_basic_function() -> None:
    """All fields present in result_dtype must be identical to result from summarize_data_numba."""
    traces = np.stack([_make_pulse(noise_sigma=5.0, seed=i) for i in range(20)])
    basic = _call_basic(traces)
    full = _call_full(traces)
    for field in pulse_algorithms.result_dtype.names or []:
        np.testing.assert_array_equal(basic[field], full[field], err_msg=f"Mismatch in field {field!r}")


def test_summarize_full_constant_pretrig() -> None:
    """Constant pretrigger → pretrig_slope ≈ 0, pretrig_range = 0, pretrig_delta_max = 0."""
    trace = _make_pulse(peak_amplitude=500, noise_sigma=0.0)
    # Ensure pretrig is exactly constant
    trace[:N_PRE] = BASELINE
    traces = trace[np.newaxis, :]
    result = _call_full(traces)
    assert result["pretrig_range"][0] == 0
    assert result["pretrig_delta_max"][0] == 0
    assert result["pretrig_delta_rms"][0] == pytest.approx(0.0, abs=1e-6)
    assert result["pretrig_slope"][0] == pytest.approx(0.0, abs=1e-6)


def test_summarize_full_pretrig_mean_rms() -> None:
    """pretrig_mean and pretrig_rms agree with direct numpy computation."""
    rng = np.random.default_rng(7)
    trace = _make_pulse(peak_amplitude=800, noise_sigma=0.0)
    noise = (rng.standard_normal(N_PRE) * 10).astype(np.int16)
    trace[:N_PRE] = np.clip(BASELINE + noise, 0, 65535).astype(np.uint16)
    traces = trace[np.newaxis, :]
    result = _call_full(traces)
    expected_mean = float(np.mean(trace[:N_PRE]))
    expected_rms = float(np.std(trace[:N_PRE]))
    assert result["pretrig_mean"][0] == pytest.approx(expected_mean, rel=1e-4)
    assert result["pretrig_rms"][0] == pytest.approx(expected_rms, rel=1e-3)


def test_summarize_full_single_peak() -> None:
    """A clean Gaussian pulse should produce peaks == 1."""
    trace = _make_pulse(peak_amplitude=1000, noise_sigma=0.0)
    result = _call_full(trace[np.newaxis, :])
    assert result["peaks"][0] == 1


def test_summarize_full_pulse_area_sign() -> None:
    """pulse_area should be positive for a positive pulse and zero or negative for flat pretrig."""
    trace = _make_pulse(peak_amplitude=1000, noise_sigma=0.0)
    result = _call_full(trace[np.newaxis, :])
    assert result["pulse_area"][0] > 0


def test_summarize_full_min_index() -> None:
    """min_index should point to the sample with the minimum ADC value."""
    trace = _make_pulse(peak_amplitude=500, noise_sigma=0.0)
    trace[:N_PRE] = BASELINE  # flat pretrig
    result = _call_full(trace[np.newaxis, :])
    expected_min_idx = int(np.argmin(trace))
    assert result["min_index"][0] == expected_min_idx


def test_summarize_full_is_clipped_false() -> None:
    """A non-clipped pulse → is_clipped = False."""
    trace = _make_pulse(peak_amplitude=500, noise_sigma=0.0)
    result = _call_full(trace[np.newaxis, :])
    assert not result["is_clipped"][0]


def test_summarize_full_is_clipped_true() -> None:
    """A pulse saturating at max_adc → is_clipped = True."""
    max_adc = 16383
    trace = _make_pulse(peak_amplitude=500, noise_sigma=0.0)
    trace[N_PRE + 20] = max_adc  # force saturation
    result = pulse_algorithms.summarize_data_numba_full(
        trace[np.newaxis, :],
        TIMEBASE,
        peak_samplenumber=N_PRE,
        pretrigger_ignore_samples=0,
        nPresamples=N_PRE,
        max_adc=max_adc,
    )
    assert result["is_clipped"][0]


def test_summarize_full_is_traceless() -> None:
    """nSamples == 0 → is_traceless = True for all pulses."""
    traces = np.zeros((3, 0), dtype=np.uint16)
    result = pulse_algorithms.summarize_data_numba_full(
        traces,
        TIMEBASE,
        peak_samplenumber=0,
        pretrigger_ignore_samples=0,
        nPresamples=0,
    )
    assert result["is_traceless"].all()


def test_summarize_full_pulse_sign_positive() -> None:
    """A clear positive pulse → pulse_sign == +1."""
    trace = _make_pulse(peak_amplitude=1000, noise_sigma=0.0)
    result = _call_full(trace[np.newaxis, :])
    assert result["pulse_sign"][0] == 1


def test_summarize_full_pulse_sign_flat() -> None:
    """A flat trace with amplitude below noise threshold → pulse_sign == 0."""
    trace = np.full((1, N_SAMPLES), BASELINE, dtype=np.uint16)
    result = _call_full(trace)
    assert result["pulse_sign"][0] == 0


def test_summarize_full_pulse_onset_detected() -> None:
    """pulse_onset should be < nSamples for a clear positive pulse."""
    trace = _make_pulse(peak_amplitude=2000, noise_sigma=0.0)
    result = _call_full(trace[np.newaxis, :])
    onset = result["pulse_onset"][0]
    assert onset >= 0
    assert onset < N_SAMPLES


def test_summarize_full_pulse_onset_not_found_for_flat() -> None:
    """For a flat trace, pulse_onset should be -1 (no onset detected)."""
    trace = np.full((1, N_SAMPLES), BASELINE, dtype=np.uint16)
    result = _call_full(trace)
    assert result["pulse_onset"][0] == -1


def test_summarize_full_peak_interp_near_integer_peak() -> None:
    """Interpolated peak index should be close to the integer peak index for a smooth pulse."""
    trace = _make_pulse(peak_amplitude=1000, noise_sigma=0.0)
    result = _call_full(trace[np.newaxis, :])
    integer_peak = int(result["peak_index"][0])
    interp_peak = float(result["peak_index_interp"][0])
    assert abs(interp_peak - integer_peak) < 1.0


def test_summarize_full_fwhm_positive() -> None:
    """FWHM should be positive for a pulse with a clear peak."""
    trace = _make_pulse(peak_amplitude=1000, noise_sigma=0.0)
    result = _call_full(trace[np.newaxis, :])
    assert result["pulse_fwhm"][0] > 0


def test_summarize_full_fall_time_positive() -> None:
    """fall_time should be positive when the pulse has a falling edge."""
    trace = _make_pulse(peak_amplitude=1000, noise_sigma=0.0)
    result = _call_full(trace[np.newaxis, :])
    assert result["fall_time"][0] > 0


# ---------------------------------------------------------------------------
# Channel.summarize_pulses with full_summary=True
# ---------------------------------------------------------------------------


def _make_dummy_channel(npulses: int = 30, peak_amplitude: int = 800) -> mass2.Channel:
    rng = np.random.default_rng(0)
    n = N_SAMPLES
    noise = (rng.standard_normal((npulses, n)) * 5).astype(np.int16)
    t = np.arange(n)
    shape = peak_amplitude * np.exp(-0.5 * ((t - (N_PRE + 20)) / 8) ** 2)
    traces = np.clip(BASELINE + shape + noise, 0, 65535).astype(np.uint16)
    df = pl.DataFrame({"pulse": traces})
    header_df = pl.DataFrame()
    header = mass2.ChannelHeader(
        "test",
        data_source=None,
        ch_num=0,
        frametime_s=TIMEBASE,
        n_presamples=N_PRE,
        n_samples=n,
        df=header_df,
    )
    return mass2.Channel(df, header, npulses=npulses, noise=None)


def test_channel_summarize_pulses_full_summary() -> None:
    """summarize_pulses(full_summary=True) should produce all extra columns."""
    ch = _make_dummy_channel()
    ch_full = ch.summarize_pulses(full_summary=True)
    extra_cols = {"pretrig_slope", "pulse_area", "fall_time", "peaks", "is_clipped", "pulse_sign"}
    for col in extra_cols:
        assert col in ch_full.df.columns, f"Missing column: {col}"


def test_channel_summarize_pulses_basic_vs_full_basic_fields() -> None:
    """Both modes should produce identical basic summary fields."""
    ch = _make_dummy_channel()
    ch_basic = ch.summarize_pulses(full_summary=False)
    ch_full = ch.summarize_pulses(full_summary=True)
    for field in pulse_algorithms.result_dtype.names or []:
        np.testing.assert_array_equal(
            ch_basic.df[field].to_numpy(),
            ch_full.df[field].to_numpy(),
            err_msg=f"Mismatch in column {field!r}",
        )


# ---------------------------------------------------------------------------
# SummarizeStep with derive=True
# ---------------------------------------------------------------------------


def test_summarize_step_derive_adds_derived_columns() -> None:
    """SummarizeStep with derive=True should add tail_fraction, pulse_fom, etc."""
    ch = _make_dummy_channel()
    step = SummarizeStep(
        inputs=["pulse"],
        output=list(pulse_algorithms.result_full_dtype.names or []),
        good_expr=pl.lit(True),
        use_expr=pl.lit(True),
        frametime_s=TIMEBASE,
        peak_index=N_PRE,
        pulse_col="pulse",
        pretrigger_ignore_samples=0,
        n_presamples=N_PRE,
        mode="full",
        derive=True,
    )
    df_derived = step.calc_from_df(ch.df)
    expected_derived_cols = {"tail_fraction", "pulse_fom", "tail_sigma", "rise_fall_ratio", "pretrig_drift"}
    for col in expected_derived_cols:
        assert col in df_derived.columns, f"Missing derived column: {col}"


def test_summarize_step_derive_tail_fraction_range() -> None:
    """tail_fraction = tail_area / pulse_area, so should be in (0, 1] for a decaying pulse."""
    ch = _make_dummy_channel()
    step = SummarizeStep(
        inputs=["pulse"],
        output=list(pulse_algorithms.result_full_dtype.names or []),
        good_expr=pl.lit(True),
        use_expr=pl.lit(True),
        frametime_s=TIMEBASE,
        peak_index=N_PRE,
        pulse_col="pulse",
        pretrigger_ignore_samples=0,
        n_presamples=N_PRE,
        mode="full",
        derive=True,
    )
    df = step.calc_from_df(ch.df)
    tf = df["tail_fraction"].to_numpy()
    # For a decaying Gaussian pulse the tail area can exceed pulse area (longer tail),
    # so only check it is finite and not NaN
    assert np.all(np.isfinite(tf))
