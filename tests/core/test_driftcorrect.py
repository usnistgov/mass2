import datetime
import numpy as np
import polars as pl
import pytest
import mass2
import pulsedata


def test_drift_correct(N=2e5, a0=100, b0=100, slope=0.01):
    N = int(N)
    rng = np.random.default_rng(42)
    a = rng.standard_normal(N) + a0
    b = rng.standard_normal(N) + b0
    dc_perfect = mass2.core.drift_correction.DriftCorrection(slope=-slope, offset=a0)
    b_tilted = dc_perfect(a, b)
    dc = mass2.core.drift_correct(indicator=a, uncorrected=b_tilted)
    assert dc.slope == pytest.approx(slope, abs=1e-3)
    assert dc.offset == pytest.approx(b0, abs=1e-2)


def test_time_drift_correct():
    def _do_steps(ch: mass2.Channel) -> mass2.Channel:
        return (
            ch.summarize_pulses()
            .with_good_expr_pretrig_rms_and_postpeak_deriv(8, 8)
            .filter5lag(f_3db=10000)
            .driftcorrect(indicator_col="pretrig_mean", uncorrected_col="5lagy", use_expr=pl.lit(True))
            .time_drift_correct("timestamp", "5lagy_dc", "5lagy_tdc")
        )

    # Make sure we can run a time drift correct
    p = pulsedata.pulse_noise_ljh_pairs["20230626"]
    data = mass2.Channels.from_ljh_folder(p.pulse_folder, p.noise_folder, exclude_ch_nums=[4102])
    data = data.map(_do_steps)

    # Check that time-drift-correct changed the values at least a little
    ch = data.ch0
    p0 = ch.df["5lagy_dc"].to_numpy()
    p1 = ch.df["5lagy_tdc"].to_numpy()
    assert not np.all(p0 == p1)

    # Now verify that if you modify the times by going outside the range trained on, then applying the
    # TDC step to that data changes nothing.
    tdc_step = ch.steps[-1]
    times = ch.df["timestamp"]
    delta_t = datetime.timedelta(days=10)
    df2 = tdc_step.calc_from_df(ch.df.drop("5lagy_tdc").with_columns(timestamp=times + delta_t))
    p2 = df2["5lagy_tdc"].to_numpy()
    assert np.all(p0 == p2)
