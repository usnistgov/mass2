import numpy as np
import mass2
import polars as pl

# set seed to control shuffle in the function and random errors in make_truth_ph
rng = np.random.default_rng(1)


def make_truth_ph(
    e,
    e_spurious,
    e_err_scale,
    pfit_gain_truth=np.polynomial.Polynomial([6, -1e-6, -1e-10]),
) -> None:
    # return peak heights by inverting a quadratic gain curve and adding energy errors
    cba_truth = pfit_gain_truth.convert().coef
    assert len(cba_truth) == 3

    def energy2ph_truth(energy):
        # ph2energy is equivalent to this with y=energy, x=ph
        # y = x/(c + b*x + a*x^2)
        # so
        # y*c + (y*b-1)*x + a*x^2 = 0
        # and given that we've selected for well formed calibrations,
        # we know which root we want
        c, bb, a = cba_truth * energy
        b = bb - 1
        ph = (-b - np.sqrt(b**2 - 4 * a * c)) / (2 * a)
        return ph

    ph_truth_with_err = np.array([energy2ph_truth(ee) + rng.standard_normal() * e_err_scale for ee in e])
    ph_spurious_with_err = np.array([energy2ph_truth(ee) + rng.standard_normal() * e_err_scale for ee in e_spurious])
    ph = np.hstack([ph_truth_with_err, ph_spurious_with_err])
    return ph, ph_truth_with_err


def test_find_optimal_assignment_many() -> None:
    e = np.arange(1000, 9000, 1000)  # energies of "real" peaks
    e_spurious = [
        100,
        500,
        1500,
        1700,
        7900,
        2200,
        2300,
        2400,
        3300,
        3700,
        4500,
    ]  # energies of spurious or fake peaks
    ph, ph_truth_with_err = make_truth_ph(e=e, e_spurious=e_spurious, e_err_scale=10)
    # ph contains pulseheights of both real and spurious with errs,
    # first all the real peaks in e order, then all the spurious peaks in e_spurious order
    # ph_truth_with_err contains pulseheights of only real peaks, sorted to match the order of e
    # from all peaks (ph) find the one corresponding to energies e
    rng.shuffle(ph)  # in place shuffle
    result = mass2.core.rough_cal.find_optimal_assignment2(ph, e, map(str, e))
    # test that assigned peaks match known peaks of energy e
    assert np.allclose(result.ph_assigned, ph_truth_with_err, rtol=0.001)
    assert np.allclose([result.ph2energy(result.energy2ph(ee)) for ee in e], e)


def test_find_optimal_assignment_1() -> None:
    e = np.array([1000])  # energies of "real" peaks
    e_spurious = [
        100,
        500,
        1500,
        1700,
        7900,
        2200,
        2300,
        2400,
        3300,
        3700,
        4500,
    ]  # energies of spurious or fake peaks
    ph, ph_truth_with_err = make_truth_ph(e=e, e_spurious=e_spurious, e_err_scale=10)
    # ph contains pulseheights of both real and spurious with errs
    # ph_truth_with_err contains pulseheights of only real peaks, sorted to match the order of e
    # from all peaks (ph) find the one corresponding to energies e
    result = mass2.core.rough_cal.find_optimal_assignment2(ph, e, map(str, e))
    # test that assigned peaks match known peaks of energy e
    assert np.allclose(result.ph_assigned, ph_truth_with_err, rtol=0.001)
    assert np.allclose([result.ph2energy(result.energy2ph(ee)) for ee in e], e)


def test_find_optimal_assignment_2() -> None:
    e = np.array([1000, 3000])  # energies of "real" peaks
    e_spurious = [
        100,
        500,
        1500,
        1700,
        7900,
        2200,
        2300,
        2400,
        3300,
        3700,
        4500,
    ]  # energies of spurious or fake peaks
    ph, ph_truth_with_err = make_truth_ph(e=e, e_spurious=e_spurious, e_err_scale=10)
    # ph contains pulseheights of both real and spurious with errs
    # ph_truth_with_err contains pulseheights of only real peaks, sorted to match the order of e
    # from all peaks (ph) find the one corresponding to energies e
    result = mass2.core.rough_cal.find_optimal_assignment2(ph, e, map(str, e))
    # test that assigned peaks match known peaks of energy e
    assert np.allclose(result.ph_assigned, ph_truth_with_err, rtol=0.001)
    assert np.allclose([result.ph2energy(result.energy2ph(ee)) for ee in e], e)


def test_find_optimal_assignment_3() -> None:
    e = np.array([1000, 3000, 5000])  # energies of "real" peaks
    e_spurious = [3300, 3700, 4500]  # energies of spurious or fake peaks
    ph, ph_truth_with_err = make_truth_ph(e=e, e_spurious=e_spurious, e_err_scale=10)
    # ph contains pulseheights of both real and spurious with errs
    # ph_truth_with_err contains pulseheights of only real peaks, sorted to match the order of e
    # from all peaks (ph) find the one corresponding to energies e
    result = mass2.core.rough_cal.find_optimal_assignment2(ph, e, map(str, e))
    # test that assigned peaks match known peaks of energy e
    assert np.allclose(result.ph_assigned, ph_truth_with_err, rtol=0.001)
    assert np.allclose([result.ph2energy(result.energy2ph(ee)) for ee in e], e)


def test_rank_3peak_assignments() -> None:
    e = np.array([1000, 3000, 5000])  # energies of "real" peaks
    e_spurious = [2900, 3700, 4500]  # energies of spurious or fake peaks
    ph, ph_truth_with_err = make_truth_ph(e=e, e_spurious=e_spurious, e_err_scale=10)
    df3peak, _dfe = mass2.core.rough_cal.rank_3peak_assignments(ph, e, map(str, e))
    ph_assigned_top_rank = np.array([df3peak["ph0"][0], df3peak["ph1"][0], df3peak["ph2"][0]])
    assert np.allclose(ph_truth_with_err, ph_assigned_top_rank)


def dummy_channel(npulses=100, seed=4, signal=np.zeros(50, dtype=np.int16), ch_num: int = 0) -> mass2.Channel:
    rng = np.random.default_rng(seed)
    n = len(signal)
    noise_traces = np.asarray(rng.standard_normal((npulses, n)) * 20 + 5000, dtype=np.int16)
    pulse_traces = np.outer(rng.uniform(0.8, 1.2, size=npulses), signal).astype(np.int16)
    header_df = pl.DataFrame()
    frametime_s = 1e-5
    df_noise = pl.DataFrame({"pulse": noise_traces})
    noise_ch = mass2.NoiseChannel(df_noise, header_df, frametime_s)
    header = mass2.ChannelHeader(
        "dummy for test",
        data_source=None,
        ch_num=ch_num,
        frametime_s=frametime_s,
        n_presamples=n // 2,
        n_samples=n,
        df=header_df,
    )
    df = pl.DataFrame({"pulse": pulse_traces + noise_traces})
    ch = mass2.Channel(df, header, npulses=npulses, noise=noise_ch)
    return ch


def test_one_peak_rough_cal_combinatoric() -> None:
    ch = dummy_channel(signal=np.ones(50, dtype=np.int16) * 100)
    ch = ch.summarize_pulses()
    ch = ch.rough_cal_combinatoric(["AlKAlpha"], "peak_value", calibrated_col="energy_peak_value", ph_smoothing_fwhm=5)


def test_one_peak_rough_cal_combinatoric_height_info() -> None:
    ch = dummy_channel(signal=np.ones(50, dtype=np.int16) * 100)
    ch = ch.summarize_pulses()
    ch = ch.rough_cal_combinatoric_height_info(
        ["AlKAlpha"], [[1]], "peak_value", calibrated_col="energy_peak_value", ph_smoothing_fwhm=5
    )


def test_two_peak_rough_cal_combinatoric() -> None:
    """When two peaks happen to have a gain that grows with energy, you will get an energy assignment
    that extrapolates to zero gain for negative-size pulses. This is fine, but it is NOT fine to
    cut pulses that exceed that value (as ALL pulses will exceed it). Tests for issue #95."""
    signal = np.zeros(50, dtype=np.int16)
    signal[25:] = 1000
    chA = dummy_channel(signal=signal)
    chB = dummy_channel(signal=signal * 2)
    df = pl.concat([chA.df, chB.df])

    ch = mass2.Channel(df, chA.header, npulses=len(df), noise=chA.noise)
    ch = ch.summarize_pulses()
    # Try to calibrate with 2 peaks, like with ^153Gd data
    ch = ch.rough_cal_combinatoric([100, 180], "pulse_average", calibrated_col="energy_pulse_average", ph_smoothing_fwhm=5)
    assignment = ch.steps[-1].assignment_result
    assert assignment.phzerogain() < 0
    df_good = ch.df.filter(ch.good_expr)
    assert len(df_good) > 0
