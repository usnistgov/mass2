import numpy as np
import os
import pytest
import polars as pl
from polars.testing import assert_frame_equal

import mass2
import pulsedata
import tempfile
import pathlib


def test_ljh_to_polars():
    p = pulsedata.pulse_noise_ljh_pairs["20230626"]
    ljh_noise = mass2.LJHFile.open(p.noise_folder / "20230626_run0000_chan4102.ljh")
    _df_noise, _header_df_noise = ljh_noise.to_polars()
    ljh = mass2.LJHFile.open(p.pulse_folder / "20230626_run0001_chan4102.ljh")
    _df, _header_df = ljh.to_polars()


def dummy_channel(npulses=100, seed=4, signal=np.zeros(50, dtype=np.int16), ch_num: int = 0):
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


def test_ljh_fractional_record(tmp_path):
    "Verify that it's allowed to open an LJH file with an non-integer # of binary records"
    # It should not be an error to open an LJH file with a non-integer number of records.
    # That situation might occur when the file is still being written, depending on how the
    # writer handles write-buffering.

    # Specifically, copy the LJH file through the first `npulses` binary records, plus exactly
    # half of the next record. Check that the resulting file can be opened.
    # Then later add enough raw data to have `2*npulses` records. Make sure it can be re-opened.
    npulses = 10
    p = pulsedata.pulse_noise_ljh_pairs["20230626"]
    ljh = mass2.LJHFile.open(p.pulse_folder / "20230626_run0001_chan4102.ljh")
    assert ljh.npulses >= 2 * npulses
    binary_size1 = int((npulses + 0.5) * ljh.pulse_size_bytes)
    binary_size2 = (2 * npulses) * ljh.pulse_size_bytes
    total_size1 = binary_size1 + ljh.header_size

    input_file_path = ljh.filename
    ragged_ljh_file_path = tmp_path / "test_file.ljh"

    with open(input_file_path, "rb") as source_file:
        data_to_copy_initially = source_file.read(total_size1)
        data_to_append_later = source_file.read(binary_size2 - binary_size1)
    with open(ragged_ljh_file_path, "wb") as destination_file:
        destination_file.write(data_to_copy_initially)

    ljh2 = mass2.LJHFile.open(ragged_ljh_file_path)
    assert ljh2.npulses == npulses
    assert ljh2.header_size == ljh.header_size
    assert ljh2.pulse_size_bytes * ljh2.npulses + ljh2.header_size < os.path.getsize(ragged_ljh_file_path)
    for i in range(npulses):
        assert np.all(ljh2.read_trace(i) == ljh.read_trace(i))

    # Now extend the file to contain 2*npulses binary records
    with open(ragged_ljh_file_path, "ab") as destination_file:
        destination_file.write(data_to_append_later)

    # Reopen it.
    ljh3 = ljh2.reopen_binary()
    assert ljh3.npulses == 2 * npulses
    assert ljh3.header_size == ljh.header_size
    assert ljh3.pulse_size_bytes * ljh3.npulses + ljh3.header_size == os.path.getsize(ragged_ljh_file_path)
    for i in range(npulses):
        assert np.all(ljh3.read_trace(i) == ljh.read_trace(i))


def test_follow_mass_filtering_rst():  # noqa: PLR0914
    # following https://github.com/usnistgov/mass/blob/master/doc/filtering.rst

    rng = np.random.default_rng(3)

    # make a pulse and call mass2.core.FilterMaker directly
    # test that the calculated values are correct per the mass docs
    n = 504
    Maxsignal = 1000.0
    sigma_noise = 1.0
    tau = [0.05, 0.25]
    t = np.linspace(-1, 1, n)
    npre = (t < 0).sum()
    signal = np.exp(-t / tau[1]) - np.exp(-t / tau[0])
    signal[t <= 0] = 0
    signal *= Maxsignal / signal.max()

    noise_covar = np.zeros(n)
    noise_covar[0] = sigma_noise**2
    maker = mass2.core.FilterMaker(signal, npre, noise_covar, peak=Maxsignal)
    mass_filter = maker.compute_5lag()

    assert mass_filter.nominal_peak == pytest.approx(1000, rel=1e-2)
    assert mass_filter.variance**0.5 == pytest.approx(0.1549, rel=1e-3)
    assert mass_filter.predicted_v_over_dv == pytest.approx(2741.65, rel=1e-3)
    assert mass_filter.filter_records(signal)[0] == pytest.approx(Maxsignal)

    # then compare to the equivalent code in moss
    # 1. generate noise with the same covar
    # 2. make a channel and noise channel
    # 3. call filter5lag
    # 4. check outputs match and make sense

    # 250 pulses of length 504
    # noise that wil have covar of the form [1, 0, 0, 0, ...]
    npulses = 250
    noise_traces = rng.standard_normal((npulses, n))
    pulse_traces = np.tile(signal, (npulses, 1)) + noise_traces
    header_df = pl.DataFrame({"continuous": [True]})
    frametime_s = 1e-5
    df_noise = pl.DataFrame({"pulse": noise_traces})
    noise_ch = mass2.NoiseChannel(df_noise, header_df, frametime_s)
    header = mass2.ChannelHeader(
        "dummy for test",
        data_source=None,
        ch_num=0,
        frametime_s=frametime_s,
        n_presamples=n // 2,
        n_samples=n,
        df=header_df,
    )
    df = pl.DataFrame({"pulse": pulse_traces})
    ch = mass2.Channel(df, header, npulses=npulses, noise=noise_ch)
    ch = ch.filter5lag()
    step: mass2.core.OptimalFilterStep = ch.steps[-1]
    assert isinstance(step, mass2.core.OptimalFilterStep)
    filter: mass2.core.Filter = step.filter
    assert filter.predicted_v_over_dv == pytest.approx(mass_filter.predicted_v_over_dv, rel=1e-2)
    # test that the mass normaliztion in place
    # a pulse filtered value (5lagy) should roughly equal its peak height
    assert np.mean(ch.df["5lagy"].to_numpy()) == pytest.approx(Maxsignal, rel=1e-2)
    # compare v_dv achieved (signal/fwhm) to predicted using 2.355*std=fwhm
    assert Maxsignal / (2.355 * np.std(ch.df["5lagy"].to_numpy())) == pytest.approx(mass_filter.predicted_v_over_dv, rel=5e-2)
    assert filter._filter_type == "5lag"

    assert isinstance(ch.last_avg_pulse, np.ndarray)
    assert isinstance(ch.last_noise_autocorrelation, np.ndarray)
    assert isinstance(ch.last_noise_psd[1], np.ndarray)


def test_noise_autocorr():
    rng = np.random.default_rng()
    header_df = pl.DataFrame()
    frametime_s = 1e-5
    # 250 pulses of length 500
    # noise that wil have covar of the form [1, 0, 0, 0, ...]
    noise_traces = rng.standard_normal((250, 500))
    df_noise = pl.DataFrame({"pulse": noise_traces})
    noise_ch = mass2.NoiseChannel(df_noise, header_df, frametime_s)
    assert len(noise_ch.df) == 250
    assert len(noise_ch.df["pulse"][0]) == 500
    noise_autocorr_mass = mass2.core.noise_algorithms.calc_discontinuous_autocorrelation(noise_traces)
    assert len(noise_autocorr_mass) == 500
    assert noise_autocorr_mass[0] == pytest.approx(1, rel=1e-1)
    assert np.mean(np.abs(noise_autocorr_mass[1:])) == pytest.approx(0, abs=1e-2)

    ac_direct = mass2.core.noise_algorithms.calc_continuous_autocorrelation(noise_traces, n_lags=500)
    assert len(ac_direct) == 500
    assert ac_direct[0] == pytest.approx(1, rel=1e-1)
    assert np.mean(np.abs(ac_direct[1:])) == pytest.approx(0, abs=1e-2)

    spect = noise_ch.spectrum()
    assert len(spect.autocorr_vec) == 500
    assert spect.autocorr_vec[0] == pytest.approx(1, rel=3e-2)
    assert np.mean(np.abs(spect.autocorr_vec[1:])) == pytest.approx(0, abs=1e-2)


def test_noise_psd():
    rng = np.random.default_rng(1)
    header_df = pl.DataFrame()
    frametime_s = 0.5
    # 250 pulses of length 500
    # noise that wil have 1 arb/Hz value
    # In the case of white noise, the power spectral density (in V²/Hz) is simply the variance of the noise:
    # PSD = sigma**2/delta_f
    # sigma**2 = 1
    # delta_f == 1
    # PSD = 1/Hz
    noise_traces = rng.standard_normal((1000, 500))
    df_noise = pl.DataFrame({"pulse": noise_traces})
    noise_ch = mass2.NoiseChannel(df=df_noise, header_df=header_df, frametime_s=frametime_s)
    assert noise_ch.frametime_s == frametime_s

    # segfactor is the number of pulses
    f_mass, psd_mass = mass2.mathstat.power_spectrum.computeSpectrum(noise_traces.ravel(), segfactor=1000, dt=frametime_s)
    assert len(f_mass) == 251  # half the length of the noise traces + 1
    expect = np.ones(251)
    assert np.allclose(psd_mass, expect, atol=0.15)

    psd_raw_periodogram = mass2.core.noise_algorithms.noise_psd_periodogram(noise_traces, dt=frametime_s)
    assert len(psd_raw_periodogram.frequencies) == 251  # half the length of the noise traces + 1
    assert np.allclose(f_mass, psd_raw_periodogram.frequencies)
    assert np.allclose(psd_raw_periodogram.psd[1:-1], expect[1:-1], atol=0.15)
    assert psd_raw_periodogram.psd[0] == pytest.approx(0.5, rel=1e-1)  # scipy handles the 0 bin and last bin differently
    assert psd_raw_periodogram.psd[-1] == pytest.approx(0.5, rel=1e-1)

    psd_raw = mass2.core.noise_algorithms.calc_noise_result(noise_traces, continuous=True, dt=frametime_s)
    assert len(psd_raw.frequencies) == 251  # half the length of the noise traces + 1
    assert np.allclose(f_mass, psd_raw.frequencies)
    assert np.allclose(psd_raw.psd[1:-1], expect[1:-1], atol=0.15)

    psd = noise_ch.spectrum()
    assert len(psd.frequencies) == 251
    assert np.allclose(psd_raw.frequencies[:5], psd.frequencies[:5])
    assert np.allclose(psd_raw.psd, psd.psd)


def test_get_pulses_2d():
    rng = np.random.default_rng(1)
    header_df = pl.DataFrame()
    frametime_s = 0.5
    # 1000 pulses of length 500
    noise_traces = rng.standard_normal((10, 5))
    df_noise = pl.DataFrame({"pulse": noise_traces})
    noise_ch = mass2.NoiseChannel(df=df_noise, header_df=header_df, frametime_s=frametime_s)
    pulses = noise_ch.get_records_2d()
    assert pulses.shape[0] == 10  # npulses
    assert pulses.shape[1] == 5  # length of pulses


def test_ravel_behavior():
    # noise_algorithms.calc_noise_result relies on this behavior
    # 10 pulses of length 5
    # first pulse = a[0,:]==[0 1 2 3 4]
    a = np.arange(50).reshape(10, 5)
    assert np.allclose(a[0, :], np.arange(5))
    assert np.allclose(a.ravel(), np.arange(50))


def test_noise_psd_ordering_should_be_extended_to_colored_noise():
    header_df = pl.DataFrame()
    frametime_s = 0.5
    noise_traces = np.tile(np.arange(10), (5, 1))
    assert np.allclose(noise_traces[0, :], np.arange(10))
    assert np.allclose(noise_traces.shape, np.array([5, 10]))
    df_noise = pl.DataFrame({"pulse": noise_traces})
    noise_ch = mass2.NoiseChannel(df=df_noise, header_df=header_df, frametime_s=frametime_s)
    assert noise_ch.frametime_s == frametime_s

    # segfactor is the number of pulses
    f_mass, psd_mass = mass2.mathstat.power_spectrum.computeSpectrum(noise_traces.ravel(), segfactor=5, dt=frametime_s)
    assert len(f_mass) == 6  # half the length of the noise traces + 1
    # expect = np.ones(6)

    psd_raw_periodogram = mass2.core.noise_algorithms.noise_psd_periodogram(noise_traces, dt=frametime_s)
    assert len(psd_raw_periodogram.frequencies) == 6  # half the length of the noise traces + 1
    assert np.allclose(f_mass, psd_raw_periodogram.frequencies)
    assert np.allclose(psd_raw_periodogram.psd[1:-1], psd_mass[1:-1], atol=0.15)

    psd_raw = mass2.core.noise_algorithms.calc_noise_result(noise_traces, continuous=False, dt=frametime_s)
    assert len(psd_raw.frequencies) == 6  # half the length of the noise traces + 1
    assert np.allclose(f_mass, psd_raw.frequencies)
    assert np.allclose(psd_raw.psd[1:-1], psd_mass[1:-1], atol=0.15)

    psd = noise_ch.spectrum(excursion_nsigma=1e100)
    assert len(psd.frequencies) == 6
    assert np.allclose(psd_raw.frequencies[:5], psd.frequencies[:5])
    assert np.allclose(psd_raw.psd, psd.psd)


def test_concat_dfs_with_concat_state():
    df1 = pl.DataFrame({"a": [1, 2, 3]})
    df2 = pl.DataFrame({"a": [7, 8]})
    df_concat = mass2.core.misc.concat_dfs_with_concat_state(df1, df2)
    assert df_concat.shape == (5, 2)
    assert df_concat["concat_state"].to_list() == [0] * 3 + [1] * 2
    df_concat2 = mass2.core.misc.concat_dfs_with_concat_state(df_concat, df2)
    assert df_concat2.shape == (7, 2)


def test_col_map_step():
    ch = dummy_channel()

    def std_of_pulses_chunk(pulse):
        return np.std(pulse)

    ch2 = ch.with_column_map_step("pulse", "std_of_pulses", std_of_pulses_chunk)
    assert ch2.df["std_of_pulses"][0] == np.std(ch2.df["pulse"].to_numpy()[0, :])
    step = ch2.steps[-1]
    assert step.inputs == ["pulse"]
    assert step.output == ["std_of_pulses"]


def test_pretrig_mean_jump_fix_step():
    ch = dummy_channel()
    pretrig_mean = np.arange(len(ch.df)) % 50 + 725
    ch = ch.with_columns(pl.DataFrame({"pretrig_mean": pretrig_mean}))
    ch2 = ch.correct_pretrig_mean_jumps(period=50)
    assert "pulse" in ch2.df.columns
    assert all(np.diff(ch2.df["ptm_jf"].to_numpy()) == 1)
    step = ch2.steps[-1]
    assert step.inputs == ["pretrig_mean"]
    assert step.output == ["ptm_jf"]
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpfilename = os.path.join(tmpdir, "steps.pkl")
        ch2.save_recipes(tmpfilename)
        steps2 = mass2.misc.unpickle_object(tmpfilename)
        assert len(steps2) == 1
        assert isinstance(steps2[0][0], mass2.core.recipe.PretrigMeanJumpFixStep)


def test_extract_column_names_from_polars_expr():
    extract = mass2.core.misc.extract_column_names_from_polars_expr
    assert set(extract(pl.col("a"))) == set(["a"])
    assert set(extract(pl.col("a") + pl.col("b"))) == set(["a", "b"])
    assert set(extract(pl.col("a") * pl.col("b"))) == set(["a", "b"])


def test_select_step():
    ch = dummy_channel()
    ch = ch.with_columns(pl.DataFrame({"a": np.arange(len(ch.df)), "b": np.arange(len(ch.df)) * 2}))
    ch2 = ch.with_select_step({"a*5": pl.col("a") * 5, "a+b": pl.col("a") + pl.col("b")})
    assert "pulse" in ch2.df.columns
    assert all(ch2.df["a*5"].to_numpy() == ch.df["a"].to_numpy() * 5)
    assert all(ch2.df["a+b"].to_numpy() == ch.df["a"].to_numpy() + ch.df["b"].to_numpy())
    step = ch2.steps[-1]
    assert set(step.inputs) == set(["a", "b"])
    assert set(step.output) == set(["a*5", "a+b"])
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpfilename = os.path.join(tmpdir, "steps.pkl")
        ch2.save_recipes(tmpfilename)
        steps2 = mass2.misc.unpickle_object(tmpfilename)
        assert len(steps2) == 1
        assert isinstance(steps2[0][0], mass2.core.recipe.SelectStep)


def test_filtering_steps():
    "Make sure we can compute and apply both 5-lag and ATS-type optimal filters."
    t = np.arange(-25, 25)
    signal = 10000 * (np.exp(-t / 12.0) - np.exp(-t / 3.0))
    signal[t < 0] = 0
    ch = dummy_channel(npulses=100, signal=signal)
    ch = ch.filter5lag(f_3db=20000)
    ch = ch.summarize_pulses()
    ch = ch.filterATS(f_3db=20000)
    for field in ("5lagy", "5lagx", "ats_x", "ats_y"):
        assert not (np.allclose(ch.df[field].to_numpy().mean(), 0))


def test_categorize_step():
    ch = dummy_channel(npulses=10)
    ch = ch.with_columns(pl.DataFrame({"a": np.arange(len(ch.df)), "b": np.arange(len(ch.df)) * 2}))
    category_condition_dict = {
        "alessthan5": pl.col("a") < 5,
        "b10": pl.col("b") == 10,
    }
    ch2 = ch.with_categorize_step(category_condition_dict=category_condition_dict)
    assert "pulse" in ch2.df.columns
    step = ch2.steps[-1]
    assert set(step.inputs) == set(["a", "b"])
    assert step.output == ["category"]
    df = ch2.df.with_columns(pl.Series("expected", ["alessthan5"] * 5 + ["b10"] + ["fallback"] * 4))
    assert (df["expected"] == df["category"].cast(str)).all()
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpfilename = os.path.join(tmpdir, "steps.pkl")
        ch2.save_recipes(tmpfilename)
        steps2 = mass2.misc.unpickle_object(tmpfilename)
        assert len(steps2) == 1
        assert isinstance(steps2[0][0], mass2.core.recipe.CategorizeStep)


def test_include_exclude():
    "Check that both the include and exclude lists work as intended"
    p = pulsedata.pulse_noise_ljh_pairs["20230626"]
    data9 = mass2.Channels.from_ljh_folder(p.pulse_folder, p.noise_folder, exclude_ch_nums=[4102])
    assert set(data9.channels.keys()) == {4109}
    data2 = mass2.Channels.from_ljh_folder(p.pulse_folder, p.noise_folder, include_ch_nums=[4102])
    assert set(data2.channels.keys()) == {4102}


def test_steps():
    "Apply some steps, and be sure that `Recipe.trim_dead_ends(...) works"

    def squareme(d):
        return d**2

    # Perform 5 offical Recipe: summarize, filter, a pointless "squareme" step, drift correction, and another pointless one.
    def _do_steps(ch: mass2.Channel) -> mass2.Channel:
        return (
            ch.summarize_pulses()
            .with_good_expr_pretrig_rms_and_postpeak_deriv(8, 8)
            .filter5lag(f_3db=10000)
            .with_column_map_step("pretrig_rms", "pointless_pretrig_meansq", squareme)
            .driftcorrect(indicator_col="pretrig_mean", uncorrected_col="5lagy", use_expr=True)
            .with_column_map_step("postpeak_deriv", "pointless_otherthing", squareme)
        )

    p = pulsedata.pulse_noise_ljh_pairs["20230626"]
    data = mass2.Channels.from_ljh_folder(p.pulse_folder, p.noise_folder, exclude_ch_nums=[4102])
    data = data.map(_do_steps)
    ch = data.channels[4109]

    # Check that the result has 5 steps
    steps = ch.steps
    assert len(steps) == 5

    def is_in_calsteps(x: mass2.core.RecipeStep, steps: mass2.core.Recipe) -> bool:
        """An approximate test whether RecipeStep `x` is in the RecipeStep chain `steps`, testing only equality of
        name, inputs, outputs, rather than identity. (We don't want to check identity, because the RecipeStep object
        may have been changed by a step.drop_debug() operation.)
        """
        for s in steps:
            if x.name == s.name and set(x.inputs) == set(s.inputs) and set(x.output) == set(s.output):
                return True
        return False

    # Check that keeping only 5lagy_dc means step 2 is trimmed
    trim_steps = steps.trim_dead_ends(["5lagy_dc"])
    assert len(trim_steps) == 3
    assert trim_steps[1].spectrum is None
    for i, expect in enumerate((True, True, False, True, False)):
        assert is_in_calsteps(steps[i], trim_steps) == expect

    # Check that keeping 5lagy_dc and some other things don't change the trim result
    trim_steps = steps.trim_dead_ends(["5lagy_dc", "pretrig_rms", "5lagx"])
    assert trim_steps[1].spectrum is None
    assert len(trim_steps) == 3
    for i, expect in enumerate((True, True, False, True, False)):
        assert is_in_calsteps(steps[i], trim_steps) == expect

    # Check that keeping only pointless_pretrig_meansq means only steps 0 and 2 survive
    trim_steps = steps.trim_dead_ends(["pointless_pretrig_meansq"])
    assert len(trim_steps) == 2
    assert trim_steps[1] is steps[2]
    for i, expect in enumerate((True, False, True, False, False)):
        assert is_in_calsteps(steps[i], trim_steps) == expect

    with pytest.raises(ValueError):
        steps.trim_dead_ends("this field doesn't exist")


def test_save_analysis(tmpdir):
    """Test save and load analysis features, including a bad channel, for dummy data."""
    ch_num = 94
    bad_num = 95
    ch = dummy_channel(ch_num=ch_num)
    ch = ch.summarize_pulses().with_good_expr_pretrig_rms_and_postpeak_deriv()
    ch2 = dummy_channel(ch_num=bad_num)
    ch2 = ch2.summarize_pulses().with_good_expr_pretrig_rms_and_postpeak_deriv()
    bch = ch2.as_bad(None, "testing that bad channels also get saved/restored", backtrace=None)
    data = mass2.Channels({ch_num: ch}, description="dummy dataset", bad_channels={bad_num: bch})

    dir = pathlib.Path(tmpdir)
    savefile = dir / "test_save"
    actual_savefile = savefile.with_suffix(".zip")
    data.save_analysis(savefile)
    data2 = mass2.Channels.load_analysis(actual_savefile)

    # Verify that the good channel's data is restored
    # It's a dummy channel, not ljh-backed, so the pulse data will be gone.
    restored_ch = data2.channels[ch_num]
    assert len(restored_ch.df) == len(ch.df)
    assert restored_ch.header.ch_num == ch_num
    assert_frame_equal(restored_ch.df, ch.df.drop("pulse"), check_column_order=False)

    restored_ch2 = data2.bad_channels[bad_num]
    assert len(restored_ch2.ch.df) == len(ch2.df)
    assert restored_ch2.ch.header.ch_num == bad_num
    assert_frame_equal(restored_ch2.ch.df, ch2.df.drop("pulse"))


def test_save_analysis_with_ljh(tmpdir):
    """Test save and load analysis features for LJH-based data, including restoration of raw data columns."""

    def _do_steps(ch: mass2.Channel) -> mass2.Channel:
        return ch.summarize_pulses().with_good_expr_pretrig_rms_and_postpeak_deriv(8, 8)

    p = pulsedata.pulse_noise_ljh_pairs["20230626"]
    data = mass2.Channels.from_ljh_folder(p.pulse_folder, p.noise_folder, limit=5000, exclude_ch_nums=[4102])
    data = data.map(_do_steps)
    ch = data.channels[4109]

    dir = pathlib.Path(tmpdir)
    savefile = dir / "test_save"
    actual_savefile = savefile.with_suffix(".zip")
    data.save_analysis(savefile)
    data2 = mass2.Channels.load_analysis(actual_savefile)

    # Verify that the good channel's data is restored
    # It's an ljh-backed channel, so the raw pulse and timing data should be restored, too.
    restored_ch = data2.channels[4109]
    assert restored_ch.header.ch_num == 4109
    assert len(restored_ch.df) == len(ch.df)
    assert_frame_equal(restored_ch.df, ch.df, check_column_order=False)
