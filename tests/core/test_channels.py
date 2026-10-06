import numpy as np
import os
import pytest
import polars as pl
from polars.testing import assert_frame_equal
import dataclasses


import mass2
import pulsedata
import tempfile
import pathlib
from mass2.core.misc import PulseDataFromNumpy


def test_ljh_to_polars():
    p = pulsedata.pulse_noise_ljh_pairs["20230626"]
    ljh_noise = mass2.LJHFile.open(p.noise_folder / "20230626_run0000_chan4102.ljh")
    _df_noise, _header_df_noise = ljh_noise.to_polars()
    ljh = mass2.LJHFile.open(p.pulse_folder / "20230626_run0001_chan4102.ljh")
    _df, _header_df = ljh.to_polars()


def dummy_dataframe(npulses: int) -> pl.DataFrame:
    """Generate a dataframe with `npulses` rows. The Python API requires keeping a column in it, or it will
    have zero rows."""
    idx = np.arange(npulses)
    return pl.DataFrame({"_dummy": idx, "subframecount": idx * 100000})


def dummy_channel(npulses=100, seed=4, signal=np.zeros(50, dtype=np.int16), ch_num: int = 0):
    rng = np.random.default_rng(seed)
    n = len(signal)
    noise_traces = np.asarray(rng.standard_normal((npulses, n)) * 20 + 5000, dtype=np.int16)
    pulse_traces = np.outer(rng.uniform(0.8, 1.2, size=npulses), signal).astype(np.int16)
    header_df = pl.DataFrame()
    frametime_s = 1e-5
    df_noise = dummy_dataframe(len(noise_traces))
    noise_ch = mass2.NoiseChannel(df_noise, header_df, frametime_s, pulseframer=PulseDataFromNumpy(noise_traces))
    header = mass2.ChannelHeader(
        "dummy for test",
        data_source=None,
        ch_num=ch_num,
        frametime_s=frametime_s,
        n_presamples=n // 2,
        n_samples=n,
        df=header_df,
    )
    pulseframer = PulseDataFromNumpy(pulse_traces + noise_traces)
    df = dummy_dataframe(pulseframer.npulses)
    ch = mass2.Channel(df, header, npulses=npulses, noise=noise_ch, pulseframer=pulseframer)
    return ch


def test_with_columns():
    """Verify 3 different syntax choices for adding columns in `mass2.Channel.with_columns()`.
    See issue #139 for more context. Use
    1. Named arguments, either pl.Expr or numpy array-like
    2. Arguments as a sequence of pl.Series
    3. A full pl.DataFrame
    """
    ch = dummy_channel().summarize_pulses()
    rt = ch.df["pulse_rms"].to_numpy() ** 0.5
    df = pl.DataFrame({"F": pl.Series(rt), "G": pl.Series(rt)})
    ch = (
        ch.with_columns(A=pl.col("pulse_rms").sqrt(), B=pl.col("pulse_rms") ** 0.5, C=rt)
        .with_columns(pl.col("pulse_rms").sqrt().alias("D"), pl.col("pulse_rms").sqrt().alias("E"))
        .with_columns(df)
    )
    for columnname in "ABCDEFG":
        assert np.allclose(ch.df[columnname].to_numpy(), rt)


def test_combine_channels():
    # Check Channel.combine_channels()
    N1, N2 = 70, 30
    ch1 = dummy_channel(npulses=N1)
    ch2 = dummy_channel(npulses=N2)
    d1 = {
        "set1": ch1,
        "set2": ch2,
    }
    ch = mass2.Channel.combine_channels("sourcenumber", d1)
    assert len(ch1.df) == N1
    assert len(ch2.df) == N2
    assert len(ch.df) == N1 + N2
    assert "sourcenumber" not in ch2.df.columns
    assert "sourcenumber" in ch.df.columns
    assert len(ch.df.filter(pl.col("sourcenumber") == "set1")) == N1
    assert len(ch.df.filter(pl.col("sourcenumber") == "set2")) == N2

    # Check Channels.combine_channels()
    data1 = mass2.Channels.from_oneChannel(ch1)
    data2 = mass2.Channels.from_oneChannel(ch2)
    d2 = {
        "sampleA": data1,
        "sampleB": data2,
    }
    data = mass2.Channels.combine_channels("samplecode", d2)
    assert data1.ch0.npulses == N1
    assert data2.ch0.npulses == N2
    assert data.ch0.npulses == N1 + N2
    assert "samplecode" not in data1.ch0.df.columns
    assert "samplecode" in data.ch0.df.columns
    assert len(data.ch0.df.filter(pl.col("samplecode") == "sampleA")) == N1
    assert len(data.ch0.df.filter(pl.col("samplecode") == "sampleB")) == N2


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
    df_noise = dummy_dataframe(npulses)
    noise_ch = mass2.NoiseChannel(df_noise, header_df, frametime_s, PulseDataFromNumpy(noise_traces))
    header = mass2.ChannelHeader(
        "dummy for test",
        data_source=None,
        ch_num=0,
        frametime_s=frametime_s,
        n_presamples=n // 2,
        n_samples=n,
        df=header_df,
    )
    df = dummy_dataframe(npulses)
    pulseframer = PulseDataFromNumpy(pulse_traces)
    ch = mass2.Channel(df, header, npulses=npulses, noise=noise_ch, pulseframer=pulseframer)
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
    psd = ch.last_noise_psd
    assert psd is not None
    assert isinstance(psd[1], np.ndarray)
    assert isinstance(ch.last_v_over_dv, float)


def test_noise_autocorr():
    rng = np.random.default_rng()
    header_df = pl.DataFrame()
    frametime_s = 1e-5
    # 250 pulses of length 500
    # noise that wil have covar of the form [1, 0, 0, 0, ...]
    npulses = 250
    noise_traces = rng.standard_normal((npulses, 500))
    df_noise = dummy_dataframe(npulses)
    noise_ch = mass2.NoiseChannel(df_noise, header_df, frametime_s, PulseDataFromNumpy(noise_traces))
    assert len(noise_ch.df) == 250
    assert noise_ch.pulseframer is not None
    assert len(noise_ch.pulseframer.load_raw_pulse(0)["pulse"]) == 500
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
    npulses = 1000
    noise_traces = rng.standard_normal((npulses, 500))
    df_noise = dummy_dataframe(npulses)
    noise_ch = mass2.NoiseChannel(
        df=df_noise, header_df=header_df, frametime_s=frametime_s, pulseframer=PulseDataFromNumpy(noise_traces)
    )
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
    # 10 pulses of length 5
    npulses = 10
    noise_traces = rng.standard_normal((npulses, 5))
    df_noise = dummy_dataframe(npulses)
    noise_ch = mass2.NoiseChannel(
        df=df_noise, header_df=header_df, frametime_s=frametime_s, pulseframer=PulseDataFromNumpy(noise_traces)
    )
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
    pulse_len = 10
    nfreq = 1 + pulse_len // 2
    npulses = 5
    noise_traces = np.tile(np.arange(10), (npulses, 1))
    assert np.allclose(noise_traces[0, :], np.arange(pulse_len))
    assert np.allclose(noise_traces.shape, np.array([npulses, pulse_len]))
    df_noise = dummy_dataframe(npulses)
    noise_ch = mass2.NoiseChannel(
        df=df_noise, header_df=header_df, frametime_s=frametime_s, pulseframer=PulseDataFromNumpy(noise_traces)
    )
    assert noise_ch.frametime_s == frametime_s

    f_mass, psd_mass = mass2.mathstat.power_spectrum.computeSpectrum(noise_traces.ravel(), segfactor=npulses, dt=frametime_s)
    assert len(f_mass) == nfreq

    psd_raw_periodogram = mass2.core.noise_algorithms.noise_psd_periodogram(noise_traces, dt=frametime_s)
    assert len(psd_raw_periodogram.frequencies) == nfreq
    assert np.allclose(f_mass, psd_raw_periodogram.frequencies)
    assert np.allclose(psd_raw_periodogram.psd[1:-1], psd_mass[1:-1], atol=0.15)

    psd_raw = mass2.core.noise_algorithms.calc_noise_result(noise_traces, continuous=False, dt=frametime_s)
    assert len(psd_raw.frequencies) == nfreq
    assert np.allclose(f_mass, psd_raw.frequencies)
    assert np.allclose(psd_raw.psd[1:-1], psd_mass[1:-1], atol=0.15)

    psd = noise_ch.spectrum(excursion_nsigma=1e100)
    assert len(psd.frequencies) == nfreq
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
    assert ch.pulseframer is not None
    raw_df = ch.pulseframer.load_raw_chunk(0, ch.npulses)
    ch = dataclasses.replace(ch, df=ch.df.with_columns(raw_df))

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
    ch = ch.with_columns(pretrig_mean=pretrig_mean)
    ch2 = ch.correct_pretrig_mean_jumps(period=50)
    assert all(np.diff(ch2.df["ptm_jf"].to_numpy()) == 1)
    step = ch2.steps[-1]
    assert step.inputs == ["pretrig_mean", "subframecount"]
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
    n = len(ch.df)
    ch = ch.with_columns(a=np.arange(n), b=(2 * np.arange(n)))
    ch2 = ch.with_select_step({"a*5": pl.col("a") * 5, "a+b": pl.col("a") + pl.col("b")})
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
    ch = ch.filter1lag(f_3db=20000)
    ch = ch.summarize_pulses()
    ch = ch.filterATS(f_3db=20000)
    for field in ("5lagy", "5lagx", "1lagy", "ats_x", "ats_y"):
        assert not (np.allclose(ch.df[field].to_numpy().mean(), 0))
    assert np.allclose(ch.df["1lagx"].to_numpy().mean(), 0)


def test_categorize_step():
    ch = dummy_channel(npulses=10)
    n = len(ch.df)
    ch = ch.with_columns(a=np.arange(n), b=(2 * np.arange(n)))
    category_condition_dict = {
        "alessthan5": pl.col("a") < 5,
        "b10": pl.col("b") == 10,
    }
    ch2 = ch.with_categorize_step(category_condition_dict=category_condition_dict)
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
            .driftcorrect(indicator_col="pretrig_mean", uncorrected_col="5lagy", use_expr=pl.lit(True))
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
    assert_frame_equal(restored_ch.df, ch.df, check_column_order=False)

    restored_ch2 = data2.bad_channels[bad_num]
    assert len(restored_ch2.ch.df) == len(ch2.df)
    assert restored_ch2.ch.header.ch_num == bad_num
    assert_frame_equal(restored_ch2.ch.df, ch2.df)


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


def test_change_time_zone():
    p = pulsedata.pulse_noise_ljh_pairs["20230626"]
    filename = p.pulse_folder / "20230626_run0001_chan4109.ljh"
    ch = mass2.Channel.from_ljh(str(filename))

    # Make sure that this test CHANGES time zones. The new zone will be Fiji time.
    # In the unlikely event that you run these tests from Fiji, change to Tokyo time.
    new_tz = "Pacific/Fiji"
    if mass2.core.channel._local_timezone_name == new_tz:
        new_tz = "Pacific/Tokyo"

    df1 = ch.df.with_columns(pl.col(pl.Datetime).dt.convert_time_zone(new_tz))
    step = mass2.core.ChangeTimeZoneStep.new(new_tz)
    ch2 = ch.with_step(step)
    df2 = ch2.df
    assert (df1["timestamp"] == df2["timestamp"]).all()
    assert ch.df["timestamp"].dtype != df2["timestamp"].dtype


def test_channel_mismatched_n_samples():
    ch = dummy_channel()
    assert ch.pulseframer is not None
    raw_df = ch.pulseframer.load_raw_chunk(0, ch.npulses)
    ch = dataclasses.replace(ch, df=ch.df.with_columns(raw_df))
    bad_header = dataclasses.replace(ch.header, n_samples=ch.header.n_samples + 1)
    with pytest.raises(ValueError, match="n_samples"):
        mass2.Channel(ch.df, bad_header, npulses=ch.npulses, noise=ch.noise)


def test_ch_from_numpy():
    "Test that we can read random values from a numpy file"
    nsamp, npulses = 100, 60
    raw = np.random.default_rng().normal(10000, 1000, size=(nsamp, npulses)).astype(np.int16)
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "data.npy")
        np.save(fname, raw)

        ch = mass2.Channel.from_numpy(10000, nsamp // 2, fname, fname, "description", ch_num=5)
        data = mass2.Channels.from_oneChannel(ch)
        assert data.ch0.noise is not None
        assert data.ch0.pulseframer is not None
        assert data.ch0.noise.pulseframer is not None
        raw_df1 = data.ch0.pulseframer.load_raw_chunk(0, npulses)
        raw_df2 = data.ch0.noise.pulseframer.load_raw_chunk(0, npulses)
        for i in range(npulses):
            assert np.all(raw_df1["pulse"][i].to_numpy() == raw[:, i])
            assert np.all(raw_df2["pulse"][i].to_numpy() == raw[:, i])


def test_ch_from_numpy2():
    "Test that we can read actual pulse data from a numpy file"
    pulse_noise_pair = pulsedata.numpy["noise_limited_optical_tes"]
    noisepath = pulse_noise_pair.noise
    pulsepath = pulse_noise_pair.pulse
    rate = 16000.0
    npre = 300
    ch = mass2.Channel.from_numpy(rate, npre, pulsepath, noisepath, "description", ch_num=5)
    data = mass2.Channels.from_oneChannel(ch)
    assert data.ch0.noise is not None
    assert data.ch0.pulseframer is not None
    raw_df = data.ch0.pulseframer.load_raw_chunk(0, ch.npulses)
    for i in range(ch.npulses):
        pulse = raw_df["pulse"][i].to_numpy()
        assert np.all(pulse < 5000) and np.all(pulse > -5000)


def test_flux_jump_correction_non_time_ordered_data():
    """Check for [issue 166](https://github.com/usnistgov/mass2/issues/166)

    Test that flux-jump correction still works even if raw data are re-ordered.
    """
    PERIOD = 4096
    Npulses = 40
    steps = 400
    ptm_jumpy = np.arange(Npulses, dtype=np.float32) * steps + 2000
    assert ptm_jumpy[-1] - ptm_jumpy[0] > PERIOD  # If not, you're not really testing the problem
    ptm_correct = ptm_jumpy.copy()
    ptm_jumpy[10:20] += 2 * PERIOD
    info = {"pretrig_mean": ptm_jumpy, "subframecount": np.arange(Npulses) * 10000000}
    df = pl.DataFrame(info)
    header = mass2.ChannelHeader("", None, 100, 1e-5, 100, 200, pl.DataFrame())
    ch = mass2.Channel(df, header, Npulses)

    # First, make sure that correct_pretrig_mean_jumps works as expected: creating column "ptm_jf"
    # with the corrected values.
    assert np.all(df["pretrig_mean"].to_numpy() == ptm_jumpy)
    ch1 = ch.correct_pretrig_mean_jumps(period=PERIOD)
    assert np.all(ch1.df["pretrig_mean"].to_numpy() == ptm_jumpy)
    assert np.all(ch1.df["ptm_jf"].to_numpy() == ptm_correct)

    # Now test for issue 166, where a time-unordered data set fails.
    shuffled_df = ch.df.sample(fraction=1.0, shuffle=True, seed=91)
    ch2 = dataclasses.replace(ch, df=shuffled_df)

    ch3 = ch2.correct_pretrig_mean_jumps(period=PERIOD)
    sort_idx = ch3.df["subframecount"].to_numpy().argsort()
    assert np.all(ch3.df["ptm_jf"].to_numpy()[sort_idx] == ptm_correct)
