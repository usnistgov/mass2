import numpy as np
import os
import shutil
import pytest
import polars as pl
from polars.testing import assert_frame_equal
import dataclasses


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
        ch
        .with_columns(A=pl.col("pulse_rms").sqrt(), B=pl.col("pulse_rms") ** 0.5, C=rt)
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
    df_noise = pl.DataFrame({"pulse": noise_traces})
    noise_ch = mass2.NoiseChannel(df_noise, header_df, frametime_s)
    noiseresult = noise_ch.spectrum()
    noiseresult.autocorr_vec[:] = 0
    noiseresult.autocorr_vec[0] = sigma_noise**2
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
    step = mass2.core.filter_steps.OptimalFilterStep(
        inputs=["pulse"],
        output=["5lagx", "5lagy"],
        good_expr=ch.good_expr,
        use_expr=pl.lit(True),
        filter=mass_filter,
        spectrum=noiseresult,
        filter_maker=maker,
        transform_raw=ch.transform_raw,
    )
    ch = ch.with_step(step)
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


def test_filter5lag_uses_noise_channel():
    """Test that filter5lag() correctly extracts the noise spectrum from the NoiseChannel.

    test_follow_mass_filtering_rst verifies the filter math with an exact spectrum.
    This test exercises the full code path: filter5lag() -> noise_ch.spectrum() ->
    FilterMaker -> OptimalFilterStep, using enough noise traces (2000) for the
    empirical covariance estimate to be close to the theoretical white-noise value.
    """
    rng = np.random.default_rng(7)
    n = 504
    Maxsignal = 1000.0
    sigma_noise = 1.0
    tau = [0.05, 0.25]
    t = np.linspace(-1, 1, n)
    npre = int((t < 0).sum())
    signal = np.exp(-t / tau[1]) - np.exp(-t / tau[0])
    signal[t <= 0] = 0
    signal *= Maxsignal / signal.max()

    # Theoretical reference filter
    noise_covar = np.zeros(n)
    noise_covar[0] = sigma_noise**2
    theoretical_filter = mass2.core.FilterMaker(signal, npre, noise_covar, peak=Maxsignal).compute_5lag()

    # 2000 noise traces: enough for the empirical autocorrelation to approximate white noise
    noise_traces = rng.standard_normal((2000, n)) * sigma_noise
    header_df = pl.DataFrame({"continuous": [True]})
    frametime_s = 1e-5
    noise_ch = mass2.NoiseChannel(pl.DataFrame({"pulse": noise_traces}), header_df, frametime_s)

    npulses = 250
    pulse_traces = np.tile(signal, (npulses, 1)) + rng.standard_normal((npulses, n)) * sigma_noise
    header = mass2.ChannelHeader(
        "dummy for noise test",
        data_source=None,
        ch_num=0,
        frametime_s=frametime_s,
        n_presamples=npre,
        n_samples=n,
        df=header_df,
    )
    ch = mass2.Channel(pl.DataFrame({"pulse": pulse_traces}), header, npulses=npulses, noise=noise_ch)
    ch = ch.filter5lag()

    step: mass2.core.OptimalFilterStep = ch.steps[-1]
    assert isinstance(step, mass2.core.OptimalFilterStep)

    # The empirically-derived filter should be close to the theoretical optimum (10% tolerance)
    assert step.filter.predicted_v_over_dv == pytest.approx(theoretical_filter.predicted_v_over_dv, rel=0.10)

    # Verify the noise autocorrelation stored on the channel is approximately white
    autocorr = ch.last_noise_autocorrelation
    assert isinstance(autocorr, np.ndarray)
    assert autocorr[0] == pytest.approx(sigma_noise**2, rel=0.10)
    assert np.mean(np.abs(autocorr[1:])) == pytest.approx(0, abs=0.05 * sigma_noise**2)


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
    ch = ch.with_columns(pretrig_mean=pretrig_mean)
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
    n = len(ch.df)
    ch = ch.with_columns(a=np.arange(n), b=(2 * np.arange(n)))
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
            ch
            .summarize_pulses()
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
    # load_pulse=False: dummy channel has no LJH backing, so pulse cannot be restored.
    data2 = mass2.Channels.load_analysis(actual_savefile, load_pulse=False)

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
        for i in range(npulses):
            assert np.all(data.ch0.df["pulse"][i].to_numpy() == raw[:, i])
            assert np.all(data.ch0.noise.df["pulse"][i].to_numpy() == raw[:, i])


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
    for i in range(ch.npulses):
        pulse = data.ch0.df["pulse"][i].to_numpy()
        assert np.all(pulse < 5000) and np.all(pulse > -5000)


def test_drop_pulse():
    """Channel.drop_pulse() removes the pulse column and propagates to noise; is idempotent."""
    ch = dummy_channel().summarize_pulses()  # ensure non-pulse columns exist alongside pulse
    assert "pulse" in ch.df.columns
    assert "pulse" in ch.noise.df.columns

    ch_dropped = ch.drop_pulse()
    assert "pulse" not in ch_dropped.df.columns
    assert "pulse" not in ch_dropped.noise.df.columns
    # npulses must reflect row count of the surviving df (has non-pulse columns, so row count preserved)
    assert ch_dropped.npulses == ch.npulses

    # Calling again must not raise
    ch_dropped2 = ch_dropped.drop_pulse()
    assert "pulse" not in ch_dropped2.df.columns


def test_noise_channel_drop_pulse():
    """NoiseChannel.drop_pulse() removes the pulse column and is idempotent."""
    rng = np.random.default_rng(1)
    noise_traces = rng.standard_normal((10, 20)).astype(np.float32)
    noise_ch = mass2.NoiseChannel(pl.DataFrame({"pulse": noise_traces}), pl.DataFrame(), 1e-5)
    assert "pulse" in noise_ch.df.columns

    dropped = noise_ch.drop_pulse()
    assert "pulse" not in dropped.df.columns
    assert "pulse" not in dropped.drop_pulse().df.columns


def test_with_replacement_df_updates_npulses():
    """with_replacement_df() must update npulses to match the new DataFrame length."""
    ch = dummy_channel(npulses=100)
    new_df = ch.df.head(40)
    ch2 = ch.with_replacement_df(new_df)
    assert ch2.npulses == 40
    assert len(ch2.df) == 40


def test_with_step_drops_pulse_from_history():
    """with_step() must store the previous df without the pulse column in df_history."""
    ch = dummy_channel()
    ch = ch.with_columns(extra_col=np.arange(len(ch.df)))
    assert ch.df_history == []

    ch2 = ch.summarize_pulses()

    assert len(ch2.df_history) == 1
    assert "pulse" not in ch2.df_history[0].columns
    assert "extra_col" in ch2.df_history[0].columns
    # Current df keeps the pulse column
    assert "pulse" in ch2.df.columns


def test_concat_ch_merges_data_sources():
    """concat_ch() must merge and deduplicate data_sources in insertion order."""
    ch1 = dataclasses.replace(dummy_channel(), data_sources=["file_a.ljh"])
    ch2 = dataclasses.replace(dummy_channel(), data_sources=["file_b.ljh"])
    ch3 = dataclasses.replace(dummy_channel(), data_sources=["file_a.ljh", "file_c.ljh"])

    combined = ch1.concat_ch(ch2)
    assert combined.data_sources == ["file_a.ljh", "file_b.ljh"]
    assert combined.npulses == ch1.npulses + ch2.npulses

    # file_a.ljh appears in both — deduplicated, file_c.ljh appended
    combined2 = ch1.concat_ch(ch3)
    assert combined2.data_sources == ["file_a.ljh", "file_c.ljh"]


def test_channel_combine_channels_tracks_data_sources():
    """Channel.combine_channels() must collect data_sources from all constituents."""
    ch1 = dataclasses.replace(dummy_channel(), data_sources=["file_a.ljh"])
    ch2 = dataclasses.replace(dummy_channel(), data_sources=["file_b.ljh"])

    combined = mass2.Channel.combine_channels("run", {"run1": ch1, "run2": ch2})
    assert "file_a.ljh" in combined.data_sources
    assert "file_b.ljh" in combined.data_sources


def test_channels_combine_channels_intersection():
    """Channels.combine_channels() must only keep channels present in ALL constituents."""
    ch0 = dummy_channel(ch_num=0)
    ch1 = dummy_channel(ch_num=1)
    ch2 = dummy_channel(ch_num=2)

    data_a = mass2.Channels({0: ch0, 1: ch1}, description="a")
    data_b = mass2.Channels({1: ch1, 2: ch2}, description="b")

    combined = mass2.Channels.combine_channels("run", {"a": data_a, "b": data_b})
    # Channel 1 is the only one present in both
    assert set(combined.channels.keys()) == {1}


def test_concat_dfs_with_concat_state_categorical():
    """concat_dfs_with_concat_state must preserve Categorical dtype through the concat."""
    df1 = pl.DataFrame({
        "source_file": pl.Series(["a.ljh", "a.ljh"], dtype=pl.Categorical),
        "val": [1, 2],
    })
    df2 = pl.DataFrame({
        "source_file": pl.Series(["b.ljh", "b.ljh"], dtype=pl.Categorical),
        "val": [3, 4],
    })

    result = mass2.core.misc.concat_dfs_with_concat_state(df1, df2)

    assert result.shape == (4, 3)  # source_file, val, concat_state
    assert result["source_file"].dtype == pl.Categorical
    assert result["source_file"].cast(pl.String).to_list() == ["a.ljh", "a.ljh", "b.ljh", "b.ljh"]


def test_save_analysis_trim_pulse_false(tmpdir):
    """save_analysis(trim_pulse=False) embeds pulse in the archive so it survives load without LJH files."""
    ch = dummy_channel(ch_num=0).summarize_pulses()
    original_pulse = np.array(ch.df["pulse"][0])
    data = mass2.Channels({0: ch}, description="test")

    savefile = pathlib.Path(tmpdir) / "embedded_pulse"
    actual_savefile = savefile.with_suffix(".zip")
    data.save_analysis(savefile, trim_pulse=False)

    data2 = mass2.Channels.load_analysis(actual_savefile)
    restored_ch = data2.channels[0]

    assert "pulse" in restored_ch.df.columns
    assert len(restored_ch.df) == len(ch.df)
    assert np.allclose(np.array(restored_ch.df["pulse"][0]), original_pulse)


def test_load_analysis_load_pulse_false(tmpdir):
    """load_analysis(load_pulse=False) returns channels without pulse when the archive has none."""
    ch = dummy_channel(ch_num=0).summarize_pulses()
    data = mass2.Channels({0: ch}, description="test")

    savefile = pathlib.Path(tmpdir) / "no_pulse"
    actual_savefile = savefile.with_suffix(".zip")
    data.save_analysis(savefile)  # default trim_pulse=True

    data2 = mass2.Channels.load_analysis(actual_savefile, load_pulse=False)
    restored_ch = data2.channels[0]

    assert "pulse" not in restored_ch.df.columns
    assert "pretrig_mean" in restored_ch.df.columns


def test_channel_load_pulse_drop_pulse_ljh():
    """Channel.load_pulse() reconstructs the pulse column from the source LJH file."""
    p = pulsedata.pulse_noise_ljh_pairs["20230626"]
    pulse_path = str(p.pulse_folder / "20230626_run0001_chan4109.ljh")
    noise_path = str(p.noise_folder / "20230626_run0000_chan4109.ljh")

    ch_full = mass2.Channel.from_ljh(pulse_path, noise_path)
    assert "pulse" in ch_full.df.columns
    assert "source_file" in ch_full.df.columns

    ch_no_pulse = mass2.Channel.from_ljh(pulse_path, noise_path, load_pulses=False)
    assert "pulse" not in ch_no_pulse.df.columns
    assert "source_file" in ch_no_pulse.df.columns
    assert "source_id" in ch_no_pulse.df.columns
    assert ch_no_pulse.npulses == ch_full.npulses

    ch_restored = ch_no_pulse.load_pulse()
    assert "pulse" in ch_restored.df.columns
    assert len(ch_restored.df) == len(ch_full.df)
    assert np.allclose(ch_full.df["pulse"].to_numpy(), ch_restored.df["pulse"].to_numpy())


def test_noise_channel_load_pulse_ljh():
    """NoiseChannel.load_pulse() rehydrates pulse from the source LJH file."""
    p = pulsedata.pulse_noise_ljh_pairs["20230626"]
    noise_path = str(p.noise_folder / "20230626_run0000_chan4109.ljh")

    noise_full = mass2.NoiseChannel.from_ljh(noise_path)
    assert "pulse" in noise_full.df.columns

    noise_no_pulse = mass2.NoiseChannel.from_ljh(noise_path, load_pulses=False)
    assert "pulse" not in noise_no_pulse.df.columns
    assert "source_file" in noise_no_pulse.df.columns
    assert len(noise_no_pulse.df) == len(noise_full.df)

    noise_restored = noise_no_pulse.load_pulse()
    assert "pulse" in noise_restored.df.columns
    assert len(noise_restored.df) == len(noise_full.df)


def test_requires_pulse_transparent():
    """@requires_pulse lets decorated methods give identical results with or without a pre-loaded pulse."""
    p = pulsedata.pulse_noise_ljh_pairs["20230626"]
    pulse_path = str(p.pulse_folder / "20230626_run0001_chan4109.ljh")
    noise_path = str(p.noise_folder / "20230626_run0000_chan4109.ljh")

    ch_with = mass2.Channel.from_ljh(pulse_path, noise_path)
    # drop_pulse preserves source_file/source_id so load_pulse() can rehydrate
    ch_without = ch_with.drop_pulse()
    assert "pulse" not in ch_without.df.columns

    # summarize_pulses returns a Channel — decorator drops pulse when it was absent
    ch_sum_with = ch_with.summarize_pulses()
    ch_sum_without = ch_without.summarize_pulses()

    assert "pulse" not in ch_sum_without.df.columns
    for col in ("pretrig_mean", "pretrig_rms", "pulse_rms", "postpeak_deriv"):
        assert col in ch_sum_with.df.columns
        assert col in ch_sum_without.df.columns
        assert np.allclose(
            ch_sum_with.df[col].to_numpy(),
            ch_sum_without.df[col].to_numpy(),
        )

    # compute_average_pulse returns NDArray — pulse is NOT stripped from the return value
    avg_with = ch_with.compute_average_pulse()
    avg_without = ch_without.compute_average_pulse()
    assert isinstance(avg_with, np.ndarray)
    assert isinstance(avg_without, np.ndarray)
    assert np.allclose(avg_with, avg_without)


def test_ipc_cache_generation(tmp_path):
    """from_ljh(generate_cache=True) writes .ipc files; use_cache=True reads back identical data."""
    p = pulsedata.pulse_noise_ljh_pairs["20230626"]
    src = p.pulse_folder / "20230626_run0001_chan4109.ljh"
    dst = tmp_path / src.name
    shutil.copy(src, dst)

    ch_ref = mass2.Channel.from_ljh(str(dst), use_cache=False, generate_cache=True)
    ipc_path = dst.with_suffix(".ipc")
    header_ipc_path = dst.with_suffix(".header.ipc")
    assert ipc_path.exists()
    assert header_ipc_path.exists()

    ch_cached = mass2.Channel.from_ljh(str(dst), use_cache=True, generate_cache=False)
    assert "pulse" in ch_cached.df.columns
    assert len(ch_cached.df) == len(ch_ref.df)
    assert np.allclose(ch_cached.df["pulse"].to_numpy(), ch_ref.df["pulse"].to_numpy())


def test_ipc_cache_load_pulses_false(tmp_path):
    """from_ljh with use_cache=True and load_pulses=False reads from cache but drops pulse."""
    p = pulsedata.pulse_noise_ljh_pairs["20230626"]
    src = p.pulse_folder / "20230626_run0001_chan4109.ljh"
    dst = tmp_path / src.name
    shutil.copy(src, dst)

    ch_full = mass2.Channel.from_ljh(str(dst), use_cache=False, generate_cache=True)

    ch_no_pulse = mass2.Channel.from_ljh(str(dst), use_cache=True, load_pulses=False)
    assert "pulse" not in ch_no_pulse.df.columns
    assert "source_file" in ch_no_pulse.df.columns
    assert ch_no_pulse.npulses == ch_full.npulses


def test_ipc_cache_corrupted_falls_back(tmp_path):
    """from_ljh falls back to raw LJH and returns correct data when the cache is corrupt."""
    p = pulsedata.pulse_noise_ljh_pairs["20230626"]
    src = p.pulse_folder / "20230626_run0001_chan4109.ljh"
    dst = tmp_path / src.name
    shutil.copy(src, dst)
    ch_ref = mass2.Channel.from_ljh(str(dst), use_cache=False)

    ipc_path = dst.with_suffix(".ipc")
    header_ipc_path = dst.with_suffix(".header.ipc")

    # Write corrupt .ipc files (wrong schema — ChannelHeader construction will fail)
    pl.DataFrame({"corrupt_col": [1]}).write_ipc(ipc_path)
    pl.DataFrame({"corrupt_col": [1]}).write_ipc(header_ipc_path)
    # Make cache appear newer than source so use_cache tries to read it
    new_mtime = dst.stat().st_mtime + 10
    os.utime(ipc_path, (new_mtime, new_mtime))
    os.utime(header_ipc_path, (new_mtime, new_mtime))

    ch = mass2.Channel.from_ljh(str(dst), use_cache=True)
    assert len(ch.df) == len(ch_ref.df)
    assert "pulse" in ch.df.columns


def test_iter_pulse_batches():
    """iter_pulse_batches() covers all pulses in order, each batch having pulse loaded."""
    p = pulsedata.pulse_noise_ljh_pairs["20230626"]
    pulse_path = str(p.pulse_folder / "20230626_run0001_chan4109.ljh")
    noise_path = str(p.noise_folder / "20230626_run0000_chan4109.ljh")
    ch = mass2.Channel.from_ljh(pulse_path, noise_path, load_pulses=False)
    chunk_size = 2000

    batches = list(ch.iter_pulse_batches(chunk_size=chunk_size))
    assert len(batches) > 1
    for batch in batches:
        assert "pulse" in batch.df.columns
        assert len(batch.df) <= chunk_size
    assert sum(len(b.df) for b in batches) == ch.npulses


def test_channels_map_auto_load_pulse():
    """map(auto_load_pulse=True) retries a failed function with pulse, then drops pulse from result."""
    p = pulsedata.pulse_noise_ljh_pairs["20230626"]
    pulse_path = str(p.pulse_folder / "20230626_run0001_chan4109.ljh")
    noise_path = str(p.noise_folder / "20230626_run0000_chan4109.ljh")

    ch = mass2.Channel.from_ljh(pulse_path, noise_path, load_pulses=False)
    data = mass2.Channels({4109: ch}, description="test")

    def needs_pulse_directly(ch):
        # Accesses pulse column directly — raises ColumnNotFoundError if absent
        arr = np.vstack(ch.df["pulse"].to_list())
        return ch.with_columns(peak_val=arr.max(axis=1))

    # auto_load_pulse=True: retried with pulse loaded, result has pulse dropped
    data_ok = data.map(needs_pulse_directly, auto_load_pulse=True)
    assert "peak_val" in data_ok.channels[4109].df.columns
    assert "pulse" not in data_ok.channels[4109].df.columns

    # auto_load_pulse=False: function fails, channel ends up in bad_channels
    data_bad = data.map(needs_pulse_directly, auto_load_pulse=False, allow_throw=False)
    assert 4109 in data_bad.bad_channels


def test_channels_map_batched():
    """map(batched=True) processes a pulse-less channel in chunks and recombines correctly."""
    p = pulsedata.pulse_noise_ljh_pairs["20230626"]
    pulse_path = str(p.pulse_folder / "20230626_run0001_chan4109.ljh")
    noise_path = str(p.noise_folder / "20230626_run0000_chan4109.ljh")

    # Reference: normal processing with pulse present
    ch_ref = mass2.Channel.from_ljh(pulse_path, noise_path).summarize_pulses()

    # Batched: pulse-less channel — default chunk_size covers all pulses in one batch
    ch_no_pulse = mass2.Channel.from_ljh(pulse_path, noise_path, load_pulses=False)
    data = mass2.Channels({4109: ch_no_pulse}, description="test")
    data_batched = data.map(lambda ch: ch.summarize_pulses(), batched=True)
    ch_batched = data_batched.channels[4109]

    assert len(ch_batched.df) == len(ch_ref.df)
    assert "pretrig_mean" in ch_batched.df.columns
    assert "pulse" not in ch_batched.df.columns
    assert np.allclose(
        ch_ref.df["pretrig_mean"].to_numpy(),
        ch_batched.df["pretrig_mean"].to_numpy(),
    )
