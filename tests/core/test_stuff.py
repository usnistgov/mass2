import numpy as np
import os
import pytest
import polars as pl

import mass2 as mass
import pulsedata


def test_ljh_to_polars():
    p = pulsedata.pulse_noise_ljh_pairs["20230626"]
    ljh_noise = mass.LJHFile.open(p.noise_folder / "20230626_run0000_chan4102.ljh")
    df_noise, header_df_noise = ljh_noise.to_polars()
    ljh = mass.LJHFile.open(p.pulse_folder / "20230626_run0001_chan4102.ljh")
    df, header_df = ljh.to_polars()


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
    ljh = mass.LJHFile.open(p.pulse_folder / "20230626_run0001_chan4102.ljh")
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

    ljh2 = mass.LJHFile.open(ragged_ljh_file_path)
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

    # make a pulse and call mass.FilterMaker directly
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
    maker = mass.FilterMaker(signal, npre, noise_covar, peak=Maxsignal)
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
    noise_traces = rng.standard_normal((250, n))
    pulse_traces = np.tile(signal, (250, 1)) + noise_traces
    header_df = pl.DataFrame()
    frametime_s = 1e-5
    df_noise = pl.DataFrame({"pulse": noise_traces})
    noise_ch = mass.NoiseChannel(df_noise, header_df, frametime_s)
    header = mass.ChannelHeader(
        "dummy for test",
        ch_num=0,
        frametime_s=frametime_s,
        n_presamples=n // 2,
        n_samples=n,
        df=header_df,
    )
    df = pl.DataFrame({"pulse": pulse_traces})
    ch = mass.Channel(df, header, noise=noise_ch)
    ch = ch.filter5lag()
    step: mass.Filter5LagStep = ch.steps[-1]
    assert isinstance(step, mass.Filter5LagStep)
    filter: mass.Filter = step.filter
    assert filter.v_over_dv == pytest.approx(mass_filter.predicted_v_over_dv, rel=1e-2)
    # test that the mass normaliztion in place
    # a pulse filtered value (5lagy) should roughly equal its peak height
    assert np.mean(ch.df["5lagy"].to_numpy()) == pytest.approx(Maxsignal, rel=1e-2)
    # compare v_dv achieved (signal/fwhm) to predicted using 2.355*std=fwhm
    assert Maxsignal / (2.355 * np.std(ch.df["5lagy"].to_numpy())) == pytest.approx(mass_filter.predicted_v_over_dv, rel=5e-2)
    assert filter.filter_type == "mass 5lag"
    assert filter.dt == frametime_s


def test_noise_autocorr():
    rng = np.random.default_rng()
    header_df = pl.DataFrame()
    frametime_s = 1e-5
    # 250 pulses of length 500
    # noise that wil have covar of the form [1, 0, 0, 0, ...]
    noise_traces = rng.standard_normal((250, 500))
    df_noise = pl.DataFrame({"pulse": noise_traces})
    noise_ch = mass.NoiseChannel(df_noise, header_df, frametime_s)
    assert len(noise_ch.df) == 250
    assert len(noise_ch.df["pulse"][0]) == 500
    noise_autocorr_mass = mass.power_spectrum.autocorrelation_broken_from_pulses(noise_traces.T)
    assert len(noise_autocorr_mass) == 500
    assert noise_autocorr_mass[0] == pytest.approx(1, rel=1e-1)
    assert np.mean(np.abs(noise_autocorr_mass[1:])) == pytest.approx(0, abs=1e-2)

    ac_direct = mass.noise_algorithms.autocorrelation(noise_traces, dt=frametime_s).ac
    assert len(ac_direct) == 500
    assert ac_direct[0] == pytest.approx(1, rel=1e-1)
    assert np.mean(np.abs(ac_direct[1:])) == pytest.approx(0, abs=1e-2)

    spect = noise_ch.spectrum()
    assert len(spect.autocorr_vec) == 500
    assert spect.autocorr_vec[0] == pytest.approx(1, rel=1e-2)
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
    noise_ch = mass.NoiseChannel(df=df_noise, header_df=header_df, frametime_s=frametime_s)
    assert noise_ch.frametime_s == frametime_s

    # segfactor is the number of pulses
    f_mass, psd_mass = mass.power_spectrum.computeSpectrum(noise_traces.ravel(), segfactor=1000, dt=frametime_s)
    assert len(f_mass) == 251  # half the length of the noise traces + 1
    expect = np.ones(251)
    assert np.allclose(psd_mass, expect, atol=0.15)

    psd_raw_periodogram = mass.noise_algorithms.noise_psd_periodogram(noise_traces, dt=frametime_s)
    assert len(psd_raw_periodogram.frequencies) == 251  # half the length of the noise traces + 1
    assert np.allclose(f_mass, psd_raw_periodogram.frequencies)
    assert np.allclose(psd_raw_periodogram.psd[1:-1], expect[1:-1], atol=0.15)
    assert psd_raw_periodogram.psd[0] == pytest.approx(0.5, rel=1e-1)  # scipy handles the 0 bin and last bin differently
    assert psd_raw_periodogram.psd[-1] == pytest.approx(0.5, rel=1e-1)

    psd_raw = mass.noise_algorithms.noise_psd_mass(noise_traces, dt=frametime_s)
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
    noise_ch = mass.NoiseChannel(df=df_noise, header_df=header_df, frametime_s=frametime_s)
    pulses = noise_ch.get_records_2d()
    assert pulses.shape[0] == 10  # npulses
    assert pulses.shape[1] == 5  # length of pulses


def test_ravel_behavior():
    # noise_algorithms.noise_psd_mass relies on this behavior
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
    noise_ch = mass.NoiseChannel(df=df_noise, header_df=header_df, frametime_s=frametime_s)
    assert noise_ch.frametime_s == frametime_s

    # segfactor is the number of pulses
    f_mass, psd_mass = mass.power_spectrum.computeSpectrum(noise_traces.ravel(), segfactor=5, dt=frametime_s)
    assert len(f_mass) == 6  # half the length of the noise traces + 1
    # expect = np.ones(6)

    psd_raw_periodogram = mass.noise_algorithms.noise_psd_periodogram(noise_traces, dt=frametime_s)
    assert len(psd_raw_periodogram.frequencies) == 6  # half the length of the noise traces + 1
    assert np.allclose(f_mass, psd_raw_periodogram.frequencies)
    assert np.allclose(psd_raw_periodogram.psd[1:-1], psd_mass[1:-1], atol=0.15)

    psd_raw = mass.noise_algorithms.noise_psd_mass(noise_traces, dt=frametime_s)
    assert len(psd_raw.frequencies) == 6  # half the length of the noise traces + 1
    assert np.allclose(f_mass, psd_raw.frequencies)
    assert np.allclose(psd_raw.psd[1:-1], psd_mass[1:-1], atol=0.15)

    psd = noise_ch.spectrum(excursion_nsigma=1e100)
    assert len(psd.frequencies) == 6
    assert np.allclose(psd_raw.frequencies[:5], psd.frequencies[:5])
    assert np.allclose(psd_raw.psd, psd.psd)
