import moss
import pulsedata
import numpy as np
import pytest


def test_ljh_to_polars():
    p = pulsedata.pulse_noise_ljh_pairs["20230626"]
    ljh_noise = moss.LJHFile(p.noise_folder/"20230626_run0000_chan4102.ljh")
    df_noise, header_df_noise = ljh_noise.to_polars()
    ljh = moss.LJHFile(p.pulse_folder/"20230626_run0001_chan4102.ljh")
    df, header_df = ljh.to_polars()

def test_follow_mass_filtering_rst():
    # following https://github.com/usnistgov/mass/blob/master/doc/filtering.rst
    import numpy as np
    import mass
    import polars as pl
    np.random.seed(1)

    ### make a pulse and call mass.FilterMaker directly
    ### test that the calculated values are correct per the mass docs
    n = 504
    Maxsignal = 1000.0
    sigma_noise = 1.0
    tau = [.05, .25]
    t = np.linspace(-1, 1, n)
    npre = (t < 0).sum()
    signal = (np.exp(-t/tau[1]) - np.exp(-t/tau[0]) )
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

    ### then compare to the equivalent code in moss
    ### 1. generate noise with the same covar
    ### 2. make a channel and noise channel
    ### 3. call filter5lag
    ### 4. check outputs match and make sense

    # 250 pulses of length 504
    # noise that wil have covar of the form [1, 0, 0, 0, ...]
    noise_traces = np.random.randn(250, n)
    pulse_traces = np.tile(signal, (250,1))+noise_traces
    header_df = pl.DataFrame()
    frametime_s = 1e-5
    df_noise = pl.DataFrame({"pulse": noise_traces})
    noise_ch = moss.NoiseChannel(df_noise, header_df, frametime_s)
    header = moss.ChannelHeader("dummy for test",ch_num=0, frametime_s=frametime_s, 
                                n_presamples=n//2,                                
                                n_samples =n, df=header_df)
    df = pl.DataFrame({"pulse":pulse_traces})
    ch = moss.Channel(df, header, noise=noise_ch)
    ch = ch.filter5lag()
    step: moss.Filter5LagStep = ch.steps[-1]
    assert isinstance(step, moss.Filter5LagStep)
    filter: moss.Filter = step.filter
    assert filter.v_over_dv == pytest.approx(mass_filter.predicted_v_over_dv, rel=1e-2)
    # test that the mass normaliztion in place
    # a pulse filtered value (5lagy) should roughly equal its peak height
    assert np.mean(ch.df["5lagy"].to_numpy()) == pytest.approx(Maxsignal, rel=1e-2)
    # compare v_dv achieved (signal/fwhm) to predicted using 2.355*std=fwhm
    assert Maxsignal/(2.355*np.std(ch.df["5lagy"].to_numpy())) == pytest.approx(mass_filter.predicted_v_over_dv, rel=5e-2)
    assert filter.filter_type == "mass 5lag"
    assert filter.dt == frametime_s



# def test_filter5lag():
#     import polars as pl
#     n = 500
#     Maxsignal = 1000.0
#     sigma_noise = 1.0
#     tau = [.05, .25]
#     t = np.linspace(-1, 1, n+4)
#     npre = (t < 0).sum()
#     signal = (np.exp(-t/tau[1]) - np.exp(-t/tau[0]) )
#     signal[t <= 0] = 0
#     signal *= Maxsignal / signal.max()

#     noise_covar = np.zeros(n)
#     noise_covar[0] = sigma_noise**2
#     noise_trace = np.random.randn(n)
#     df = pl.DataFrame({"pulse": signal})
#     df_noise = pl.DataFrame({"pulse": noise_trace})

#     frametime_s = 10e-6
#     header_df = pl.DataFrame()
#     header = moss.ChannelHeader("dummy for test", 0, frametime_s, n_presamples=n//2,
#                                 n_samples =n, df=header_df)
#     noise_ch = moss.NoiseChannel(df_noise, header_df, frametime_s)
#     ch = moss.Channel(df, header, noise=noise_ch)
#     ch.filter5lag()

def test_noise_autocorr():
    import polars as pl
    import mass
    header_df = pl.DataFrame()
    frametime_s = 1e-5
    # 250 pulses of length 500
    # noise that wil have covar of the form [1, 0, 0, 0, ...]
    noise_traces = np.random.randn(250, 500)
    df_noise = pl.DataFrame({"pulse": noise_traces})
    noise_ch = moss.NoiseChannel(df_noise, header_df, frametime_s)
    assert len(noise_ch.df) == 250
    assert len(noise_ch.df["pulse"][0]) == 500
    noise_autocorr_mass = mass.power_spectrum.autocorrelation_broken_from_pulses(noise_traces.T)
    assert len(noise_autocorr_mass) == 500
    assert noise_autocorr_mass[0] == pytest.approx(1, rel=1e-1)
    assert np.mean(np.abs(noise_autocorr_mass[1:])) == pytest.approx(0, abs=1e-2)

    ac_direct = moss.noise_algorithms.autocorrelation(noise_traces, dt=frametime_s).ac
    assert len(ac_direct) == 500
    assert ac_direct[0] == pytest.approx(1, rel=1e-1)
    assert np.mean(np.abs(ac_direct[1:])) == pytest.approx(0, abs=1e-2) 

    spect = noise_ch.spectrum()
    assert len(spect.autocorr_vec) == 500
    assert spect.autocorr_vec[0] == pytest.approx(1, rel=1e-2)
    assert np.mean(np.abs(spect.autocorr_vec[1:])) == pytest.approx(0, abs=1e-2)

def test_noise_psd():
    import polars as pl
    import mass
    np.random.seed(1)
    header_df = pl.DataFrame()
    frametime_s = 0.5
    # 250 pulses of length 500
    # noise that wil have 1 arb/Hz value
    # In the case of white noise, the power spectral density (in V²/Hz) is simply the variance of the noise:
    # PSD = sigma**2/delta_f
    # sigma**2 = 1
    # delta_f == 1
    #PSD = 1/Hz
    noise_traces = np.random.randn(1000, 500)
    df_noise = pl.DataFrame({"pulse": noise_traces})
    noise_ch = moss.NoiseChannel(df=df_noise, 
                                 header_df=header_df, 
                                 frametime_s=frametime_s)
    assert noise_ch.frametime_s == frametime_s

    # segfactor is the number of pulses
    f_mass, psd_mass = mass.power_spectrum.computeSpectrum(noise_traces.ravel(), segfactor=1000, dt=frametime_s)
    assert len(f_mass) == 251 # half the length of the noise traces + 1
    expect = np.ones(251)
    assert np.allclose(psd_mass, expect, atol=0.15)

    psd_raw_periodogram = moss.noise_algorithms.noise_psd_periodogram(noise_traces, dt=frametime_s)
    assert len(psd_raw_periodogram.frequencies) == 251 # half the length of the noise traces + 1
    assert np.allclose(f_mass, psd_raw_periodogram.frequencies)
    assert np.allclose(psd_raw_periodogram.psd[1:-1], expect[1:-1], atol=0.15)
    assert psd_raw_periodogram.psd[0] == pytest.approx(0.5, rel=1e-1) # scipy handles the 0 bin and last bin differently
    assert psd_raw_periodogram.psd[-1] == pytest.approx(0.5, rel=1e-1)

    psd_raw = moss.noise_algorithms.noise_psd_mass(noise_traces, dt=frametime_s) 
    assert len(psd_raw.frequencies) == 251 # half the length of the noise traces + 1
    assert np.allclose(f_mass, psd_raw.frequencies)
    assert np.allclose(psd_raw.psd[1:-1], expect[1:-1], atol=0.15)

    psd = noise_ch.spectrum()
    assert len(psd.frequencies) == 251
    assert np.allclose(psd_raw.frequencies[:5], psd.frequencies[:5])
    assert np.allclose(psd_raw.psd, psd.psd)

def test_get_pulses_2d():
    import polars as pl
    np.random.seed(1)
    header_df = pl.DataFrame()
    frametime_s = 0.5
    # 1000 pulses of length 500
    noise_traces = np.random.randn(10, 5)
    df_noise = pl.DataFrame({"pulse": noise_traces})
    noise_ch = moss.NoiseChannel(df=df_noise, 
                                 header_df=header_df, 
                                 frametime_s=frametime_s)
    pulses = noise_ch.get_records_2d()
    assert pulses.shape[0] == 10 # npulses
    assert pulses.shape[1] == 5 # length of pulses

def test_ravel_behavior():
    # noise_algorithms.noise_psd_mass relies on this behavior
    # 10 pulses of length 5
    # first pulse = a[0,:]==[0 1 2 3 4]
    a = np.arange(50).reshape(10,5)
    assert np.allclose(a[0,:], np.arange(5))
    assert np.allclose(a.ravel(), np.arange(50))

def test_noise_psd_colored():
    import polars as pl
    import mass
    np.random.seed(1)
    header_df = pl.DataFrame()
    frametime_s = 0.5
    noise_traces = np.tile(np.arange(10), (5, 1))
    assert np.allclose(noise_traces[0,:], np.arange(10))
    assert np.allclose(noise_traces.shape, np.array([5, 10]))
    df_noise = pl.DataFrame({"pulse": noise_traces})
    noise_ch = moss.NoiseChannel(df=df_noise, 
                                 header_df=header_df, 
                                 frametime_s=frametime_s)
    assert noise_ch.frametime_s == frametime_s

    # segfactor is the number of pulses
    f_mass, psd_mass = mass.power_spectrum.computeSpectrum(noise_traces.ravel(), segfactor=5, dt=frametime_s)
    assert len(f_mass) == 6 # half the length of the noise traces + 1
    expect = np.ones(6)

    psd_raw_periodogram = moss.noise_algorithms.noise_psd_periodogram(noise_traces, dt=frametime_s)
    assert len(psd_raw_periodogram.frequencies) == 6 # half the length of the noise traces + 1
    assert np.allclose(f_mass, psd_raw_periodogram.frequencies)
    assert np.allclose(psd_raw_periodogram.psd[1:-1], psd_mass[1:-1], atol=0.15)


    psd_raw = moss.noise_algorithms.noise_psd_mass(noise_traces, dt=frametime_s) 
    assert len(psd_raw.frequencies) == 6 # half the length of the noise traces + 1
    assert np.allclose(f_mass, psd_raw.frequencies)
    assert np.allclose(psd_raw.psd[1:-1], psd_mass[1:-1], atol=0.15)

    psd = noise_ch.spectrum(excursion_nsigma=1e100)
    assert len(psd.frequencies) == 6
    assert np.allclose(psd_raw.frequencies[:5], psd.frequencies[:5])
    assert np.allclose(psd_raw.psd, psd.psd)

# def test_vdv_of_simple_filter_case_standalone_mass_psd_and_autocorr():
#     # we create a test signal consisting of zeros, with a 1 in the middle
#     # then white noise with std_dev=1
#     # we take a psd of the noise, then calculate the a fourier_filter, 
#     # and check v_dv
#     # this test case is similar making a measurement with a single sample and std_dev=1
#     # so the signal to noise (v_dv) should = 1
#     # here we get 0.3
#     import mass
#     import scipy.signal
#     # create noise with n=1000 traces of m=10 samples each, assume units of V
#     noise_traces = np.random.normal(loc=0.0, scale=1, size=(1000,10))
#     dt = 1.0
#     f_mass, psd_mass = mass.power_spectrum.computeSpectrum(noise_traces.ravel(), segfactor=1000, dt=dt)
#     # the psd frequencies should have length m/2+1=6
#     # and have a max value of sample_rate/2 = 0.5/dt = 0.5
#     f_expected = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
#     assert np.allclose(f_mass, f_expected)
#     # we should have power 1 V^2/sqrt(hz) in every bin for a two sided fft
#     # but since we are considering an fft of real values, we choose to put all the power
#     # in the positive frequency bins, and get 2 V^2/sqrt(hz) in each bin
#     psd_expected = [2, 2, 2, 2, 2, 2]
#     assert np.allclose(psd_mass, psd_expected, rtol=0.15)
#     noise_autocorr_mass = mass.power_spectrum.autocorrelation_broken_from_pulses(noise_traces.T)
#     # every sample is indepdenent, and the std_dev of each sample is 1
#     # so the autocorrelation should have 1 for the 0 lag, and 0 for all other lags
#     noise_autocorr_expected = np.array([1,0,0,0,0,0,0,0,0,0], dtype=np.float64)
#     assert np.allclose(noise_autocorr_mass, noise_autocorr_expected, atol=0.15)
#     avg_signal = np.array([0,0,0,0,0,1,0,0,0,0], dtype=np.float64)
#     # our signal is all zeros and a single 1
#     # with perfect baseline knowledge, this is the same as measuing a signal of
#     # size 1 with a single sample with noise of std_dev=1
#     # so we expect a signal to noise of 1
#     # with 9 baseline samples, we expect approximatley 1/sqrt(9) = 0.3 additional noise
#     # so the expected resolving power is ~0.7
#     n_pretrigger = 2   # n_pretrigger should not matter for calculating v_over_dv, FilterMaker requires however
#     peak_signal = np.amax(avg_signal)-avg_signal[0] # peak signal is used for normalization of the filter to follow the "mass convention" of the filt_value of a pulse equaling the max-pretrigger_mean
#     maker_with_autocorr = mass.FilterMaker(avg_signal, n_pretrigger, noise_psd=psd_mass,
#                              noise_autocorr=noise_autocorr_mass,
#                              sample_time_sec=dt, peak=peak_signal)
#     filter_with_autocorr = maker_with_autocorr.compute_fourier(fmax=None, f_3db=None)
#     # providing noise_autocorr does not change the filter computation in compute_fourier
#     # but does change the v_over_dv calculation
#     maker_without_autocorr = mass.FilterMaker(avg_signal, n_pretrigger, noise_psd=psd_mass,
#                              noise_autocorr=None,
#                              sample_time_sec=dt, peak=peak_signal)
#     filter_without_autocorr = maker_without_autocorr.compute_fourier(fmax=None, f_3db=None)

#     # the filters should be identical
#     assert np.allclose(filter_with_autocorr.values, filter_without_autocorr.values)

#     # the v_over_dv values should be very close
#     assert filter_with_autocorr.predicted_v_over_dv == pytest.approx(filter_without_autocorr.predicted_v_over_dv, rel=0.1)
#     # the absolute scale should be in range of 0.7-1, so lets use 1 with reltol 0.4
#     assert filter_with_autocorr.predicted_v_over_dv == pytest.approx(1, rel=0.4)

