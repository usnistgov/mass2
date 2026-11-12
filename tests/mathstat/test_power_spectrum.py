import pytest
import numpy as np
import pylab as plt
import mass2

from mass2.mathstat.power_spectrum import computeSpectrum

rng = np.random.default_rng(34234)


def test_basic():
    for n in [1000, 2000, 2500, 33333, 10000]:
        for segfactor in [1, 4, 10, 33]:
            data = np.zeros(n)
            psd_only = computeSpectrum(data, segfactor=segfactor, dt=None, window=None)
            f, psd = computeSpectrum(data, segfactor=segfactor, dt=1e-6, window=None)
            assert len(f) == len(psd) == n // (2 * segfactor) + 1
            assert np.array_equal(psd_only, psd)
            assert np.all(psd == 0)


def test_window_sequence():
    data = np.random.default_rng(0).normal(size=256)
    expected = computeSpectrum(data, segfactor=2, dt=1.0, window=np.hanning(128))[1]
    assert np.array_equal(computeSpectrum(data, segfactor=2, dt=1.0, window=list(np.hanning(128)))[1], expected)
    ps = mass2.mathstat.power_spectrum.PowerSpectrum(64)
    ps.addDataSegment(data[:128], window=list(np.hanning(128)))
    assert np.array_equal(ps.spectrum(), computeSpectrum(data[:128], dt=1.0, window=np.hanning(128))[1])


def test_values():
    f, psd = mass2.mathstat.power_spectrum.computeSpectrum(np.arange(10), segfactor=1, dt=1)
    expected = [405.0, 52.36067977, 14.47213595, 7.63932023, 5.52786405, 5.0]
    assert psd == pytest.approx(expected)
    assert f == pytest.approx(np.linspace(0, 0.5, 6))


# test functions used in the oct23 true bq analysis
# where the pulses did not live in ljh files, instead the pulses needed to be passed
# explicitly as arguments to the psd and autocorr calculation
#
# This documents how power spectrum and noise autocorrelation can be computed,
# and how to use them to create a `FilterMaker` and in turn a `Filter`.
def test_creating_filter_from_non_LJH_data():
    record_len = 100
    npre = 50
    n_noise_records = 50
    frametime_s = 1e-5
    avg_pulse_values = 1000 * np.arange(record_len, dtype=float)
    # ^ we're just testing math here, this "average pulse" is easier than making an actual pulse shape

    noise_traces = rng.standard_normal((n_noise_records, record_len))
    noise_autocorr = mass2.core.noise_algorithms.calc_discontinuous_autocorrelation(noise_traces)
    noise_psd_calculator = mass2.mathstat.power_spectrum.PowerSpectrum(record_len // 2, dt=frametime_s)
    window = np.ones(record_len)
    for i in range(n_noise_records):
        noise_psd_calculator.addDataSegment(noise_traces[i], window=window)
    # test that we can plot the noise spectrum
    noise_psd_calculator.plot()
    plt.close()
    noise_psd = noise_psd_calculator.spectrum()

    maker = mass2.core.FilterMaker(
        avg_pulse_values, npre, noise_autocorr=noise_autocorr, noise_psd=noise_psd, sample_time_sec=frametime_s
    )
    filter_obj = maker.compute_5lag()
    print("predicted resolutions")
    filter_obj.report(std_energy=1000)
    chosen_filter = filter_obj.values
    _ = chosen_filter @ avg_pulse_values[2:-2]


def test_copy_independent():
    r = np.random.default_rng(0)
    po = mass2.mathstat.power_spectrum.PowerSpectrumOverlap(8)
    po.addDataSegment(r.normal(size=8))
    po.addDataSegment(r.normal(size=8))
    before = po.spectrum().copy()
    c = po.copy()
    assert type(c) is mass2.mathstat.power_spectrum.PowerSpectrumOverlap
    c.addDataSegment(100 * r.normal(size=8))
    assert np.array_equal(po.spectrum(), before)


def test_window_callable_returning_list():
    d = np.random.default_rng(0).standard_normal(16)
    ref = mass2.mathstat.power_spectrum.PowerSpectrum(8)
    ref.addDataSegment(d, window=np.hanning(16))
    p = mass2.mathstat.power_spectrum.PowerSpectrum(8)
    p.addDataSegment(d, window=lambda n: list(np.hanning(n)))
    assert np.allclose(p.spectrum(), ref.spectrum())
