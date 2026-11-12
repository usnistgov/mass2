import numpy as np
import pytest
import mass2


def test_phase_correct(plot=False):
    # the final fit resolutions are quite sensitive to this, easily varying from 3 to 5 eV
    rng = np.random.default_rng(5632)
    energies = np.arange(4000)
    ph_peaks = []
    line_names = ["MnKAlpha", "FeKAlpha", "CuKAlpha", "CrKAlpha"]
    for i, name in enumerate(line_names):
        spect = mass2.spectra[name]
        energies[i * 1000 : (i + 1) * 1000] = spect.rvs(size=1000, rng=rng, instrument_gaussian_fwhm=3)
        ph_peaks.append(spect.nominal_peak_energy)
    phase = np.linspace(-0.6, 0.6, len(energies))
    rng.shuffle(energies)
    rng.shuffle(phase)
    ph = energies + phase * 10  # this pushes the resolution up to roughly 10 eV

    phaseCorrector = mass2.core.phase_correct.phase_correct(phase, ph, ph_peaks=ph_peaks)
    corrected = phaseCorrector(phase, ph)

    resolutions = []
    for name in line_names:
        line = mass2.spectra[name]
        model = line.model()
        bin_edges = np.arange(-100, 100) + line.peak_energy
        bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
        counts, _ = np.histogram(corrected, bin_edges)
        params = model.guess(counts, bin_centers=bin_centers, dph_de=1)
        params["dph_de"].set(1.0, vary=False)
        result = model.fit(counts, params, bin_centers=bin_centers)
        resolutions.append(result.best_values["fwhm"])
        if plot:
            result.plotm()
    print(resolutions)
    assert resolutions[0] <= 4.5
    assert resolutions[1] <= 4.4
    assert resolutions[2] <= 4.0
    assert resolutions[3] <= 4.4


def test_phase_correct_single_narrow_line():
    rng = np.random.default_rng(3)
    phase = rng.uniform(-0.5, 0.5, 5000)
    ph = 1000 + 2 * phase + rng.normal(0, 0.5, len(phase))
    corrected = mass2.core.phase_correct.phase_correct(phase, ph, ph_peaks=[1000])(phase, ph)
    assert np.all(np.isfinite(corrected))
    assert corrected.std() < 0.55


def test_phase_correct_no_peaks_found():
    with pytest.raises(ValueError, match="no peaks found"):
        mass2.core.phase_correct.phase_correct(np.zeros(1000), np.full(1000, 5.0))


def test_phase_correct_integer_pulse_heights():
    rng = np.random.default_rng(1)
    phase = rng.uniform(-0.5, 0.5, 5000)
    ph = np.round(rng.choice([3000.0, 5000.0], len(phase)) + 20 * phase).astype(np.int64)
    expected = mass2.core.phase_correct.phase_correct(phase, ph.astype(float), ph_peaks=[3000, 5000])(phase, ph.astype(float))
    corrector = mass2.core.phase_correct.phase_correct(phase, ph, ph_peaks=[3000, 5000])
    assert np.array_equal(corrector(phase, ph), expected)
