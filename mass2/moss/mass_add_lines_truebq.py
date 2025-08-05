# -*- coding: utf-8 -*-
import mass2 as mass
import numpy as np

if "Am241Q" not in mass.spectra.keys():
    mass.addline("Am241",
                 linetype="Q",
                 material="DES",
                 reference_short="Rough Estimate",
                 reference_plot_instrument_gaussian_fwhm=1000.0,
                 nominal_peak_energy=5637.82e3,
                 energies=np.array([5637.82e3, 5637.82e3-59.5409e3]),
                 lorentzian_fwhm=np.array(np.array([50, 50])),
                 reference_amplitude=np.array(np.array([10, 3])),
                 reference_amplitude_type=mass.LORENTZIAN_PEAK_HEIGHT,
                 ka12_energy_diff=60e3,
                 position_uncertainty=1.5
                 )


if "Am243Q" not in mass.spectra.keys():
    mass.addline("Am243",
                 linetype="Q",
                 material="DES",
                 reference_short="Rough Estimate",
                 reference_plot_instrument_gaussian_fwhm=1000.0,
                 nominal_peak_energy=5.3641e6,
                 energies=np.array([5.3641e6, 5.4388e6, 5.395e6]),
                 lorentzian_fwhm=np.array(np.array([50, 50, 50])),
                 reference_amplitude=np.array(np.array([10000, 5400, 720])),
                 reference_amplitude_type=mass.LORENTZIAN_PEAK_HEIGHT,
                 ka12_energy_diff=60e3,
                 position_uncertainty=1.5
                 )
