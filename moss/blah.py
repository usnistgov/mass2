import polars as pl
import pylab as plt
import numpy as np
import marimo as mo
import moss

import mass 
import os
massroot = os.path.split(os.path.split(mass.__file__)[0])[0]

noise_folder = os.path.join(massroot,"tests","ljh_files","20230626","0000")
pulse_folder = os.path.join(massroot,"tests","ljh_files","20230626","0001")
data = moss.Channels.from_ljh_folder(
    pulse_folder=pulse_folder, noise_folder=noise_folder
)
data2 = data.transform_channels(
    lambda channel: channel.summarize_pulses()
    .with_good_expr_pretrig_mean_and_postpeak_deriv()
    .rough_cal(
        ["MnKAlpha", "MnKBeta", "CuKAlpha", "CuKBeta", "PdLAlpha", "PdLBeta"],
        uncalibrated_col="peak_value",
        calibrated_col="energy_peak_value",
        ph_smoothing_fwhm=50,
    )
    .filter5lag(f_3db=10e3)
    .rough_cal(
        ["MnKAlpha", "MnKBeta", "CuKAlpha", "CuKBeta", "PdLAlpha", "PdLBeta"],
        uncalibrated_col="5lagy",
        calibrated_col="energy_5lagy",
        ph_smoothing_fwhm=50,
    )
    .driftcorrect()
    .rough_cal(
        ["MnKAlpha", "MnKBeta", "CuKAlpha", "CuKBeta", "PdLAlpha", "PdLBeta"],
        uncalibrated_col="5lagy_dc",
        calibrated_col="energy_5lagy_dc",
        ph_smoothing_fwhm=50,
    )
)