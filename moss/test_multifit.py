import moss
import numpy as np
import mass
import pytest
from pytest import approx
import polars as pl

def generate_and_fit_fake_data(
    n_mn_counts=2000,
    mn_fwhm_ev=4,
    n_gaussian_counts=1000,
    gaussian_centers_ev=[2000,4000],
    gaussian_fwhm_ev=4,
    seed=1
):
    """
    Generate fake spectral data with MnKAlpha line and Gaussian peak, then fit with MultiFit.
    """
    
    rng = np.random.default_rng(seed)
    
    # Generate MnKAlpha line data
    line = mass.calibration.fluorescence_lines.MnKAlpha
    mn_values = line.rvs(size=n_mn_counts, instrument_gaussian_fwhm=mn_fwhm_ev, rng=rng)
    
    gaussian_values = []
    for gaussian_center_ev in gaussian_centers_ev:
        # Generate Gaussian peak data
        gaussian_values.append(rng.normal(
            loc=gaussian_center_ev, 
            scale=gaussian_fwhm_ev / 2.3548, 
            size=n_gaussian_counts
        ))

    # Combine all photon energies
    all_values = np.concatenate([mn_values, gaussian_values[0], gaussian_values[1]])
    df = pl.DataFrame({"energy": all_values})   
    ch = moss.Channel(df, header=moss.ChannelHeader(description="Fake spectral data for testing MultiFit", 
                                                    ch_num=1, frametime_s=0.1,
                                                    n_presamples=100, n_samples=100,
                                                    df= None))

    
    # Set up MultiFit
    multifit = (moss.MultiFit(
        default_fit_width=80, 
        default_bin_size=1
    ).with_line("MnKAlpha")
    .with_line(gaussian_centers_ev[0])
    .with_line(gaussian_centers_ev[1])
    )
    mass.line_models.VALIDATE_BIN_SIZE=False
    ch = ch.rough_cal_combinatoric(["MnKAlpha", gaussian_centers_ev[0], gaussian_centers_ev[1]],
                                   ph_smoothing_fwhm=4, n_extra=1,
                                   uncalibrated_col="energy",
                                   calibrated_col="energy2")
    ch = ch.multifit_mass_cal(multifit, 
                            previous_cal_step_index=-1,
                            calibrated_col="energy3")
    

    return ch

def test_multifit_and_saving_and_loading_steps_in_channel():
    ch = generate_and_fit_fake_data()
    ch.step_plot(-1)
    import tempfile, os
    with tempfile.TemporaryDirectory() as tmpdir:        # cleaned up on exit
        filename = os.path.join(tmpdir, "steps.pkl")    # just a string path
        ch.save_steps(filename)
        steps_loaded = moss.misc.unpickle_object(filename)
    steps_loaded[ch.header.ch_num][-1].dbg_plot(ch.df)
    import pylab as plt
    plt.close()
    plt.close()

def test_multifit_and_saving_and_loading_steps_in_channels():
    ch = generate_and_fit_fake_data()
    data = moss.Channels({ch.header.ch_num: ch}, "for test_multifit_in_channels")
    import tempfile, os
    with tempfile.TemporaryDirectory() as tmpdir:        # cleaned up on exit
        filename = os.path.join(tmpdir, "steps.pkl")    # just a string path
        data.save_steps(filename)
        data2 = data.load_steps(filename) # it would be better to start with a data that doesn't have the output of the steps... 
        # and maybe throw an error for writing over existing columns?
        # fow now, I'm happy this is tested at all
        steps = moss.misc.unpickle_object(filename)
    data3 = data.with_steps_dict(steps) # pretty much the blow by blow version of load_steps

 

if __name__ == "__main__":
    # Run the function to generate and fit fake data
    test_multifit_in_channel()
    test_multifit_in_channels()