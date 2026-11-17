import numpy as np
import pylab as plt
import mass2
import polars as pl
import os
import tempfile
import pytest


def generate_and_fit_fake_data(
    monkeypatch,
    n_mn_counts=2000,
    mn_fwhm_ev=4,
    n_gaussian_counts=1000,
    gaussian_centers_ev=[2000, 4000],
    gaussian_fwhm_ev=4,
    seed=1,
):
    """
    Generate fake spectral data with MnKAlpha line and Gaussian peak, then fit with MultiFit.
    """

    rng = np.random.default_rng(seed)

    # Generate MnKAlpha line data
    line = mass2.calibration.fluorescence_lines.MnKAlpha
    mn_values = line.rvs(size=n_mn_counts, instrument_gaussian_fwhm=mn_fwhm_ev, rng=rng)

    gaussian_values = []
    for gaussian_center_ev in gaussian_centers_ev:
        # Generate Gaussian peak data
        gaussian_values.append(
            rng.normal(
                loc=gaussian_center_ev,
                scale=gaussian_fwhm_ev / 2.3548,
                size=n_gaussian_counts,
            )
        )

    # Combine all photon energies
    all_values = np.concatenate([mn_values, gaussian_values[0], gaussian_values[1]])
    df = pl.DataFrame({"energy": all_values})
    ch = mass2.Channel(
        df,
        header=mass2.ChannelHeader(
            description="Fake spectral data for testing MultiFit",
            data_source=None,
            ch_num=1,
            frametime_s=0.1,
            n_presamples=100,
            n_samples=100,
            df=None,
        ),
        npulses=len(all_values),
    )

    # Set up MultiFit
    multifit = (
        mass2
        .MultiFit(default_fit_width=80, default_bin_size=1)
        .with_line("MnKAlpha")
        .with_line(gaussian_centers_ev[0])
        .with_line(gaussian_centers_ev[1])
    )
    monkeypatch.setattr(mass2.calibration.line_models, "VALIDATE_BIN_SIZE", False)
    ch = ch.rough_cal_combinatoric(
        ["MnKAlpha", gaussian_centers_ev[0], gaussian_centers_ev[1]],
        ph_smoothing_fwhm=4,
        n_extra=1,
        uncalibrated_col="energy",
        calibrated_col="energy2",
    )
    ch = ch.multifit_mass_cal(multifit, previous_cal_step_index=-1, calibrated_col="energy3")

    energy = ch.df["energy"].to_numpy()
    energy3 = ch.df["energy3"].to_numpy()
    for center in [line.peak_energy, *gaussian_centers_ev]:
        near = np.abs(energy - center) < 20
        assert np.abs(energy3[near] - energy[near]).max() < 0.5
    return ch


def test_multifit_and_saving_and_loading_steps_in_channel(monkeypatch):
    ch = generate_and_fit_fake_data(monkeypatch)
    ch.step_plot(-1)

    with tempfile.TemporaryDirectory() as tmpdir:  # cleaned up on exit
        filename = os.path.join(tmpdir, "steps.pkl")  # just a string path
        ch.save_recipes(filename)
        steps_loaded = mass2.misc.unpickle_object(filename)
    steps_loaded[ch.header.ch_num][-1].dbg_plot(ch.df)

    plt.close()
    plt.close()


def test_multifit_and_saving_and_loading_steps_in_channels(monkeypatch):
    ch = generate_and_fit_fake_data(monkeypatch)
    data = mass2.Channels({ch.header.ch_num: ch}, "for test_multifit_in_channels")

    with tempfile.TemporaryDirectory() as tmpdir:  # cleaned up on exit
        filename = os.path.join(tmpdir, "steps.pkl")  # just a string path
        data.save_recipes(filename)
        _ = data.load_recipes(filename)  # it would be better to start with a data that doesn't have the output of the steps...
        # and maybe throw an error for writing over existing columns?
        # fow now, I'm happy this is tested at all
        steps = mass2.misc.unpickle_object(filename)
    _ = data.with_steps_dict(steps)  # pretty much the blow by blow version of load_recipes


if __name__ == "__main__":
    # Run the function to generate and fit fake data
    mp = pytest.MonkeyPatch()
    test_multifit_and_saving_and_loading_steps_in_channel(mp)
    test_multifit_and_saving_and_loading_steps_in_channels(mp)


@pytest.mark.parametrize("a", [-1e-17, 0.0])
def test_quadratic_gain_energy2ph_nearly_linear(a):
    pfit_gain = np.polynomial.Polynomial([0.5, -2e-6, a])
    step = mass2.core.MultiFitQuadraticGainStep(["x"], ["y"], pl.lit(True), pl.lit(True), pfit_gain, None, 0.0)
    rough = mass2.core.rough_cal.BestAssignmentPfitGainResult(0.0, None, None, None, pfit_gain, None, None, None)
    energy = np.array([2000.0, 4000.0, 8000.0])
    for ph in (step.energy2ph(energy), rough.energy2ph(energy)):
        assert step.ph2energy(ph) == pytest.approx(energy, rel=1e-13)


def test_to_pfit_gain_two_lines_is_linear(monkeypatch):
    monkeypatch.setattr(mass2.calibration.line_models, "VALIDATE_BIN_SIZE", False)
    rng = np.random.default_rng(1)
    e = np.hstack([mass2.spectra[name].rvs(5000, 4, rng=rng) for name in ("MnKAlpha", "CuKAlpha")])
    mf = mass2.MultiFit(default_fit_width=80, default_bin_size=0.5).with_line("MnKAlpha").with_line("CuKAlpha")
    mf = mf.fit_series_without_use_expr(pl.Series("energy_rough", e))
    pfit_gain, _ = mf.to_pfit_gain(lambda x: 2.0 * np.asarray(x))
    assert pfit_gain.convert().degree() <= 1
    assert pfit_gain(np.array([14000.0, 24000.0])) == pytest.approx(2.0, rel=1e-3)


@pytest.mark.parametrize("method", ["multifit_mass_cal", "multifit_quadratic_gain_cal"])
def test_multifit_cal_applies_use_expr(monkeypatch, method):
    monkeypatch.setattr(mass2.calibration.line_models, "VALIDATE_BIN_SIZE", False)
    rng = np.random.default_rng(3)
    lines = ["TiKAlpha", "CrKAlpha", "MnKAlpha", "FeKAlpha", "CuKAlpha"]
    gain = np.polynomial.Polynomial([0.5, -1e-6, -1e-12])
    phgrid = np.linspace(1, 40000, 200001)
    egrid = phgrid / gain(phgrid)
    ph = [np.interp(np.hstack([mass2.spectra[name].rvs(6000, 4, rng=rng) for name in lines]), egrid, phgrid) for _ in "AB"]
    ph[1] *= 1.001
    header = mass2.ChannelHeader(
        description="fake", data_source=None, ch_num=1, frametime_s=1e-5, n_presamples=100, n_samples=200, df=None
    )
    df = pl.DataFrame({"filtValue": np.hstack(ph), "state": ["A"] * len(ph[0]) + ["B"] * len(ph[1])})
    ch = mass2.Channel(df, header=header, npulses=len(df))
    use = pl.col("state") == "A"
    multifit = mass2.MultiFit(default_fit_width=80, default_bin_size=0.5)
    for name in lines:
        multifit = multifit.with_line(name)
    ch = ch.rough_cal(lines, uncalibrated_col="filtValue", calibrated_col="e1", fwhm_pulse_height_units=5, use_expr=use)
    ch = getattr(ch, method)(multifit, previous_cal_step_index=-1, calibrated_col="e2", use_expr=use)
    e0 = mass2.spectra["MnKAlpha"].peak_energy
    assert ch.steps[-1].ph2energy(np.interp(e0, egrid, phgrid)) == pytest.approx(e0, abs=0.5)
