import marimo

__generated_with = "0.14.16"
app = marimo.App(width="medium", app_title="MASS v2 intro")


@app.cell
def _(mo):
    mo.md(
        """
    #MASS version 2: introduction to internals
    MASS is the Microcalorimeter Analysis Software Suite. Version 2 is a replacement for Version 1 of MASS (2011-2025). Version 2 supports many algorithms for pulse filtering, calibration, and corrections. It is built on modern open source data science software, including [Pola.rs](https://pola.rs) and [Marimo](https://marimo.io). MASS v2 supports some key features that v1 struggled with, including:

      * consecutive data set analysis
      * online (aka realtime) analysis
      * easily supporting different analysis chains
    """
    )
    return


@app.cell
def _():
    import polars as pl
    import pylab as plt
    import numpy as np
    import marimo as mo
    import pulsedata
    return mo, np, pl, plt, pulsedata


@app.cell
def _():
    import mass2
    return (mass2,)


@app.cell
def _(mo):
    mo.md(
        """
    # Load data
    Here we load the data, then we explore the internals a bit to show how MASS version 2 is built.
    """
    )
    return


@app.cell
def _(mass2, pulsedata):
    _p = pulsedata.pulse_noise_ljh_pairs["bessy_20240727"]
    data = mass2.Channels.from_ljh_folder(
        pulse_folder=_p.pulse_folder, noise_folder=_p.noise_folder
    )
    print(data)
    return (data,)


@app.cell
def _(mo):
    mo.md(
        """
    # basic analysis
    The variables `data` is the conventional name for a `Channels` object. It contains a list of `Channel` objects, conventinally assigned to a variable `ch` when accessed individualy. One `Channel` represents a single pixel, whiles a `Channels` is a collection of pixels, like a whole array.

    The data tends to consist of pulse shapes (arrays of length 100 to 1000 in general) and per pulse quantities, such as the pretrigger mean. These data are stored internally as pola.rs `DataFrame` objects.

    The next cell shows a basic analysis on multiple channels. The function `data.transform_channels` takes a one argument function, where the one argument is a `Channel` and the function returns a `Channel`, `data.transform_channels` returns a `Channels`. There is no mutation, and we can't re-use variable names in a reactive notebook, so we store the result in a new variable `data2`.
    """
    )
    return


@app.cell
def _(data2):
    data2.ch0.df.columns
    return


@app.cell
def _(data3, mass2):
    data3.ch0.plot_scatter("timestamp", "energy_pulse_rms", color_col="state_label")
    mass2.show()
    return


@app.cell
def _(data3, mass2):
    data3.ch0.plot_scatter("energy_5lagy_dc", "5lagy_dc", color_col="state_label")
    mass2.show()
    return


@app.cell
def _(data, mass2):
    def _do_analysis(ch: mass2.Channel) -> mass2.Channel:
        return ch.summarize_pulses().with_good_expr_pretrig_rms_and_postpeak_deriv()

    data2 = data.map(_do_analysis)
    data2 = data2.with_experiment_state_by_path()
    return (data2,)


@app.cell
def _(data2, mass2, pl):
    line_names = ["OKAlpha", "FeLAlpha", "NiLAlpha", "CKAlpha", "NKAlpha", "CuLAlpha"]

    def _do_analysis(ch: mass2.Channel) -> mass2.Channel:
        return (
            ch.rough_cal_combinatoric(
                line_names,
                uncalibrated_col="pulse_rms",
                calibrated_col="energy_pulse_rms",
                ph_smoothing_fwhm=25,
                use_expr=pl.col("state_label") == "CAL2",
            )
            .filter5lag()
            .driftcorrect(indicator_col="pretrig_mean", uncorrected_col="5lagy",
                          use_expr=(pl.col("state_label") == "SCAN3").and_(pl.col("energy_pulse_rms").is_between(590, 610)))
            .rough_cal_combinatoric(
                line_names,
                uncalibrated_col="5lagy_dc",
                calibrated_col="energy_5lagy_dc",
                ph_smoothing_fwhm=30,
                use_expr=pl.col("state_label") == "CAL2",
            )
        )

    data3 = data2.map(_do_analysis)
    return (data3,)


@app.cell
def _(data3, mass2):
    data3.ch0.step_plot(-1)
    mass2.show()
    return


@app.cell
def _(data3, mass2, np):
    data3.ch0.plot_hist("energy_5lagy_dc", np.arange(0, 1000, 0.25))
    mass2.show()
    return


@app.cell
def _(data3, dropdown_ch, mass2, pl):
    _result = data3.channels[dropdown_ch.value].linefit(600, col="energy_5lagy_dc", dlo=20, dhi=20,
                                                        binsize=0.25,
                                                        use_expr=(pl.col("state_label") == "SCAN3").and_(pl.col("5lagx").is_between(-1, -0.4)))
    _result.plotm()
    mass2.show()
    return


@app.cell
def _(data3, dropdown_ch, mass2, pl, plt):
    data3.channels[dropdown_ch.value].plot_scatter("5lagx", "energy_5lagy_dc", use_expr=pl.col("state_label") == "SCAN3")
    plt.grid()
    plt.ylim(595, 605)
    plt.xlim(-1.5, 1)
    mass2.show()
    return


@app.cell
def _(data3, dropdown_ch, mass2, pl, plt):
    data3.channels[dropdown_ch.value].plot_scatter("rise_time", "energy_5lagy_dc", use_expr=pl.col("state_label") == "SCAN3")
    plt.grid()
    plt.ylim(595, 605)
    mass2.show()
    return


@app.cell
def _(data3, dropdown_ch, mass2, pl, plt):
    data3.channels[dropdown_ch.value].plot_scatter("pretrig_mean", "energy_5lagy_dc", use_expr=pl.col("state_label") == "SCAN3")
    plt.ylim(595, 605)
    plt.grid()
    mass2.show()
    return


@app.cell
def _(data3, dropdown_ch, mass2):
    data3.channels[dropdown_ch.value].noise.spectrum().plot()
    mass2.show()
    return


@app.cell
def _(data3, dropdown_ch, mass2, plt):
    plt.plot(data3.channels[dropdown_ch.value].noise.df["pulse"][:10].to_numpy().T)
    plt.plot(data3.channels[dropdown_ch.value].df["pulse"][:10].to_numpy().T)
    plt.title("first 10 noise traces and first 10 pulse traces")
    mass2.show()
    return


@app.cell
def _(data3, mo):
    chs = list(data3.channels.keys())
    dropdown_ch = mo.ui.dropdown({str(k): k for k in chs}, value=str(chs[0]), label="ch")
    steps = data3.ch0.steps
    steps[0].description
    steps_d = {f"{i} {steps[i].description}": i for i in range(len(steps))}
    dropdown_step = mo.ui.dropdown(steps_d, value=list(steps_d.keys())[-1], label="step")
    return dropdown_ch, dropdown_step


@app.cell
def _(data3, dropdown_ch, dropdown_step, mass2, mo):
    _ch = data3.channels[dropdown_ch.value]
    _ch.step_plot(dropdown_step.value)
    mo.vstack([dropdown_ch, dropdown_step, mass2.show()])
    return


@app.cell
def _(data3, dropdown_ch):
    # use this filter to calculate baseline resolution
    _ch = data3.channels[dropdown_ch.value]
    _df = _ch.noise.df
    for step in _ch.steps:
        _df = step.calc_from_df(_df)
    df_baseline = _df
    df_baseline
    return (df_baseline,)


@app.cell
def _(data3, df_baseline, dropdown_ch, mass2, np, pl, plt):
    def gain(e):
        _ch = data3.channels[dropdown_ch.value]
        calstep = _ch.steps[4]
        ph = calstep.energy2ph(e)
        gain = ph/e
        return gain

    _ch = data3.channels[dropdown_ch.value]
    calstep = _ch.steps[4]
    _df = df_baseline.filter(pl.col("5lagx").is_between(-3, 3))
    _baseline_energies = _df["energy_5lagy_dc"].to_numpy()
    fig_ = plt.hist(_baseline_energies, np.arange(-4, 4, 0.25))
    _fwhm_baseline = np.std(_baseline_energies)*2.35
    plt.title(f"ch={dropdown_ch.value} {_fwhm_baseline=:.2f}  \n{_fwhm_baseline*gain(0.001)/gain(700)=:.2f} eV")
    plt.xlabel("energy / eV")
    mass2.show()
    return


if __name__ == "__main__":
    app.run()
