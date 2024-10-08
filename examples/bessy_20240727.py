import marimo

__generated_with = "0.8.17"
app = marimo.App(width="medium", app_title="MOSS intro")


@app.cell
def __(mo):
    mo.md(
        """
        #MOSS internals introdution
        MOSS is the Microcalorimeter Online Spectral Software, a replacement for MASS. MOSS support many algorithms for pulse filtering, calibration, and corrections. MOSS is built on modern open source data science software, including pola.rs and marimo. MOSS supports some key features that MASS struggled with including:
          * consecutive data set analysis
          * online (aka realtime) analysis
          * easily supporting different analysis chains
        """
    )
    return


@app.cell
def __():
    import polars as pl
    import pylab as plt
    import numpy as np
    import marimo as mo
    import pulsedata
    return mo, np, pl, plt, pulsedata


@app.cell
def __():
    import moss
    return moss,


@app.cell
def __(mo):
    mo.md(
        """
        # Load data
        Here we load the data, then we explore the internals a bit to show how MOSS is built.
        """
    )
    return


@app.cell
def __(moss, pulsedata):
    _p = pulsedata.pulse_noise_ljh_pairs["bessy_20240727"]
    data = moss.Channels.from_ljh_folder(
        pulse_folder=_p.pulse_folder, noise_folder=_p.noise_folder
    )
    data
    return data,


@app.cell
def __(mo):
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
def __(data2):
    data2.ch0.df.columns
    return


@app.cell
def __(data3, moss):
    data3.ch0.plot_scatter("timestamp", "energy_pulse_rms", color_col="state_label")
    moss.show()
    return


@app.cell
def __(data3, moss):
    data3.ch0.plot_scatter("energy_5lagy_dc", "5lagy_dc", color_col="state_label")
    moss.show()
    return


@app.cell
def __(data, moss):
    def _do_analysis(ch: moss.Channel) -> moss.Channel:
        return ch.summarize_pulses().with_good_expr_pretrig_mean_and_postpeak_deriv()


    data2 = data.map(_do_analysis)
    data2 = data2.with_experiment_state_by_path()
    return data2,


@app.cell
def __(data2, moss, pl):
    line_names = ["OKAlpha", "FeLAlpha", "NiLAlpha", "CKAlpha", "NKAlpha", "CuLAlpha"]


    def _do_analysis(ch: moss.Channel) -> moss.Channel:
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
                          use_expr=(pl.col("state_label")=="SCAN3").and_(pl.col("energy_pulse_rms").is_between(590,610)))
            .rough_cal_combinatoric(
                line_names,
                uncalibrated_col="5lagy_dc",
                calibrated_col="energy_5lagy_dc",
                ph_smoothing_fwhm=30,
                use_expr=pl.col("state_label") == "CAL2",
            )
        )


    data3 = data2.map(_do_analysis)
    return data3, line_names


@app.cell
def __(data3, moss):
    data3.ch0.step_plot(-1)
    moss.show()
    return


@app.cell
def __(data3, moss, np):
    data3.ch0.plot_hist("energy_5lagy_dc",np.arange(0,1000,0.25))
    moss.show()
    return


@app.cell
def __(data3, dropdown_ch, moss, pl):
    _result = data3.channels[dropdown_ch.value].linefit(600, col="energy_5lagy_dc", dlo=20,dhi=20,
                                                        binsize=0.25,
                                use_expr=(pl.col("state_label")=="SCAN3").and_(pl.col("5lagx").is_between(-1,-0.4)))
    _result.plotm()
    moss.show()
    return


@app.cell
def __(data3, dropdown_ch, moss, pl, plt):
    data3.channels[dropdown_ch.value].plot_scatter("5lagx", "energy_5lagy_dc", use_expr=pl.col("state_label")=="SCAN3")
    plt.grid()
    plt.ylim(595,605)
    plt.xlim(-1.5,1)
    moss.show()
    return


@app.cell
def __(data3, dropdown_ch, moss, pl, plt):
    data3.channels[dropdown_ch.value].plot_scatter("rise_time", "energy_5lagy_dc", use_expr=pl.col("state_label")=="SCAN3")
    plt.grid()
    plt.ylim(595,605)
    moss.show()
    return


@app.cell
def __(data3, dropdown_ch, moss, pl, plt):
    data3.channels[dropdown_ch.value].plot_scatter("pretrig_mean", "energy_5lagy_dc", use_expr=pl.col("state_label")=="SCAN3")
    plt.ylim(595,605)
    plt.grid()
    moss.show()
    return


@app.cell
def __(data3, dropdown_ch, moss):
    data3.channels[dropdown_ch.value].noise.spectrum().plot()
    moss.show()
    return


@app.cell
def __(data3, dropdown_ch, moss, plt):
    plt.plot(data3.channels[dropdown_ch.value].noise.df["pulse"][:10].to_numpy().T)
    plt.plot(data3.channels[dropdown_ch.value].df["pulse"][:10].to_numpy().T)
    plt.title("first 10 noise traces and first 10 pulse traces")
    moss.show()
    return


@app.cell
def __(data3, mo):
    chs = list(data3.channels.keys())
    dropdown_ch = mo.ui.dropdown({str(k):k for k in chs}, value=str(chs[0]),label="ch")
    steps = data3.ch0.steps
    steps[0].description
    steps_d = {f"{i} {steps[i].description}":i for i in range(len(steps))}
    dropdown_step = mo.ui.dropdown(steps_d, value=list(steps_d.keys())[-1], label="step")
    return chs, dropdown_ch, dropdown_step, steps, steps_d


@app.cell
def __(data3, dropdown_ch, dropdown_step, mo, moss):
    _ch=data3.channels[dropdown_ch.value]
    _ch.step_plot(dropdown_step.value)
    mo.vstack([dropdown_ch, dropdown_step, moss.show()])
    return


@app.cell
def __(data3, dropdown_ch):
    # use this filter to calculate baseline resolution
    _ch=data3.channels[dropdown_ch.value]
    _df = _ch.noise.df
    for step in _ch.steps:
        _df = step.calc_from_df(_df)
    df_baseline = _df
    df_baseline
    return df_baseline, step


@app.cell
def __(data3, df_baseline, dropdown_ch, moss, np, pl, plt):
    def gain(e):
        _ch=data3.channels[dropdown_ch.value]
        calstep = _ch.steps[4]
        ph = calstep.energy2ph(e)
        gain = ph/e
        return gain

    _ch=data3.channels[dropdown_ch.value]
    calstep = _ch.steps[4]
    _df = df_baseline.filter(pl.col("5lagx").is_between(-3,3))
    _baseline_energies = _df["energy_5lagy_dc"].to_numpy()
    fig_=plt.hist(_baseline_energies, np.arange(-4,4,0.25))
    _fwhm_baseline = np.std(_baseline_energies)*2.35
    plt.title(f"ch={dropdown_ch.value} {_fwhm_baseline=:.2f}  \n{_fwhm_baseline*gain(0.001)/gain(700)=:.2f} eV")
    plt.xlabel("energy / eV")
    moss.show()
    return calstep, fig_, gain


if __name__ == "__main__":
    app.run()
