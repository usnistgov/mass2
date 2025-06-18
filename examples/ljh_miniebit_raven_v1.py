import marimo

__generated_with = "0.9.28"
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
    return (moss,)


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
def __(moss):
    pulse_folder = "/data/20241211/0003"
    noise_folder = "/data/20241211/0001"
    data = moss.Channels.from_ljh_folder(
        pulse_folder=pulse_folder, noise_folder=noise_folder,
        limit=100
    )
    data = data.map(lambda channel: channel.summarize_pulses())
    return data, noise_folder, pulse_folder


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
def __(data, pl):
    data2 = data.map(
        lambda channel: channel.with_good_expr_pretrig_mean_and_postpeak_deriv()
        .with_good_expr(pl.col("pulse_average") > 0)
        .with_good_expr(pl.col("promptness") < 0.98)
        .rough_cal_combinatoric(
            ["FeLAlpha"],
            uncalibrated_col="peak_value",
            calibrated_col="energy_peak_value",
            ph_smoothing_fwhm=50,
        )
        .filter5lag(f_3db=10e3)
        .rough_cal_combinatoric(
            ["FeLAlpha"],
            uncalibrated_col="5lagy",
            calibrated_col="energy_5lagy",
            ph_smoothing_fwhm=50,
        )
        .driftcorrect()
        .rough_cal_combinatoric(
            ["FeLAlpha", "OKAlpha"],
            uncalibrated_col="5lagy_dc",
            calibrated_col="energy_5lagy_dc",
            ph_smoothing_fwhm=50,
        )
    )
    return (data2,)


@app.cell
def __():
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        """
        # inspecting the data

        Internally, the data is stored in polars `DataFrame`s. Lets take a look. To access the dataframe for one channel we do `data2.channels[4102].df`. In `marimo` we can get a nice UI element to browse through the data by returning the `DataFrame` as the last element in a cell. marimo's nicest display doesn't work with array columns like our pulse column, so lets leave that out for now.
        """
    )
    return


@app.cell
def __(data2, pl):
    data2.ch0.with_good_expr(pl.col("promptness") < 0.98).df.select(pl.exclude("pulse"))
    return


@app.cell
def __(data2, moss, plt):
    _pulses = data2.ch0.df.limit(20)["pulse"].to_numpy()
    plt.figure()
    plt.plot(_pulses.T)
    moss.show()
    return


@app.cell
def __(data2, mo):
    mo.md(
        f"""
        To enable online analysis, we have to keep track of all the steps of our calibration, so each channel has a history of its steps that we can replay. Here we interpolate it into the markdown, each entry is a step name followed by the time it took to perform the step. 

        {data2.ch0.step_summary()=}
        """
    )
    return


@app.cell
def __(result):
    result.fit_report()
    return


@app.cell
def __(data2, mo):
    chs = list(data2.channels.keys())
    dropdown_ch = mo.ui.dropdown({str(k): k for k in chs}, value=str(chs[0]), label="ch")
    _energy_cols = [col for col in data2.dfg().columns if col.startswith("energy")]
    dropdown_col = mo.ui.dropdown(
        options=_energy_cols, value=_energy_cols[0], label="energy col"
    )
    steps = data2.ch0.steps
    steps[0].description
    steps_d = {f"{i} {steps[i].description}": i for i in range(len(steps))}
    dropdown_step = mo.ui.dropdown(steps_d, value=list(steps_d.keys())[-1], label="step")
    return chs, dropdown_ch, dropdown_col, dropdown_step, steps, steps_d


@app.cell
def __(dropdown_ch, dropdown_col, dropdown_step, mo):
    mo.vstack([dropdown_ch, dropdown_step, dropdown_col])
    return


@app.cell
def __(data2, dropdown_ch, dropdown_step, moss):
    _ch = data2.channels[dropdown_ch.value]
    _ch.step_plot(dropdown_step.value)
    moss.show()
    return


@app.cell
def __(mo):
    mo.md(r"""# plot a noise spectrum""")
    return


@app.cell
def __(data2, dropdown_ch, dropdown_col, mo, plt):
    _ch_num, _col = int(dropdown_ch.value), dropdown_col.value
    _ch = data2.channels[int(_ch_num)]
    _ch.noise.spectrum().plot()
    mo.mpl.interactive(plt.gcf())
    return


@app.cell
def __(data2, dropdown_ch, dropdown_col):
    _ch_num, _col = int(dropdown_ch.value), dropdown_col.value
    _ch = data2.channels[int(_ch_num)]
    _steps = _ch.steps
    _steps
    return


@app.cell
def __(data2, dropdown_ch, dropdown_col, mo, plt):
    _ch_num, _col = int(dropdown_ch.value), dropdown_col.value
    _ch = data2.channels[int(_ch_num)]
    result = _ch.linefit("FeLAlpha", col=_col)
    result.plotm()
    plt.title(f"reative plot of {_ch_num=} and {_col=} for you")
    mo.mpl.interactive(plt.gcf())
    return (result,)


@app.cell
def __(data2, moss):
    data2.ch0.plot_scatter("pulse_average", "energy_5lagy")
    moss.show()
    return


@app.cell
def __(data, dropdown_ch, dropdown_col, moss):
    _ch_num, _col = int(dropdown_ch.value), dropdown_col.value
    _ch = data.channels[int(_ch_num)]
    print(f"{len(_ch.df)=}")
    _ch.plot_scatter("timestamp", "pretrig_mean", use_good_expr=False)
    moss.show()
    return


@app.cell
def __(data, dropdown_ch, dropdown_col, moss):
    _ch_num, _col = int(dropdown_ch.value), dropdown_col.value
    _ch = data.channels[int(_ch_num)]
    _ch.plot_scatter("timestamp", "pulse_rms")
    moss.show()
    return


@app.cell
def __(data2, dropdown_ch, dropdown_col, moss, np):
    _ch_num, _col = int(dropdown_ch.value), dropdown_col.value
    _ch = data2.channels[int(_ch_num)]
    _ch.plot_hist("energy_5lagy_dc", np.arange(0, 3000, 5))
    moss.show()
    return


if __name__ == "__main__":
    app.run()
