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
    return mo, np, pl, plt


@app.cell
def _():
    import mass2
    return (mass2,)


@app.cell
def _(mo):
    mo.md(
        """
    # Load data: NOT CURRENTLY INSTALLED
    Here we load the data, then we explore the internals a bit to show how MASS v2 is built.
    """
    )
    return


@app.cell
def _(mass2):
    pulse_folder = "/data/20241211/0003"
    noise_folder = "/data/20241211/0001"
    data = mass2.Channels.from_ljh_folder(
        pulse_folder=pulse_folder, noise_folder=noise_folder,
        limit=100
    )
    data = data.map(lambda channel: channel.summarize_pulses())
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
def _(data, pl):
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
def _():
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    # inspecting the data

    Internally, the data is stored in polars `DataFrame`s. Lets take a look. To access the dataframe for one channel we do `data2.channels[4102].df`. In `marimo` we can get a nice UI element to browse through the data by returning the `DataFrame` as the last element in a cell. marimo's nicest display doesn't work with array columns like our pulse column, so lets leave that out for now.
    """
    )
    return


@app.cell
def _(data2, pl):
    data2.ch0.with_good_expr(pl.col("promptness") < 0.98).df.select(pl.exclude("pulse"))
    return


@app.cell
def _(data2, mass2, plt):
    _pulses = data2.ch0.df.limit(20)["pulse"].to_numpy()
    plt.figure()
    plt.plot(_pulses.T)
    mass2.show()
    return


@app.cell
def _(data2, mo):
    mo.md(
        f"""
    To enable online analysis, we have to keep track of all the steps of our calibration, so each channel has a history of its steps that we can replay. Here we interpolate it into the markdown, each entry is a step name followed by the time it took to perform the step.

    {data2.ch0.step_summary()=}
    """
    )
    return


@app.cell
def _(result):
    result.fit_report()
    return


@app.cell
def _(data2, mo):
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
    return dropdown_ch, dropdown_col, dropdown_step


@app.cell
def _(dropdown_ch, dropdown_col, dropdown_step, mo):
    mo.vstack([dropdown_ch, dropdown_step, dropdown_col])
    return


@app.cell
def _(data2, dropdown_ch, dropdown_step, mass2):
    _ch = data2.channels[dropdown_ch.value]
    _ch.step_plot(dropdown_step.value)
    mass2.show()
    return


@app.cell
def _(mo):
    mo.md(r"""# plot a noise spectrum""")
    return


@app.cell
def _(data2, dropdown_ch, dropdown_col, mo, plt):
    _ch_num, _col = int(dropdown_ch.value), dropdown_col.value
    _ch = data2.channels[int(_ch_num)]
    _ch.noise.spectrum().plot()
    mo.mpl.interactive(plt.gcf())
    return


@app.cell
def _(data2, dropdown_ch, dropdown_col):
    _ch_num, _col = int(dropdown_ch.value), dropdown_col.value
    _ch = data2.channels[int(_ch_num)]
    _steps = _ch.steps
    _steps
    return


@app.cell
def _(data2, dropdown_ch, dropdown_col, mo, plt):
    _ch_num, _col = int(dropdown_ch.value), dropdown_col.value
    _ch = data2.channels[int(_ch_num)]
    result = _ch.linefit("FeLAlpha", col=_col)
    result.plotm()
    plt.title(f"reative plot of {_ch_num=} and {_col=} for you")
    mo.mpl.interactive(plt.gcf())
    return (result,)


@app.cell
def _(data2, mass2):
    data2.ch0.plot_scatter("pulse_average", "energy_5lagy")
    mass2.show()
    return


@app.cell
def _(data, dropdown_ch, dropdown_col, mass2):
    _ch_num, _col = int(dropdown_ch.value), dropdown_col.value
    _ch = data.channels[int(_ch_num)]
    print(f"{len(_ch.df)=}")
    _ch.plot_scatter("timestamp", "pretrig_mean", use_good_expr=False)
    mass2.show()
    return


@app.cell
def _(data, dropdown_ch, dropdown_col, mass2):
    _ch_num, _col = int(dropdown_ch.value), dropdown_col.value
    _ch = data.channels[int(_ch_num)]
    _ch.plot_scatter("timestamp", "pulse_rms")
    mass2.show()
    return


@app.cell
def _(data2, dropdown_ch, dropdown_col, mass2, np):
    _ch_num, _col = int(dropdown_ch.value), dropdown_col.value
    _ch = data2.channels[int(_ch_num)]
    _ch.plot_hist("energy_5lagy_dc", np.arange(0, 3000, 5))
    mass2.show()
    return


if __name__ == "__main__":
    app.run()
