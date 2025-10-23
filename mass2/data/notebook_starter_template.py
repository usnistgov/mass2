import marimo

__generated_with = "0.14.16"
app = marimo.App(width="medium", app_title="Mass2 initial analysis")


@app.cell
def _():
    import numpy as np
    import polars as pl
    import pylab as plt
    import scipy as sp
    import marimo as mo
    import mass2


@app.cell
def _():
    def load_data():
        #
        # Notebook control parameters; change these to control what's loaded:
        #
        directory = TEMPLATE_DIRECTORY
        noise_directory = None
        max_files_limit = 2  # Set to None if you prefer no limit
        exclude_ch_nums = []
        return mass2.core.Channels.from_ljh_folder(directory, noise_directory, limit=max_files_limit, exclude_ch_nums=exclude_ch_nums)

    return load_data, mass2, mo, plt


@app.cell
def _(mo):
    mo.md(
        """
    #MASS version 2 initial analysis
    MASS is the Microcalorimeter Analysis Software Suite. Version 2 is built on modern open source data science software, including [Pola.rs](https://pola.rs) and [Marimo](https://marimo.io). MASS v2 supports some key features that v1 struggled with, including consecutive data set analysis; online (aka realtime) analysis; and easily supporting different analysis chains.
    """
    )


@app.cell
def _(load_data, mass2):
    data = load_data()

    def _do_analysis(ch: mass2.Channel) -> mass2.Channel:
        return ch.summarize_pulses().with_good_expr_pretrig_rms_and_postpeak_deriv()

    data2 = data.map(_do_analysis)
    data2 = data2.with_experiment_state_by_path()
    data2.ch0.df

    return (data2,)


@app.cell
def _(data2, plt):
    plt.clf()
    plt.hist(data2.ch0.df["pulse_rms"], 1000, histtype="step")
    plt.xlabel("Pulse RMS amplitude")
    plt.gca()


@app.cell
def _(data2, mo, plt):
    plt.plot(data2.ch0.df["pretrig_mean"], data2.ch0.df["pulse_rms"], ".")
    plt.ylabel("Pulse RMS amplitude (arbs)")
    plt.xlabel("Pretrigger mean (arbs)")
    mo.mpl.interactive(plt.gca())


if __name__ == "__main__":
    app.run()
