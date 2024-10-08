import marimo

__generated_with = "0.9.4"
app = marimo.App(width="medium", app_title="MOSS follow")


@app.cell
def __():
    import polars as pl
    import pylab as plt
    import numpy as np
    import marimo as mo
    import moss
    return mo, moss, np, pl, plt


@app.cell
def __(mo):
    mo.md(
        r"""
        # loading data
        we load the same ljh files, the steps_dict and the resulting df to compare to

        # todo
        * make a demo that uses a timer to read progressvley larger slices of the data, and outputs all the new data to some format
        """
    )
    return


@app.cell
def __(moss, pl):
    import pulsedata

    pulse_folder = pulsedata.pulse_noise_ljh_pairs["20230626"].pulse_folder
    data = moss.Channels.from_ljh_folder(pulse_folder=pulse_folder)
    steps_dict = moss.misc.unpickle_object("example_steps_dict.pkl")
    truth_dfg = pl.read_parquet("example_result.parquet")
    return data, pulse_folder, pulsedata, steps_dict, truth_dfg


@app.cell
def __(data, steps_dict):
    data2 = data.with_steps_dict(steps_dict)
    return (data2,)


@app.cell
def __(data2):
    dfg = data2.dfg().select("timestamp", "energy2_5lagy_dc", "ch_num")
    return (dfg,)


@app.cell
def __(dfg, truth_dfg):
    dfg == truth_dfg.select(dfg.columns)
    return


@app.cell
def __(data2, mo, plt):
    result = data2.linefit("MnKAlpha", col="energy2_5lagy_dc")
    result.plotm()
    assert result.params["fwhm"].value < 3.46
    mo.mpl.interactive(plt.gcf())
    return (result,)


@app.cell
def __(mo):
    mo.md(
        r"""
        # unified analysis experience
        We can create a `moss.Channels` object from any `DataFrame` with a `ch_num` column (just rename it if you named it wrong!). then all our analysis and plotting methods work. We can use this to save intermediate results, or just to use our fitting tools on already analyzed data.
        """
    )
    return


@app.cell
def __(data2, moss, truth_dfg):
    data_from_truth_dfg = moss.Channels.from_df(
        truth_dfg,
        frametime_s=data2.channels[4102].header.frametime_s,
        n_presamples=data2.channels[4102].header.n_presamples,
        n_samples=data2.channels[4102].header.n_samples,
        description="from Channels.channels_from_df with truth_dfg",
    )
    return (data_from_truth_dfg,)


@app.cell
def __(data_from_truth_dfg, mo, plt):
    result_from_truth_dfg = data_from_truth_dfg.linefit("MnKAlpha", col="energy2_5lagy_dc")
    result_from_truth_dfg.plotm()
    assert result_from_truth_dfg.params["fwhm"].value < 3.46
    mo.mpl.interactive(plt.gcf())
    return (result_from_truth_dfg,)


@app.cell
def __(data_from_truth_dfg):
    data_from_truth_dfg.channels[4102]
    return


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run()
