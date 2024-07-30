import marimo

__generated_with = "0.7.12"
app = marimo.App(width="medium", app_title="MOSS intro")


@app.cell
def __():
    import polars as pl
    import pylab as plt
    import numpy as np
    import marimo as mo
    return mo, np, pl, plt


@app.cell
def __():
    import moss
    import pulsedata
    import mass
    import pathlib
    return mass, moss, pathlib, pulsedata


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
def __(mass, moss, pulsedata):
    off_paths = moss.ljhutil.find_ljh_files(
        str(pulsedata.off["ebit_20240722_0006"]), ext=".off"
    )
    off = mass.off.OffFile(off_paths[0])
    return off, off_paths


@app.cell
def __(mass, moss, pl):
    def from_off_paths(cls, off_paths):
        channels = {}
        for path in off_paths:
            ch = from_off(moss.Channel, mass.off.OffFile(path))
            channels[ch.header.ch_num] = ch
        return moss.Channels(channels, "from_off_paths")


    def from_off(cls, off):
        df = pl.from_numpy(off._mmap)
        df = (
            df.select(
                pl.from_epoch("unixnano", time_unit="ns")
                .dt.cast_time_unit("us")
                .alias("timestamp")
            )
            .with_columns(df)
            .select(pl.exclude("unixnano"))
        )
        header = moss.ChannelHeader(
            f"{off}",
            off.header["ChannelNumberMatchingName"],
            off.framePeriodSeconds,
            off._mmap["recordPreSamples"][0],
            off._mmap["recordSamples"][0],
            pl.DataFrame(off.header),
        )
        ch = cls(df, header)
        return ch
    return from_off, from_off_paths


@app.cell
def __(from_off_paths, moss, off_paths):
    data = from_off_paths(moss.Channels, off_paths)
    data
    return data,


@app.cell
def __(data, off_paths, pathlib, pl):
    data2 = data.with_experiment_state_by_path(
        pathlib.Path(off_paths[0]).parent / "20240722_run0006_experiment_state.txt"
    ).map(lambda ch: ch
          .with_good_expr_below_nsigma_outlier_resistant([("pretriggerDelta",5),("residualStdDev", 10)], and_=pl.col("filtValue")>0)
    .driftcorrect(indicator_col="pretriggerMean", uncorrected_col="filtValue")
    )
    data2.channels[1].df.limit(1000)
    return data2,


@app.cell
def __(data2, mo, pl, plt):
    data3 = data2.map(
        lambda ch: ch.rough_cal(
            [
                "AlKAlpha",
                "MgKAlpha",
                "ClKAlpha",
                "ScKAlpha",
                "CoKAlpha",
                "MnKAlpha",
                "VKAlpha",
                "CuKAlpha",
                "KKAlpha",
            ],
            uncalibrated_col="filtValue",
            calibrated_col="energy_filtValue",
            ph_smoothing_fwhm=75,
            use_expr=pl.col("state_label") == "START",
            n_extra=5
        )
    )
    ch3 = data3.channels[1]
    ch3.step_plot(1)
    mo.mpl.interactive(plt.gcf())
    return ch3, data3


@app.cell
def __(moss):
    multifit = moss.MultiFit(default_fit_width=80, default_bin_size=0.6)
    multifit = (multifit.with_line("MgKAlpha").with_line("AlKAlpha").with_line("ClKAlpha")
    .with_line("ScKAlpha").with_line("VKAlpha").with_line("MnKAlpha")
    .with_line("CoKAlpha").with_line("CuKAlpha"))
    return multifit,


@app.cell
def __(data3, multifit):
    data4 = data3.map(lambda ch: ch.multifit_spline_cal(
        multifit, previous_cal_step_index=1, calibrated_col="energy_mf_filtValue")
    )
    return data4,


@app.cell
def __(data4, mo, pl, plt):
    ch4 = (data4.ch0.with_good_expr_below_nsigma_outlier_resistant([("pretriggerDelta",5),("residualStdDev", 10)], and_=pl.col("filtValue")>0)
         )
    df = ch4.good_df()
    states = df["state_label"].unique()
    plt.figure()
    for state in states:
        dfl = df.filter(pl.col("state_label")==state)
        plt.plot(dfl["pretriggerMean"], dfl["energy_filtValue"],".", label=f"{state=}")
    plt.grid()
    plt.legend()
    mo.mpl.interactive(plt.gcf())
    return ch4, df, dfl, state, states


@app.cell
def __(data4):
    data5 = data4.map(lambda ch: ch.driftcorrect(indicator_col="pretriggerMean", uncorrected_col="filtValue"))
    return data5,


@app.cell
def __(data4, mo, plt):
    data4.channels[1].step_plot(1)
    mo.mpl.interactive(plt.gcf())
    return


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run()
