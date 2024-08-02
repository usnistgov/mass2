import marimo

__generated_with = "0.7.14"
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
def __(from_off_paths, moss, off_paths, pathlib):
    data = from_off_paths(moss.Channels, off_paths).with_experiment_state_by_path(
        pathlib.Path(off_paths[0]).parent / "20240722_run0006_experiment_state.txt"
    )
    data
    return data,


@app.cell
def __(data, pl):
    data2 = data.map(
        lambda ch: ch.with_good_expr_below_nsigma_outlier_resistant(
            [("pretriggerDelta", 5), ("residualStdDev", 10)], and_=pl.col("filtValue") > 0)
        .driftcorrect(indicator_col="pretriggerMean", uncorrected_col="filtValue").
        rough_cal(
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
            uncalibrated_col="filtValue_dc",
            use_expr=pl.col("state_label") == "START",
        )
    )
    data2.channels[1].df.limit(1000)
    return data2,


@app.cell
def __(data2, mo, plt):
    data2.ch0.step_plot(-1)
    mo.mpl.interactive(plt.gcf())
    return


@app.cell
def __(data2, mo, pl, plt):
    result = data2.ch0.linefit("AlKAlpha", "energy_filtValue_dc", use_expr=pl.col("state_label") == "START")
    result.plotm()
    mo.mpl.interactive(plt.gcf())
    return result,


@app.cell
def __():
    def plot_a_vs_b(ch, a_col, b_col, color_by_col):
        pass
    return plot_a_vs_b,


@app.cell
def __(moss):
    multifit = moss.MultiFit(default_fit_width=80, default_bin_size=0.6)
    multifit = (
        multifit.with_line("MgKAlpha")
        .with_line("AlKAlpha")
        .with_line("ClKAlpha")
        .with_line("ScKAlpha")
        .with_line("VKAlpha")
        .with_line("MnKAlpha")
        .with_line("CoKAlpha")
        .with_line("CuKAlpha")
    )
    return multifit,


@app.cell
def __(np, pl):
    df1 = pl.DataFrame({"a":np.arange(3)})
    df2 = pl.DataFrame({"b":np.arange(3)})
    want=df1.join(df2, how="cross").filter(pl.col("a")<pl.col("b"))
    print(want)
    return df1, df2, want


if __name__ == "__main__":
    app.run()
