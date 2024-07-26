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
    off_paths = moss.ljhutil.find_ljh_files(str(pulsedata.off["ebit_20240722_0006"]), ext=".off")
    off = mass.off.OffFile(off_paths[0])
    return off, off_paths


@app.cell
def __(mass, moss, pl):
    def from_off_paths(cls, off_paths):
        channels = {}
        for path in off_paths:
            ch = from_off(moss.Channel, mass.off.OffFile(path))
            channels[ch.header.ch_num] =ch
        return moss.Channels(channels, "from_off_paths")

    def from_off(cls, off):
        df = pl.from_numpy(off._mmap)
        df = df.select(pl.from_epoch("unixnano", time_unit="ns").dt.cast_time_unit("us").alias("timestamp")).with_columns(df).select(pl.exclude("unixnano"))
        header = moss.ChannelHeader(f"{off}",
                       off.header["ChannelNumberMatchingName"],
                      off.framePeriodSeconds,
        off._mmap["recordPreSamples"][0],
        off._mmap["recordSamples"][0],
        pl.DataFrame(off.header))
        ch = cls(df,
                         header)
        return ch
    return from_off, from_off_paths


@app.cell
def __(from_off_paths, moss, off_paths):
    data = from_off_paths(moss.Channels, off_paths)
    data
    return data,


@app.cell
def __(data, mo, off_paths, pathlib):
    data2 = data.with_experiment_state_by_path(pathlib.Path(off_paths[0]).parent /"20240722_run0006_experiment_state.txt")
    mo.plain(data2.channels[1].df)
    return data2,


@app.cell
def __(data2, mo):
    ch = data2.channels[1]
    mo.plain(ch.df)
    return ch,


@app.cell
def __(ch, mo, pl, plt):
    ch2 = ch.rough_cal(
            ["AlKAlpha", "MgKAlpha", "ClKAlpha", "ScKAlpha", "CoKAlpha","MnKAlpha","VKAlpha","CuKAlpha"],
            uncalibrated_col="filtValue",
            calibrated_col="energy_peak_value",
            ph_smoothing_fwhm=50,
        use_expr=pl.col("state_label")=="START"
        )
    ch2.step_plot(0)
    mo.mpl.interactive(plt.gcf())
    return ch2,


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run()
