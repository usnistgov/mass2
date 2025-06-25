import marimo

__generated_with = "0.7.19"
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
def __(moss, off_paths):
    data = moss.Channels.from_off_paths(
        off_paths, "ebit_20240722_0006"
    ).with_experiment_state_by_path()
    data
    return data,


@app.cell
def __(data, pl):
    data2 = data.map(
        lambda ch: ch.with_columns(
            ch.df.select(filtPhase=pl.col("derivativeLike") / pl.col("filtValue"))
        )
        .with_good_expr_below_nsigma_outlier_resistant(
            [("pretriggerDelta", 5), ("residualStdDev", 10)],
        )
        .with_good_expr(pl.col("filtValue") > 0)
        .with_good_expr_nsigma_range_outlier_resistant([("filtPhase", 10)])
        .driftcorrect(indicator_col="pretriggerMean", uncorrected_col="filtValue")
        .rough_cal(
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
def __(data3, moss, pl):
    # test that calibration with 1-3 peaks works, it just uses the brightest peaks for now
    _ch = (
        data3.ch0.rough_cal_combinatoric(
            line_names=["AlKAlpha"],
            uncalibrated_col="filtValue_dc_pc",
            calibrated_col="dummy1",
            use_expr=pl.col("state_label") == "START",
            ph_smoothing_fwhm=50,
        )
        .rough_cal_combinatoric(
            line_names=["AlKAlpha", "ClKAlpha"],
            uncalibrated_col="filtValue_dc_pc",
            calibrated_col="dummy2",
            use_expr=pl.col("state_label") == "START",
            ph_smoothing_fwhm=50,
        )
        .rough_cal_combinatoric(
            line_names=["AlKAlpha", "ClKAlpha", "MgKAlpha"],
            uncalibrated_col="filtValue_dc_pc",
            calibrated_col="dummy3",
            use_expr=pl.col("state_label") == "START",
            ph_smoothing_fwhm=50,
        )
    )
    _ch.step_plot(-1)
    moss.show()

    return


@app.cell
def __(data2, pl):
    data3 = data2.map(
        lambda ch: ch.phase_correct_mass_specific_lines(
            indicator_col="filtPhase",
            uncorrected_col="filtValue_dc",
            line_names=[
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
            previous_cal_step_index=-1,
        ).rough_cal(
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
            uncalibrated_col="filtValue_dc_pc",
            use_expr=pl.col("state_label") == "START",
        )
    )
    # we should have mass stop print all this output
    return data3,


@app.cell
def __(data3, mass, pl):
    def label_lines(ch, previous_step_index, line_names=None, line_width=80):
        previous_step, previous_step_index = ch.get_step(previous_step_index)
        if line_names is None:
            line_names = previous_step.assignment_result.names_target
        (line_names, line_energies) = mass.algorithms.line_names_and_energies(line_names)
        df_close = pl.DataFrame(
            {"line_name": line_names, "line_energy": line_energies}
        ).sort(by="line_energy")
        assert ch.df["timestamp"].is_sorted()
        df2 = (
            ch.df.select(previous_step.output[0], "timestamp")
            .sort(by=previous_step.output[0])
            .join_asof(
                df_close,
                left_on=previous_step.output[0],
                right_on="line_energy",
                strategy="nearest",
                tolerance=line_width,
            )
            .sort(by="timestamp")
        )
        return ch.with_columns(df2.select("line_name"))

    ch3 = label_lines(data3.ch0, -1)
    return ch3, label_lines


@app.cell
def __(data3, mo, plt):
    data3.ch0.step_plot(-1)
    mo.mpl.interactive(plt.gcf())
    return


@app.cell
def __(data3, mo, pl, plt):
    result = data3.ch0.linefit(
        "AlKAlpha", "energy_filtValue_dc_pc", use_expr=pl.col("state_label") == "START"
    )
    result.plotm()
    mo.mpl.interactive(plt.gcf())
    return result,


@app.cell
def __(ch3, mo, pl, plt):
    ch3.plot_scatter(
        x_col=pl.col("filtPhase"),
        y_col="energy_filtValue_dc_pc",
        color_col="line_name",
        use_expr=pl.col("state_label") == "START",
    )
    plt.grid()
    mo.mpl.interactive(plt.gcf())
    return


@app.cell
def __(data2, mo, plt):
    data2.ch0.plot_scatter(
        x_col="pretriggerMean",
        y_col="energy_filtValue_dc",
        color_col="state_label",
    )
    mo.mpl.interactive(plt.gcf())
    return


@app.cell
def __(data2, mo, moss, pl, plt):
    multifit = moss.MultiFit(
        default_fit_width=80,
        default_use_expr=pl.col("state_label") == "START",
        default_bin_size=0.6,
    )
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
    mf_result = multifit.fit_ch(data2.ch0, "energy_filtValue_dc")
    mf_result.plot_results()
    mo.mpl.interactive(plt.gcf())
    return mf_result, multifit


@app.cell
def __(data3, mo, multifit, plt):
    mf_result_pc = multifit.fit_ch(data3.ch0, "energy_filtValue_dc_pc")
    mf_result_pc.plot_results()
    mo.mpl.interactive(plt.gcf())
    return mf_result_pc,


if __name__ == "__main__":
    app.run()
