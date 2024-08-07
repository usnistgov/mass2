import marimo

__generated_with = "0.7.17"
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
def __(mass, moss, pulsedata):
    off_paths = moss.ljhutil.find_ljh_files(
        str(pulsedata.off["ebit_20240723_0000"]), ext=".off"
    )
    off = mass.off.OffFile(off_paths[0])
    return off, off_paths


@app.cell
def __(moss, off_paths):
    data = moss.Channels.from_off_paths(off_paths, "ebit_20240723_0000").with_experiment_state_by_path()
    data
    return data,


@app.cell
def __(off_paths, pl):
    from pathlib import Path
    timing_file_path = Path(off_paths[0]).parent/"time_20240723.txt"
    timing_df = pl.read_csv(timing_file_path, separator=" ").select(timestamp="Calibration", calibration_status="Settings:")
    timing_df = timing_df.select("calibration_status", timestamp=pl.from_epoch("timestamp", time_unit="s"))
    timing_df
    return Path, timing_df, timing_file_path


@app.cell
def __(data, pl, timing_df):
    def with_timing_df(ch):
        df2 = ch.df.join_asof(timing_df, left_on="timestamp", right_on="timestamp")
        s = df2.select(state_label2=pl.concat_str(["state_label", "calibration_status"], ignore_nulls=True, separator="_"))
        df2 = df2.with_columns(pl.Series(s, dtype=pl.Categorical))
        return ch.with_replacement_df(df2)
    with_timing_df(data.ch0).df
    return with_timing_df,


@app.cell
def __(data, pl):
    data2 = data.map(
        lambda ch: ch.with_columns(
            ch.df.select(filtPhase=pl.col("derivativeLike") / pl.col("filtValue"))
        )
        .with_good_expr_below_nsigma_outlier_resistant(
            [("pretriggerDelta", 5), ("residualStdDev", 10)],
        )
        .with_good_expr(pl.col("filtValue")>0)
        .with_good_expr_nsigma_range_outlier_resistant([("filtPhase", 10)])
        .driftcorrect(indicator_col="pretriggerMean", uncorrected_col="filtValue")
    )
    data2.channels[1].df.limit(1000)
    return data2,


@app.cell
def __(data2, pl, with_timing_df):
    data3 = data2.map(lambda ch:     with_timing_df(ch.rough_cal_combinatoric(
            [
                "ZnLAlpha",
                "AlKAlpha",
                "ZnKAlpha",
                "ScKAlpha",
                "MnKAlpha",
                # "ClKAlpha", # Cl not visible?
                "VKAlpha",
                "CoKAlpha",
                "GeLAlpha",
                "GeKAlpha",
                "GeKBeta",
                "CuKAlpha"
            ],
            uncalibrated_col="filtValue_dc",
            calibrated_col="energy_filtValue_dc",
            ph_smoothing_fwhm=50,
            use_expr=pl.col("state_label") == "START",
            n_extra=6

        )))
    return data3,


@app.cell(hide_code=True)
def __(data3, mo):
    _ch_nums = list(str(_ch_num) for _ch_num in data3.channels.keys())
    dropdown_ch = mo.ui.dropdown(
        options=_ch_nums, value=_ch_nums[0], label="channel number"
    )
    mo.md(f"""## Rough Cal Guidance
    The rough_cal routine works well most of the time. Always plan to inspect the step plot for it. 

    1. Make sure all the lines you give it are marked as assigned or unassigned.
    2. You can turn up `n_extra` to try more lines.
    3. It's usually better to add more known lines to help it.
    4. The gain vs pulse height curve should be monotonically decreasing.

    Show rough_cal results for {dropdown_ch}. You can change it via this dropdown.
    """)
    return dropdown_ch,


@app.cell
def __(data3, dropdown_ch, mo, plt):
    data3.channels[int(dropdown_ch.value)].step_plot(-1)
    plt.gcf().suptitle(f"ch{int(dropdown_ch.value)}")
    plt.tight_layout()
    mo.mpl.interactive(plt.gcf())
    return


@app.cell
def __(data2, mo, plt):
    data2.ch0.step_plot(-1)
    mo.mpl.interactive(plt.gcf())
    return


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
def __(data3, mo, pl, plt):
    result = data3.ch0.linefit(
        "AlKAlpha", "energy_filtValue_dc", use_expr=pl.col("state_label") == "START"
    )
    result.plotm()
    mo.mpl.interactive(plt.gcf())
    return result,


@app.cell
def __(ch3, mo, pl, plt):
    ch3.plot_scatter(
        x_col=pl.col("filtPhase"),
        y_col="energy_filtValue_dc",
        color_col="line_name",
        use_expr=pl.col("state_label") == "START",
    )
    plt.grid()
    mo.mpl.interactive(plt.gcf())
    return


@app.cell
def __(ch3, mo, pl, plt):
    ch3.plot_scatter(
        x_col=pl.col("timestamp"),
        y_col="energy_filtValue_dc",
        color_col="state_label2",
        use_expr=True,
    )
    plt.grid()
    mo.mpl.interactive(plt.gcf())
    return


@app.cell
def __(data3, mo, moss, pl, plt):
    multifit = moss.MultiFit(default_fit_width=80, default_use_expr=pl.col("state_label")=="START",default_bin_size=0.6)
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
    mf_result = multifit.fit_ch(data3.ch0, "energy_filtValue_dc")
    mf_result.plot_results()
    mo.mpl.interactive(plt.gcf())
    return mf_result, multifit


@app.cell(disabled=True)
def __(data3, mo, multifit, plt):
    mf_result_pc = multifit.fit_ch(data3.ch0, "energy_filtValue_dc_pc")
    mf_result_pc.plot_results()
    mo.mpl.interactive(plt.gcf())
    return mf_result_pc,


if __name__ == "__main__":
    app.run()
