import marimo

__generated_with = "0.20.2"
app = marimo.App(width="medium", app_title="EBIT MASS2 example")


@app.cell
def _():
    import polars as pl
    import pylab as plt
    import numpy as np
    import marimo as mo

    return mo, np, pl, plt


@app.cell
def _():
    import mass2
    import pulsedata
    import pathlib

    return mass2, pulsedata


@app.cell
def _(mass2, pulsedata):
    off_paths = mass2.core.ljhutil.find_ljh_files(
        str(pulsedata.off["ebit_20240723_0000"]), ext=".off"
    )
    return (off_paths,)


@app.cell
def _(mass2, off_paths):
    data = mass2.Channels.from_off_paths(
        off_paths, "ebit_20240723_0000"
    ).with_experiment_state_by_path()
    data.channels[3].df
    return (data,)


@app.cell
def _(off_paths, pl):
    from pathlib import Path

    timing_file_path = Path(off_paths[0]).parent / "time_20240723.txt"
    timing_df = pl.read_csv(timing_file_path, separator=" ").select(
        timestamp="Calibration", calibration_status="Settings:"
    )
    timing_df = timing_df.select(
        "calibration_status", timestamp=pl.from_epoch("timestamp", time_unit="s")
    )
    timing_df
    return Path, timing_df


@app.cell
def _(Path, mo, np, off_paths, plt):
    external_trigger_file_path = (
        Path(off_paths[0]).parent / "20240723_run0000_external_trigger.bin"
    )
    with open(external_trigger_file_path, "rb") as _f:
        _header_line = (
            _f.readline()
        )  # read the one header line before opening the binary data
        external_trigger_subframe_count = np.fromfile(_f, "int64")
    plt.plot(np.diff(external_trigger_subframe_count)[:10000], ".")
    plt.title(
        f"{external_trigger_file_path.stem} is messed up unfortunatley\nthere should only be one value of difference"
    )
    plt.xlabel("external trigger index")
    plt.ylabel("different in subframe counts between external triggers")
    mo.mpl.interactive(plt.gcf())
    return


@app.cell
def _(data, pl, timing_df):
    def with_timing_df(ch):
        # load the ebit calibration source timing file from csv
        df2 = ch.df.join_asof(timing_df, left_on="timestamp", right_on="timestamp")
        s = df2.select(
            state_label2=pl.concat_str(
                ["state_label", "calibration_status"], ignore_nulls=True, separator="_"
            )
        )
        df2 = df2.with_columns(pl.Series(s, dtype=pl.Categorical))
        return ch.with_replacement_df(df2)

    with_timing_df(data.ch0).df
    return (with_timing_df,)


@app.cell
def _(data, pl):
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
    )
    data2.channels[1].df.limit(1000)
    return (data2,)


@app.cell
def _(data2, pl, with_timing_df):
    line_names = [
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
        "CuKAlpha",
    ]
    data3 = data2.map(
        lambda ch: with_timing_df(
            ch.rough_cal_combinatoric(
                line_names,
                uncalibrated_col="filtValue_dc",
                calibrated_col="energy_filtValue_dc",
                ph_smoothing_fwhm=50,
                use_expr=pl.col("state_label") == "START",
                n_extra=6,
            )
        )
    )
    data3 = data3.map(
        lambda ch: ch.phase_correct_mass_specific_lines(
            indicator_col="filtPhase",
            uncorrected_col="filtValue_dc",
            line_names=line_names,
            previous_cal_step_index=-1,
        )
    )
    return data3, line_names


@app.cell
def _(data3, label_lines, line_names, pl):
    data4 = data3.map(lambda ch: label_lines(ch, -2))
    data4 = data4.map(
        lambda ch: ch.rough_cal_combinatoric(
            line_names,
            uncalibrated_col="filtValue_dc_pc",
            calibrated_col="energy_filtValue_dc_pc",
            ph_smoothing_fwhm=50,
            use_expr=pl.col("state_label") == "START",
            n_extra=7,
        )
    )
    return (data4,)


@app.cell
def _(data3, mo):
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
    return (dropdown_ch,)


@app.cell
def _(data3, dropdown_ch, mo, plt):
    data3.channels[int(dropdown_ch.value)].step_plot(-2)
    plt.gcf().suptitle(f"ch{int(dropdown_ch.value)}")
    # plt.tight_layout()
    mo.mpl.interactive(plt.gcf())
    return


@app.cell
def _(data3, dropdown_ch, mo, plt):
    data3.channels[int(dropdown_ch.value)].step_plot(-1)
    plt.gcf().suptitle(f"ch{int(dropdown_ch.value)}")
    plt.grid(True)
    mo.mpl.interactive(plt.gcf())
    return


@app.cell
def _(data2, mo, plt):
    data2.ch0.step_plot(-1)
    mo.mpl.interactive(plt.gcf())
    return


@app.cell
def _(data3, dropdown_ch, mass2, pl):
    def label_lines(ch, previous_step_index, line_names=None, line_width=80):
        previous_step, previous_step_index = ch.get_step(previous_step_index)
        if line_names is None:
            line_names = previous_step.assignment_result.names_target
        (line_names, line_energies) = mass2.calibration.algorithms.line_names_and_energies(line_names)
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

    ch3 = label_lines(data3.channels[int(dropdown_ch.value)], -2)
    return (label_lines,)


@app.cell
def _(data4, mass2, pl):
    result = data4.ch0.linefit(
        "AlKAlpha", "energy_filtValue_dc_pc", use_expr=pl.col("state_label") == "START"
    )
    result.plotm()
    mass2.show()
    return


@app.cell
def _(data4, mo, pl, plt):
    data4.ch0.plot_scatter(
        x_col=pl.col("filtPhase"),
        y_col="energy_filtValue_dc_pc",
        color_col="line_name",
        use_expr=pl.col("state_label") == "START",
    )
    plt.grid()
    mo.mpl.interactive(plt.gcf())
    return


@app.cell
def _(data4, mo, pl, plt):
    data4.ch0.plot_scatter(
        x_col=pl.col("timestamp"),
        y_col="energy_filtValue_dc_pc",
        color_col="state_label2",
        use_expr=pl.lit(True),
    )
    plt.grid()
    mo.mpl.interactive(plt.gcf())
    return


@app.cell
def _(mass2, pl):
    multifit = mass2.MultiFit(
        default_fit_width=80,
        default_use_expr=pl.col("state_label") == "START",
        default_bin_size=0.6,
    )
    multifit = (
        multifit.with_line("AlKAlpha")  # .with_line("MgKAlpha", dlo=50)
        .with_line("ScKAlpha")
        .with_line("VKAlpha")
        .with_line("MnKAlpha")
        .with_line("CoKAlpha")
        # .with_line("CuKAlpha")
        .with_line("ZnKAlpha")
        .with_line("GeKAlpha", dlo=60)
    )
    # mf_result = multifit.fit_ch(data4.channels[int(dropdown_ch.value)], "energy_filtValue_dc_pc")
    # mf_result.plot_results()
    # mo.mpl.interactive(plt.gcf())
    return (multifit,)


@app.cell
def _(data4, multifit):
    data5 = data4.map(
        lambda ch: ch.multifit_mass_cal(
            multifit, previous_cal_step_index=-1, calibrated_col="energy2_filtValue_dc_pc"
        )
    )
    return (data5,)


@app.cell
def _(data5, dropdown_ch, mass2):
    data5.channels[int(dropdown_ch.value)].step_plot(-1)
    mass2.show()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### TODOS
    * hunter run on all data again and report any errors
    * plot rms_residual_energy vs channel number
    * plot gain spline vs channel
    * plot filt_value vs area
    * make a drift at a line plot
    """)
    return


if __name__ == "__main__":
    app.run()
