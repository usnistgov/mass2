

import marimo

__generated_with = "0.13.3"
app = marimo.App(width="medium", app_title="MOSS intro")


@app.cell
def _():
    import polars as pl
    import pylab as plt
    import numpy as np
    import marimo as mo
    return mo, np, pl, plt


@app.cell
def _():
    import moss
    return (moss,)


@app.cell
def _():
    import lmfit
    return (lmfit,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""# Load data""")
    return


@app.cell
def _(moss):
    data = moss.Channels.from_ljh_folder(
        pulse_folder=r"D:\Box\TES Data\Pax Data\20250428\0003",
        noise_folder=r"D:\Box\TES Data\Pax Data\20250428\0002",
        limit=400,
    )
    print(data)
    return (data,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Summarize Data""")
    return


@app.cell
def _(data, moss, np, pl):
    def _do_analysis(ch: moss.Channel) -> moss.Channel:
        ch = (
            ch.summarize_pulses(pretrigger_ignore_samples=20)
            .with_good_expr_pretrig_rms_and_postpeak_deriv()
            .correct_pretrig_mean_jumps(period=4096)
            .driftcorrect(
                indicator_col="ptm_jf",
                uncorrected_col="pulse_rms",
                use_expr=(
                    (pl.col("ptm_jf") - pl.col("ptm_jf"))
                    .median()
                    .abs()
                    .is_between(-1000, 1000)
                ),
            )
        )
        ptm1 = ch.df["pretrig_mean"].to_numpy()
        ptm2 = np.unwrap(ptm1 % 4096, period=4096)
        ch = ch.with_columns(pl.DataFrame({"ptm2": ptm2}))
        return ch

    data2 = data.map(_do_analysis)
    data2 = data2.with_experiment_state_by_path()
    return (data2,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Select Channel""")
    return


@app.cell(hide_code=True)
def _(data, mo):
    def _():
        chs = list(data.channels.keys())
        dropdown = mo.ui.dropdown(
            options=chs,  # (label, value)
            label="Select channel",
            value=chs[0],  # default selected value
        )
        return dropdown

    ch_dropdown = _()
    ch_dropdown
    return (ch_dropdown,)


@app.cell
def _(ch_dropdown):
    ch_num = ch_dropdown.value
    return (ch_num,)


@app.cell
def _(ch_num, data2, moss, np):
    data2.channels[ch_num].plot_hist("pulse_rms_dc", np.arange(0, 6000, 1))
    moss.show()
    return


@app.cell
def _(ch_num, data2, mo):
    mo.plain(data2.channels[ch_num].df.limit(5))
    return


@app.cell
def _(ch_num, data2, moss):
    data2.channels[ch_num].plot_scatter("timestamp", "ptm_jf")
    moss.show()
    return


@app.cell
def _(ch_num, data2, moss):
    data2.channels[ch_num].plot_scatter("ptm_jf", "pulse_rms_dc")
    moss.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Calibration""")
    return


@app.cell
def _(data2, moss, pl):
    line_names = [81000, 121800]

    def _do_analysis(ch: moss.Channel) -> moss.Channel:
        return (
            ch.rough_cal_combinatoric(
                line_names,
                uncalibrated_col="pulse_rms_dc",
                calibrated_col="energy_pulse_rms_dc",
                ph_smoothing_fwhm=5,
            )
            .filter5lag()
            .driftcorrect(
                indicator_col="ptm_jf",
                uncorrected_col="5lagy"
            )
            .rough_cal_combinatoric(
                line_names,
                uncalibrated_col="5lagy_dc",
                calibrated_col="energy_5lagy_dc",
                ph_smoothing_fwhm=5,
            )
            .with_good_expr(pl.col("energy_5lagy_dc").is_between(0, 1e6))
        )

    data3 = data2.map(_do_analysis, allow_throw=True)
    return (data3,)


@app.cell
def _(ch_num, data3, moss, plt):
    data3.channels[ch_num].plot_scatter(
        "5lagx", "energy_5lagy_dc"
    )
    plt.grid()
    plt.ylim(80e3, 82e3)
    plt.xlim(-1.5, 1)
    moss.show()
    return


@app.cell
def _(mo):
    mo.md("""# significant pulse-height rise-time correlation causing high energy tails""")
    return


@app.cell
def _(ch_num, data3, moss, plt):
    data3.channels[ch_num].plot_scatter(
        "rise_time", "energy_5lagy_dc"
    )
    plt.grid()
    plt.ylim(80e3, 82e3)
    moss.show()
    return


@app.cell
def _(ch_num, data3, moss):
    data3.channels[ch_num].noise.spectrum().plot()
    moss.show()
    return


@app.cell
def _(ch_num, data3, moss, np, plt):
    def plot_pulses(ch: moss.Channel, n_good_pulses=10, spread_col="timestamp", pulse_col="pulse", n_bad_pulses=5, x_as_time=False, use_expr=True, subtract_pretrig_mean_locally_calculated=True):
        df = ch.good_df(use_expr=use_expr).sort(by=spread_col)
        df_small = df.sort(by=spread_col).gather_every(len(df)//n_good_pulses)
        good_pulses = df_small[pulse_col].to_numpy()
        if n_bad_pulses > 0:
            df_bad = ch.bad_df(use_expr=use_expr).sort(by=spread_col)
            df_small_bad = df_bad.sort(by=spread_col).gather_every(len(df_bad)//n_bad_pulses)
            bad_pulses = df_small_bad[pulse_col].to_numpy()
        x = np.arange(ch.header.n_samples)-ch.header.n_presamples
        x_label = "sample number"
        if x_as_time:
            x *= ch.header.frametime_s*1e3
            x_label = "time (ms)"
        for i in range(n_good_pulses):
            spread_val = df_small[spread_col][i]
            pulse = good_pulses[i, :]
            if subtract_pretrig_mean_locally_calculated:
                pulse = pulse - np.mean(pulse[:int(ch.header.n_presamples*0.8)])
            if spread_col == "timestamp":
                plt.plot(x, pulse, label=f"{spread_val}")
            else:
                plt.plot(x, pulse, label=f"{spread_val:7.3g}")
        for i in range(n_bad_pulses):
            spread_val = df_small[spread_col][i]
            pulse = bad_pulses[i, :]
            if subtract_pretrig_mean_locally_calculated:
                pulse = pulse - np.mean(pulse[:int(ch.header.n_presamples*0.8)])
            if spread_col == "timestamp":
                plt.plot(x, pulse, "--", label=f"X {spread_val}")
            else:
                plt.plot(x, pulse, "--", label=f"X {spread_val:7.3g}")
        plt.legend(title=spread_col, loc="center left", bbox_to_anchor=(1, 0.5))
        plt.xlabel(x_label)
        plt.ylabel("signal (arb)")
        plt.title(ch.header.description)
        plt.tight_layout()
        return plt.gca()

    plot_pulses(data3.channels[ch_num], spread_col="energy_5lagy_dc")
    moss.show()
    return (plot_pulses,)


@app.cell
def _(ch_num, data3, mo, moss, pl, plot_pulses, plt):
    plot_pulses(data3.channels[ch_num], spread_col="rise_time",
                use_expr=pl.col("energy_5lagy_dc").is_between(80000, 82000), n_bad_pulses=0)
    plt.xlim(-25, 75)
    plt.grid()
    mo.vstack([mo.md("# pulse shape vs rise time for narrow energy window\nkind of looks like trigger time variation?\nplus a TESd direct hit"),
               moss.show()])
    return


@app.cell
def _(data3, mo):
    steps = data3.ch0.steps
    steps[0].description
    steps_d = {f"{i} {steps[i].description}": i for i in range(len(steps))}
    dropdown_step = mo.ui.dropdown(steps_d, value=list(steps_d.keys())[-1], label="step")
    return (dropdown_step,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Step Plot Viewer""")
    return


@app.cell
def _(ch_num, data3, dropdown_step, mo, moss):
    _ch = data3.channels[ch_num]
    _ch.step_plot(dropdown_step.value)
    mo.vstack([dropdown_step, moss.show()])
    return


@app.cell
def _(ch_num, data3, moss, np, plt):
    # use this filter to calculate baseline resolution
    _ch = data3.channels[ch_num]
    _df = _ch.noise.df
    for step in _ch.steps:
        _df = step.calc_from_df(_df)
    df_baseline = _df
    df_baseline
    _energy_col = "energy_5lagy_dc"
    plt.hist(df_baseline[_energy_col], bins=np.arange(-100, 100, 5)+np.median(df_baseline[_energy_col]))
    plt.xlabel(f"{_energy_col} from noise traces")
    plt.ylabel("count/bin")
    plt.title("running noise traces through the same analysis steps")
    moss.show()
    return


@app.cell
def _(ch_num, data3, moss, np):
    data3.channels[ch_num].plot_hist("energy_5lagy_dc", bin_edges=np.arange(0, 150000, 10))
    moss.show()
    return


@app.cell
def _(ch_num, data3, lmfit, plt):
    params_update = lmfit.Parameters()
    params_update.add('tail_share_hi', value=0.05, min=0.01, max=1, vary=True)
    params_update.add('tail_tau_hi', value=50, min=5, max=200, vary=True)
    result_ch = data3.channels[ch_num].linefit(81000, "energy_5lagy_dc", dlo=400, dhi=400,
                                               binsize=10, has_tails=True, params_update=params_update)
    result_ch.plotm()
    plt.show()
    return (params_update,)


@app.cell
def _(data3, moss, params_update, pl):
    result_data = data3.linefit(81000, "energy_5lagy_dc", dlo=400,
                                dhi=400, binsize=10,
                                has_tails=True, params_update=params_update,
                                use_expr=pl.col("rise_time").is_between(0.00073, 0.000731))
    result_data.plotm()
    moss.show()
    return


@app.cell
def _(data3, mo, moss, np, plt):
    data3.plot_hist("energy_5lagy_dc", np.arange(0, 300000, 20))
    plt.yscale("log")
    mo.vstack([mo.md("# coadded plot\nonly 2 point cal, fake lines may abound"), moss.show()])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# save analysis recipe""")
    return


@app.cell
def _(data3):
    data3.save_steps(data3.get_path_in_output_folder("steps_dict.pkl"))
    return


@app.cell
def _(data3, mo):
    data3.dfg().select(
        "timestamp", "energy_5lagy_dc", "ch_num"
    ).write_parquet(data3.get_path_in_output_folder("list_mode.parquet"))
    mo.md("""#save "list mode" data as parquet""")
    return


@app.cell
def _(data3):
    ljh_path = data3.get_an_ljh_path()
    base_name, post_chan = ljh_path.name.split('_chan')
    date, run_num = base_name.split("_run")
    print(base_name)
    print(run_num)
    print(date)
    return


if __name__ == "__main__":
    app.run()
