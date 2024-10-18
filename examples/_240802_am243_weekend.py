import marimo

__generated_with = "0.9.10"
app = marimo.App(width="medium")


@app.cell
def __():
    import marimo as mo
    import moss
    import numpy as np
    import pylab as plt
    from pathlib import Path
    import polars as pl
    import moss.mass_add_lines
    import lmfit
    return Path, lmfit, mo, moss, np, pl, plt


@app.cell
def __(Path, np):
    bin_path = (
        Path("E:/240802_161833_Am243_2ch_weekend")
        / "Dev2_ai6"
        / "data.bin"
    )
    trigger_filter = np.array([1] * 10 + [-1] * 10)
    threshold = 125
    min_frames_until_next = 4000
    min_frames_from_last = 4000
    energy_of_highest_peak_ev = 5364.1e3

    npre = 400
    npost= 700
    return (
        bin_path,
        energy_of_highest_peak_ev,
        min_frames_from_last,
        min_frames_until_next,
        npost,
        npre,
        threshold,
        trigger_filter,
    )


@app.cell
def __(bin_path, moss, threshold, trigger_filter):
    bin = moss.TrueBqBin.load(bin_path)
    trigger_result = bin.trigger(trigger_filter, threshold, limit_hours=200)
    # I notice that my memory usage increases significaintly when running bin.trigger
    # the function is written to only load a small amount of data at a time from a large mmap'd file
    # I believe the memory fills due to how the OS optimizes the mmap access.... it's better to use as much ram as available
    # so as we iterate through the large file, the OS just doesn't release the memory because it doesn't need to
    # it should be able to release that memory as needed to keep the computer running well
    trigger_result.plot(decimate=10, n_limit=100000, offset=0, x_axis_time_s=True)
    moss.show()
    return bin, trigger_result


@app.cell
def __():
    # long_noise = trigger_result.get_noise(
    #     n_dead_samples_after_pulse_trigger=100000, n_record_samples=500000
    # )
    # long_noise.spectrum().plot()
    # moss.show()
    return


@app.cell
def __(npost, npre, trigger_result):
    ch = trigger_result.to_channel_mmap(
        noise_n_dead_samples_after_pulse_trigger=100000,
        npre=npre,
        npost=npost,
        invert=True,
    )
    ch.df
    return (ch,)


@app.cell
def __(
    ch,
    energy_of_highest_peak_ev,
    min_frames_from_last,
    min_frames_until_next,
    phi0_dac_units,
    pl,
):
    ch2 = ch.summarize_pulses()
    ch2 = ch2.filter5lag()
    ch2 = ch2.rough_cal_combinatoric(
        [energy_of_highest_peak_ev],
        uncalibrated_col="5lagy",
        calibrated_col="energy_5lagy",
        ph_smoothing_fwhm=5,
    )
    ch2 = ch2.with_columns(
        ch2.df.select(
            frames_until_next=pl.col("framecount").shift(-1) - pl.col("framecount"),
            frames_from_last=pl.col("framecount") - pl.col("framecount").shift(),
        )
    )
    ch2 = ch2.with_good_expr(
        (pl.col("frames_until_next") > min_frames_until_next).and_(
            pl.col("frames_from_last") > min_frames_from_last
        )
    )  # good_expr is automatically used in plots, but is ignored for categorization and final livetime hist
    ch2=ch2.with_columns(ch2.df.select(pretrig_mean_orig=pl.col("pretrig_mean"),pretrig_mean=pl.col("pretrig_mean")%phi0_dac_units))
    # mo.stop(not run_button.value, "Click `click to run` recompute all figures")
    return (ch2,)


@app.cell
def __(ch2, moss, np, npre, pl):
    pulses = ch2.df["pulse"].to_numpy()
    avg_pulse = ch2.good_series("pulse", use_expr=True).limit(1000).to_numpy().mean(axis=0)
    avg_pulse = avg_pulse - np.mean(avg_pulse)
    template = avg_pulse / np.sqrt(np.dot(avg_pulse, avg_pulse))


    def residual_rms(pulse):
        pulse = pulse - np.mean(pulse)
        dot = np.dot(pulse, template)
        pulse2 = dot * template
        residual = pulse2 - pulse
        return moss.misc.root_mean_squared(residual)


    residual_rmss = [residual_rms(pulses[i, :]) for i in range(pulses.shape[0])]
    ch3 = ch2.with_columns(pl.Series("residual_rms", residual_rmss))
    ch3 = ch3.with_columns(
        ch3.df.select(residual_rms_range=pl.col("residual_rms").cut([0, 23, 10000]))
    )
    def frontload(pulse):
        a,b = npre, npre+20
        pulse_pt_sub = pulse - np.mean(pulse[:npre])
        area_front = np.sum(pulse_pt_sub[a:b])
        area_rest = np.sum(pulse_pt_sub[b:])
        return area_front/area_rest
    ch3 = ch3.with_columns(pl.Series("frontload", [frontload(pulses[i, :]) for i in range(pulses.shape[0])]))
    ch3 = ch3.with_columns(ch3.df.select(spikey=pl.col("frontload")>0.04))
    return (
        avg_pulse,
        ch3,
        frontload,
        pulses,
        residual_rms,
        residual_rmss,
        template,
    )


@app.cell
def __(ch3, min_frames_from_last, min_frames_until_next, np, pl):
    cat_cond = {
        "first_and_last": True,
        "clean": pl.all_horizontal(pl.all().is_not_null()),
        "large_residual_rms": pl.col("residual_rms") > 12,
        "spikey": pl.col("spikey"),
        "to_close_to_next": pl.col("frames_until_next") < min_frames_until_next,
        "too_close_to_last": pl.col("frames_from_last") < min_frames_from_last,
        "to_close_to_last_and_next": (
            pl.col("frames_from_last") < min_frames_from_last
        ).and_(pl.col("frames_until_next") < min_frames_until_next),
    }


    def categorize_df(df, cat_cond):
        """returns a series showing which category each pulse is in
        pulses will be assigned to the last category for which the condition evaluates to True"""
        dtype = pl.Enum(cat_cond.keys())
        physical = np.zeros(len(df), dtype=int)
        for category_int, (category_str, condition_expr) in enumerate(cat_cond.items()):
            if condition_expr is True:
                in_category = np.ones(len(df), dtype=bool)
            else:
                in_category = (
                    df.select(a=condition_expr).fill_null(False).to_numpy().flatten()
                )
            assert in_category.dtype == bool
            physical[in_category] = category_int
        category = pl.Series("category", physical).cast(dtype)
        return category


    ch4 = ch3.with_columns(categorize_df(ch3.df, cat_cond))
    return cat_cond, categorize_df, ch4


@app.cell
def __(ch, moss, plt):
    plt.plot(ch.df["pulse"][:20].to_numpy().T)
    # plt.plot(
    #     ch3.df.filter(ch3.good_expr, pl.col("residual_rms") > 12)["pulse"]
    #     .limit(20)
    #     .to_numpy()
    #     .T
    # )
    plt.title("first 20 pulses")
    plt.xlabel("framecount")
    plt.ylabel("signal (arb)")
    moss.show()
    return


@app.cell
def __(avg_pulse, moss, plt):
    plt.plot(avg_pulse)
    plt.xlabel("framecount")
    plt.ylabel("signal (arb)")
    moss.show()
    return


@app.cell
def __(ch2, moss, np, plt):
    ch2.plot_hist("energy_5lagy", np.arange(0, 6500000, 2000))
    plt.yscale("log")
    moss.show()
    return


@app.cell
def __(ch3, moss):
    ch3.plot_scatter("5lagx", "energy_5lagy", color_col="residual_rms_range")
    moss.show()
    return


@app.cell
def __(ch3, moss):
    ch3.plot_scatter("frontload", "energy_5lagy", color_col="residual_rms_range")
    moss.show()
    return


@app.cell
def __(ch3, moss):
    ch3.plot_scatter("pretrig_mean", "energy_5lagy", color_col="residual_rms_range")
    moss.show()
    return


@app.cell
def __():
    phi0_dac_units = 1880
    phi0_dac_units
    return (phi0_dac_units,)


@app.cell
def __(ch2, moss):
    ch2.plot_scatter("framecount", "pretrig_mean")
    moss.show()
    return


@app.cell
def __(ch3, moss):
    ch3.plot_scatter("residual_rms", "energy_5lagy", color_col="residual_rms_range")
    moss.show()
    return


@app.cell
def __(cat_cond, mo):
    dropdown_pulse_category = mo.ui.dropdown(
        cat_cond.keys(),
        value=list(cat_cond.keys())[0],
        label="choose pulse category to plot",
    )
    return (dropdown_pulse_category,)


@app.cell
def __(ch4, dropdown_pulse_category, mo, moss, pl, plt):
    _cat = dropdown_pulse_category.value
    _pulses = (
        ch4.df.lazy()
        .filter(pl.col("category") == _cat, pl.col("spikey").not_())
        .limit(50)
        .select("pulse")
        .collect()
        .to_series()
        .to_numpy()
    )
    plt.plot(_pulses.T)
    plt.suptitle(_cat)
    plt.xlabel("sample number")
    plt.ylabel("signal (dac units)")
    mo.vstack([dropdown_pulse_category, moss.show()])
    return


@app.cell
def __(ch4, dropdown_pulse_category, mo, moss, pl, plt):
    _cat = 'to_close_to_next'
    print(f"{_cat=}")
    _pulses = (
        ch4.df.lazy()
        .filter(pl.col("category") == _cat, pl.col("spikey").not_())
        .limit(50)
        .select("pulse")
        .collect()
        .to_series()
        .to_numpy()
    )
    plt.plot(_pulses.T)
    plt.suptitle(_cat)
    plt.xlabel("sample number")
    plt.ylabel("signal (dac units)")
    mo.vstack([dropdown_pulse_category, moss.show()])
    return


@app.cell
def __(ch4, mo):
    def get_step_desc(i):
        return str(ch4.get_step(i)[0])[:50]


    step_desc_list = [f"stepnum={i}, {get_step_desc(i)}" for i in range(len(ch4.steps))]
    dropdown_step = mo.ui.dropdown(
        step_desc_list, value=step_desc_list[0], label="choose step"
    )


    def step_num():
        return step_desc_list.index(dropdown_step.value)
    return dropdown_step, get_step_desc, step_desc_list, step_num


@app.cell
def __(ch4):
    stepplotter = (
        ch4.mo_stepplots()
    )  # can't show stepplotter in same cell it's defined, just how marimo works
    return (stepplotter,)


@app.cell
def __(stepplotter):
    stepplotter.show()
    return


@app.cell
def __(ch4, min_frames_from_last, moss, np, pl, plt):
    # live time clean hist
    def livetime_clean_energies():
        df = (
            ch4.df.lazy()
            .filter(pl.col("category").is_in(["clean"]))
            .select("energy_5lagy", "frames_from_last")
            .collect()
        )
        live_frames = np.sum(df["frames_from_last"].to_numpy() - min_frames_from_last)
        live_time_s = live_frames * ch4.header.frametime_s
        return df["energy_5lagy"], live_time_s


    energies, live_time_s = livetime_clean_energies()
    bin_edges = np.arange(0, 6000000, 1000.0)
    _, counts = moss.misc.hist_of_series(energies, bin_edges=bin_edges)
    bin_centers, bin_size = moss.misc.midpoints_and_step_size(bin_edges)
    plt.plot(bin_centers, counts)
    plt.xlabel("energy / eV")
    plt.ylabel(f"counts / {bin_size:.1f} eV bin")
    total_count_rate = np.sum(counts) / live_time_s
    total_count_rate_unc = np.sqrt(np.sum(counts)) / live_time_s
    plt.title(
        f"{np.sum(counts)} counts, {live_time_s:.2f} s live time, total count rate = {total_count_rate:.2f}+/-{total_count_rate_unc:.2f}/s"
    )
    plt.yscale("log")
    moss.show()
    return (
        bin_centers,
        bin_edges,
        bin_size,
        counts,
        energies,
        live_time_s,
        livetime_clean_energies,
        total_count_rate,
        total_count_rate_unc,
    )


@app.cell
def __(ch4, lmfit, moss, pl):
    result = ch4.linefit(
        "Am241Q",
        "energy_5lagy",
        use_expr=pl.col("category") == "clean",
        dlo=2e5,
        dhi=2e5,
        binsize=1000,
        params_update=lmfit.create_params(fwhm={"value":4000,"min":500}, peak_ph=5.4e6, dph_de={"vary":True, "min":0.9, "max":1.1})
    )
    result.plotm()
    moss.show()
    return (result,)


@app.cell
def __(result):
    result.plot(show_init=True)
    return


@app.cell
def __(ch4, pl):
    # this shows the dataframe without pulse data
    ch4.df.select(pl.exclude("pulse"))
    # uncomment below to save dataframe
    # ch4.df.write_parquet(f"{ch4.description}.parquet")
    return


if __name__ == "__main__":
    app.run()
