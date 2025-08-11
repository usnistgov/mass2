import marimo

__generated_with = "0.14.16"
app = marimo.App(width="medium", app_title="MASS v2 intro")


@app.cell
def _(mo):
    mo.md(
        """
    #MASS version 2: introduction to internals
    MASS is the Microcalorimeter Analysis Software Suite. Version 2 is a replacement for Version 1 of MASS (2011-2025). Version 2 supports many algorithms for pulse filtering, calibration, and corrections. It is built on modern open source data science software, including [Pola.rs](https://pola.rs) and [Marimo](https://marimo.io). MASS v2 supports some key features that v1 struggled with, including:

      * consecutive data set analysis
      * online (aka realtime) analysis
      * easily supporting different analysis chains
    """
    )
    return


@app.cell
def _():
    import polars as pl
    import pylab as plt
    import numpy as np
    import marimo as mo
    import pulsedata
    return mo, np, pl, plt, pulsedata


@app.cell
def _():
    import mass2
    return (mass2,)


@app.cell
def _(mo):
    mo.md(
        """
    # Load data
    Here we load the data, then we explore the internals a bit to show how MASS is built.
    """
    )
    return


@app.cell
def _(mass2, pulsedata):
    _p = pulsedata.pulse_noise_ljh_pairs["20230626"]
    data = mass2.Channels.from_ljh_folder(
        pulse_folder=_p.pulse_folder, noise_folder=_p.noise_folder
    )
    data
    return (data,)


@app.cell
def _(mo):
    mo.md(
        """
    # basic analysis
    The variables `data` is the conventional name for a `Channels` object. It contains a list of
    `Channel` objects, conventinally assigned to a variable `ch` when accessed individualy.
    One `Channel` represents a single pixel, whiles a `Channels` is a collection of pixels, like a whole array.

    The data tends to consist of pulse shapes (arrays of length 100 to 1000 in general) and per pulse quantities,
    such as the pretrigger mean. These data are stored internally as pola.rs `DataFrame` objects.

    The next cell shows a basic analysis on multiple channels. The function `data.transform_channels` takes a
    one-argument function, where the one argument is a `Channel` and the function returns a `Channel`,
      `data.transform_channels` returns a `Channels`. There is no mutation, and we can't
      re-use variable names in a reactive notebook, so we store the result in a new variable `data2`.
    """
    )
    return


@app.cell
def _(data, mass2, mo, plt):
    ch0 = data.ch0.summarize_pulses()
    line_names = ["MnKAlpha", "MnKBeta", "CuKAlpha", "CuKBeta", "PdLAlpha", "PdLBeta"]
    calibrated_col = None
    uncalibrated_col = "peak_value"
    use_expr = True
    max_fractional_energy_error_3rd_assignment: float = 0.1
    min_gain_fraction_at_ph_30k: float = 0.25
    fwhm_pulse_height_units: float = 75
    n_extra_peaks: int = 10
    import mass  # type: ignore

    if calibrated_col is None:
        calibrated_col = f"energy_{uncalibrated_col}"
    (line_names, line_energies) = mass2.algorithms.line_names_and_energies(line_names)
    uncalibrated = ch0.good_series(uncalibrated_col, use_expr=use_expr).to_numpy()
    pfresult = mass2.rough_cal.peakfind_local_maxima_of_smoothed_hist(
        uncalibrated, fwhm_pulse_height_units=fwhm_pulse_height_units
    )
    pfresult.plot()
    possible_phs = pfresult.ph_sorted_by_prominence()[: len(line_names) + n_extra_peaks]
    df3peak, dfe = mass2.rough_cal.rank_3peak_assignments(
        possible_phs,
        line_energies,
        line_names,
        max_fractional_energy_error_3rd_assignment,
        min_gain_fraction_at_ph_30k,
    )
    mo.mpl.interactive(plt.gcf())
    return (df3peak,)


@app.cell
def _(df3peak):
    df3peak
    return


@app.cell
def _(data):
    data2 = data.map(
        lambda channel: channel.summarize_pulses()
        .with_good_expr_pretrig_rms_and_postpeak_deriv()
        .rough_cal_combinatoric(
            ["MnKAlpha", "MnKBeta", "CuKAlpha", "CuKBeta", "PdLAlpha", "PdLBeta"],
            uncalibrated_col="peak_value",
            calibrated_col="energy_peak_value",
            ph_smoothing_fwhm=50,
        )
        .filter5lag(f_3db=10e3)
        .rough_cal_combinatoric(
            ["MnKAlpha", "MnKBeta", "CuKAlpha", "CuKBeta", "PdLAlpha", "PdLBeta"],
            uncalibrated_col="5lagy",
            calibrated_col="energy_5lagy",
            ph_smoothing_fwhm=50,
        )
        .driftcorrect()
        .rough_cal_combinatoric(
            ["MnKAlpha", "MnKBeta", "CuKAlpha", "CuKBeta", "PdLAlpha", "PdLBeta"],
            uncalibrated_col="5lagy_dc",
            calibrated_col="energy_5lagy_dc",
            ph_smoothing_fwhm=50,
        )
    )
    return (data2,)


@app.cell
def _(data2, mass2):
    _ch = data2.channels[4102]
    _ch2 = _ch.rough_cal_combinatoric(
        ["MnKAlpha", "MnKBeta", "CuKAlpha", "CuKBeta", "PdLAlpha", "PdLBeta"],
        uncalibrated_col="5lagy_dc",
        calibrated_col="energy_5lagy_dc2",
        ph_smoothing_fwhm=50,
    )
    _ch2.step_plot(-1)

    mass2.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    # inspecting the data

    Internally, the data is stored in polars `DataFrame`s. Lets take a look. To access the dataframe for one channel we do `data2.channels[4102].df`. In `marimo` we can get a nice UI element to browse through the data by returning the `DataFrame` as the last element in a cell. marimo's nicest display doesn't work with array columns like our pulse column, so lets leave that out for now.
    """
    )
    return


@app.cell
def _(data2, pl):
    data2.channels[4102].df.select(pl.exclude("pulse"))
    return


@app.cell
def _(data2, mo):
    mo.md(
        f"""
    To enable online analysis, we have to keep track of all the steps of our calibration, so each channel has a history of its steps that we can replay. Here we interpolate it into the markdown, each entry is a step name followed by the time it took to perform the step.

        {data2.channels[4102].step_summary()=}
    """
    )
    return


@app.cell
def _(data2, mo):
    _ch_nums = list(str(_ch_num) for _ch_num in data2.channels.keys())
    dropdown_ch = mo.ui.dropdown(
        options=_ch_nums, value=_ch_nums[0], label="channel number"
    )
    _energy_cols = [col for col in data2.dfg().columns if col.startswith("energy")]
    _energy_cols
    dropdown_col = mo.ui.dropdown(
        options=_energy_cols, value=_energy_cols[0], label="energy col"
    )
    mo.md(
        f"""MASS2 has some convenient fitting and plotting methods. We can combine them with marimo's super easy ui element to pick a channel number {dropdown_ch} and an energy column name {dropdown_col}. we can use it to make plots!

        MASS fitters are based on the best avaialble spectral shapes in the literature. The fwhm resolution value refers to only the detector portion of the resolution."""
    )
    return dropdown_ch, dropdown_col


@app.cell
def _(data2, dropdown_ch, dropdown_col, mo, plt):
    _ch_num, _col = int(dropdown_ch.value), dropdown_col.value
    _ch = data2.channels[int(_ch_num)]
    result = _ch.linefit("MnKAlpha", col=_col)
    result.plotm()
    plt.title(f"reative plot of {_ch_num=} and {_col=} for you")
    mo.mpl.interactive(plt.gcf())
    return (result,)


@app.cell
def _(result):
    result.fit_report()
    return


@app.cell
def _(mo):
    mo.md(r"""# plot a noise spectrum""")
    return


@app.cell
def _(ch, mo, plt):
    ch.noise.spectrum().plot()
    mo.mpl.interactive(plt.gcf())
    return


@app.cell
def _(mo):
    mo.md(
        """
    # replay
    here we'll apply the same steps to the original dataframe for one channel to show the replay capability

    `ch = data.channels[4102]` is one way to access the `Channel` from before all the analysis steps. Notice how it only has 3 columns, instead of the many you see for `data.channels[4102]`. The display of steps could really be improved!
    """
    )
    return


@app.cell
def _(data, mo):
    ch = data.channels[4102]
    mo.plain(ch.df)
    return (ch,)


@app.cell
def _(data2):
    steps = data2.channels[4102].steps
    steps
    return (steps,)


@app.cell
def _(mo):
    mo.md(
        """
    # apply steps
    marimo has an outline feature look on the left for the scroll looking icon, and click on it. You can navigate around this notebook with it!

    below we apply all the steps that were saved in data2 to our orignial channel, which recall was saved in `data`.
    """
    )
    return


@app.cell
def _(ch, mo, steps):
    step = steps[0]
    _ch = ch
    for step in steps:
        _ch = _ch.with_step(step)
    ch2 = _ch
    mo.plain(ch2.df)
    return (ch2,)


@app.cell
def _(mo):
    mo.md("""# make sure the results are the same!""")
    return


@app.cell
def _(ch2, data2):
    ch2.df == data2.channels[4102].df
    return


@app.cell
def _(ch2):
    # to help you remember not to mutate, everything is as immutable as python will allow! assignment throws!
    from dataclasses import FrozenInstanceError

    try:
        ch2.df = 4
    except FrozenInstanceError:
        pass
    except:
        raise
    else:
        Exception("this was supposed to throw!")
    return


@app.cell
def _(data, data2, mo, np):
    mo.md(
        f"""
    # don't worry about all the copies
    we are copying dataframes, but we aren't copying the underlying memory, so our memory usage is about the same as it would be
    if we used a mutating style of coding.

    `{np.shares_memory(data.channels[4102].df["subframecount"].to_numpy(), data2.channels[4102].df["subframecount"].to_numpy())=}`
    `{np.shares_memory(data.channels[4102].df["pulse"].to_numpy(), data2.channels[4102].df["pulse"].to_numpy())=}`
    """
    )
    return


@app.cell
def _(mo):
    mo.md("""# step plots""")
    return


@app.cell
def _(ch2, mo, plt):
    ch2.step_plot(4)
    mo.mpl.interactive(plt.gcf())
    return


@app.cell
def _(ch2, mo, plt):
    ch2.step_plot(5)
    mo.mpl.interactive(plt.gcf())
    return


@app.cell
def _(ch2, mo, plt):
    ch2.step_plot(2)
    mo.mpl.interactive(plt.gcf())
    return


@app.cell
def _(mo):
    mo.md(
        """
    # cuts? good_expr and use_expr
    We're using polars expressions in place of cuts. Each `Channel` can work with two of these, `good_expr` which is
    intended to isolate clean pulses that will yield good resolution, and `use_expr` which is intended to time slice
    to seperate out different states of the experiment. However, there is nothing binding these behaviors.

    `good_expr` is stored in the `Channel` and use automatically in many functions, including plots. `use_expr`
    is passed on a per function basis, and is not generally stored, although some steps will store the `use_expr`
    provided during that step. Many functions have something like `plot(df.filter(good_expr).filter(use_expr))` in them.
    """
    )
    return


@app.cell
def _(ch2):
    ch2.good_expr
    return


@app.cell
def _(ch2, pl):
    # here we use a good_expr to drift correct over a smaller energy range
    ch3 = ch2.driftcorrect(
        indicator_col="pretrig_mean",
        uncorrected_col="energy_5lagy_dc",
        use_expr=(pl.col("energy_5lagy_dc").is_between(2800, 2850)),
    )
    return (ch3,)


@app.cell
def _(ch3, mo, plt):
    # here we make the debug plot for that last step, and see that it has stored the use_expr and used it for its plot
    # we can see that this line has a non-optimal drift correction when learning from the full energy range,
    # but is improved when we learn just from it's energy range
    ch3.step_plot(6)
    mo.mpl.interactive(plt.gcf())
    return


@app.cell
def _(ch3):
    ch3.steps[6].use_expr
    return


@app.cell
def _(ch3, mo):
    mo.md(f"""{str(ch3.good_expr)=} remains unchanged""")
    return


@app.cell
def _(data2):
    # here we have a very simple experiment_state_file
    df_es = data2.get_experiment_state_df()
    df_es
    return


@app.cell
def _(data2, pl):
    # due to the way timestamps were define in ljh files, we actually have a non-monotonic timestamp in channel 4109,
    # so let's fix that then apply the experiment state.
    # we also drop the "pulse" column here because sorting will acually copy the underlying data
    # and we dont want to do that
    data3 = data2.map(
        lambda ch: ch.with_replacement_df(
            ch.df.select(pl.exclude("pulse")).sort(by="timestamp")
        )
    )
    data3 = data3.with_experiment_state_by_path()
    return (data3,)


@app.cell
def _(data3):
    # now lets combine the data by calling data.dfg()
    # to get one combined dataframe from all channels
    # and we downselect to just to columns we want for further processing
    dfg = data3.dfg().select("timestamp", "energy_5lagy_dc", "state_label", "ch_num")
    dfg
    return


@app.cell
def _():
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    # todos!
     * external trigger
     * check accuracy of psd level and filter vdv
     * start automated tests
     * open multi ljh example
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    # fine calibration
    first we show off the MultiFit class, then we use it to run a fine calibration step
    then we use the same multifit spec to calibrate our data, and make the debug plot
    """
    )
    return


@app.cell
def _(data3, mass2, mo, plt):
    multifit = mass2.MultiFit(default_fit_width=80, default_bin_size=0.6)
    multifit = (
        multifit.with_line("MnKAlpha")
        .with_line("CuKAlpha")
        .with_line("PdLAlpha")
        .with_line("MnKBeta")
    )
    multifit_with_results = multifit.fit_ch(data3.channels[4102], "energy_5lagy_dc")
    multifit_with_results.plot_results()
    mo.mpl.interactive(plt.gcf())
    return multifit, multifit_with_results


@app.cell
def _(multifit_with_results):
    pd_result, mn_result, mn_kbeta_result, cu_result = multifit_with_results.results
    print(mn_result.params["fwhm"].value, cu_result.params["fwhm"].value)
    assert mn_result.params["fwhm"].value < 3.58
    assert cu_result.params["fwhm"].value < 3.52
    # this is super weird, depending on what energies we use for drift correction, we get wildily different resolutions,
    # including Cu being better than Mn, and we can do sub-3eV Mn
    return


@app.cell
def _():
    return


@app.cell
def _(data3, multifit):
    data4 = data3.map(
        lambda ch: ch.multifit_mass_cal(
            multifit, previous_cal_step_index=5, calibrated_col="energy2_5lagy_dc"
        ),
        allow_throw=True
    )
    return (data4,)


@app.cell
def _(data4, mo, plt):
    data4.channels[4102].step_plot(6)
    mo.mpl.interactive(plt.gcf())
    return


@app.cell
def _(data4):
    steps_dict = {ch_num: ch.steps for ch_num, ch in data4.channels.items()}
    return (steps_dict,)


@app.cell
def _(mass2, steps_dict):
    mass2.misc.pickle_object(steps_dict, filename="example_steps_dict.pkl")
    return


@app.cell
def _(data4):
    data4.dfg().select(
        "timestamp", "energy2_5lagy_dc", "state_label", "ch_num"
    ).write_parquet("example_result.parquet")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    # "multi-ljh analysis"
    we can easily concatenate a `Channel`s or `Channels`s with `Channel.contcat_df`, `Channel.concat_ch`, and
    `channels.concat_data`.
    For now the steps and df history are dropped, since it's not quite clear how to use them helpfully.
    Internally this relies on polars ability to concat `DataFrame`s without allocation.
    """
    )
    return


@app.cell
def _(ch):
    # here we concatenate two channels and check that the length has double
    ch_concat = ch.concat_ch(ch)
    assert 2 * len(ch.df) == len(ch_concat.df)
    return


@app.cell
def _(data4):
    # here we concatenate two `Channels` objects and check that the length of the resulting dfg
    # (remember, this is the df of good pulses) has doubled
    data_concat = data4.concat_data(data4)
    assert 2 * len(data4.dfg()) == len(data_concat.dfg())
    return


@app.cell
def _(mo):
    mo.md(r"""# final coadded spectrum""")
    return


@app.cell
def _(data4, mo, plt):
    _result = data4.linefit("MnKAlpha", col="energy2_5lagy_dc")
    _result.plotm()
    mo.mpl.interactive(plt.gcf())
    return


@app.cell
def _(data2):
    ch6 = data2.channels[4102]
    return (ch6,)


@app.cell
def _(ch6, mass2, np):
    indicator = ch6.good_series("pretrig_mean", use_expr=True).to_numpy()
    uncorrected = ch6.good_series("energy_5lagy_dc", use_expr=True).to_numpy()
    dc_result = mass2.rough_cal.minimize_entropy_linear(
        indicator, uncorrected, bin_edges=np.arange(0, 9000, 1), fwhm_in_bin_number_units=4
    )
    print(dc_result)
    return


@app.cell
def _(ch6, mo, np, plt):
    ch6.plot_hist("energy_5lagy_dc", np.arange(0, 10000, 1))
    mo.mpl.interactive(plt.gcf())
    return


@app.cell
def _(ch6, mo, np, pl, plt):
    def pfit_dc(line_name, ch):
        import mass

        dlo, dhi = 50, 50
        e0 = mass2.STANDARD_FEATURES[line_name]
        pt, e = ch.good_serieses(
            ["pretrig_mean", "energy_5lagy"],
            use_expr=pl.col("energy_5lagy_dc").is_between(e0 - dlo, e0 + dhi),
        )
        ptm = np.mean(pt.to_numpy())
        ptzm = pt - ptm
        ezm = e - np.mean(e.to_numpy())
        plt.plot(ptzm, ezm, ".")
        pfit_dc = np.polynomial.Polynomial.fit(ptzm, ezm, deg=1)
        slope = pfit_dc.deriv(1).convert().coef[0]
        pt_plt = np.arange(-60, 60, 1)
        plt.plot(pt_plt, pfit_dc(pt_plt))
        plt.plot(ptzm, ezm - slope * ptzm, ".")
        plt.title(f"{slope=:.4} {slope/e0=:g} {line_name=}")
        plt.xlabel("pt zero mean")
        plt.ylabel("energy zero mean")
        ch2 = ch.with_columns(
            ch.df.select(
                pl.col("energy_5lagy") - slope * (pl.col("pretrig_mean") - ptm)
            ).rename({"energy_5lagy": f"energy_5lagy_dc_{line_name}"})
        )
        return ch2

    ch7 = pfit_dc("MnKAlpha", ch6)
    fig11 = plt.gcf()
    plt.figure()
    ch8 = pfit_dc("CuKAlpha", ch7)
    fig12 = plt.gcf()
    mo.vstack([mo.mpl.interactive(fig11), mo.mpl.interactive(fig12)])
    return (ch8,)


@app.cell
def _(ch8):
    ch8.df
    return


@app.cell
def _(ch8, mo, plt):
    result1 = ch8.linefit("CuKAlpha", col="energy_5lagy_dc")
    result1.plotm()
    # fig1 = plt.gcf()
    result2 = ch8.linefit("CuKAlpha", col="energy_5lagy_dc_CuKAlpha")
    result2.plotm()
    fig2 = plt.gcf()
    result3 = ch8.linefit("MnKAlpha", col="energy_5lagy_dc_MnKAlpha")
    result3.plotm()
    fig3 = plt.gcf()
    result4 = ch8.linefit("MnKAlpha", col="energy_5lagy_dc")
    result4.plotm()
    # fig4 = plt.gcf()
    mo.vstack([mo.mpl.interactive(fig) for fig in [fig2, fig3]])
    return


@app.cell
def _(data):
    data.ch0.header.df["Filename"][0]
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
