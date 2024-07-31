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
    ).map(
        lambda ch: ch.with_good_expr_below_nsigma_outlier_resistant(
            [("pretriggerDelta", 5), ("residualStdDev", 10)], and_=pl.col("filtValue") > 0
        ).driftcorrect(indicator_col="pretriggerMean", uncorrected_col="filtValue")
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
            n_extra=5,
        )
    )
    ch3 = data3.channels[1]
    ch3.step_plot(1)
    mo.mpl.interactive(plt.gcf())
    return ch3, data3


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
def __(data3, multifit):
    data4 = data3.map(
        lambda ch: ch.multifit_spline_cal(
            multifit, previous_cal_step_index=1, calibrated_col="energy_mf_filtValue"
        )
    )
    return data4,


@app.cell
def __(data4, mo, pl, plt):
    ch4 = data4.ch0.with_good_expr_below_nsigma_outlier_resistant(
        [("pretriggerDelta", 5), ("residualStdDev", 10)], and_=pl.col("filtValue") > 0
    )
    df = ch4.good_df()
    states = df["state_label"].unique()
    plt.figure()
    for state in states:
        dfl = df.filter(pl.col("state_label") == state)
        plt.plot(dfl["pretriggerMean"], dfl["energy_filtValue"], ".", label=f"{state=}")
    plt.grid()
    plt.legend()
    mo.mpl.interactive(plt.gcf())
    return ch4, df, dfl, state, states


@app.cell
def __(data4):
    data5 = data4.map(
        lambda ch: ch.driftcorrect(
            indicator_col="pretriggerMean", uncorrected_col="filtValue"
        )
    )
    return data5,


@app.cell
def __(data4, mo, plt):
    data4.channels[1].step_plot(1)
    mo.mpl.interactive(plt.gcf())
    return


@app.cell
def __(mass, np):
    ph = np.array(
        [
            8748.31081736,
            15168.90793839,
            7404.17301595,
            27292.39192824,
            22989.17317181,
            29516.26677965,
            36392.897071,
            31777.79574604,
            13418.9424533,
            41083.68697051,
            34064.80780035,
            24869.97719785,
            10201.60751776,
            18925.57151071,
        ]
    )
    e = np.array(
        (
            1253.688,
            1486.708,
            2622.44,
            3313.9476,
            4090.735,
            4952.216,
            5898.801,
            6930.378,
            8047.8227,
        )
    )

    line_names, _ = mass.algorithms.line_names_and_energies(
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
        ]
    )

    # start with the first peak, try assigning it to the first energy
    # then assume constant gain
    # rank the rest of the energies by how close they are to the 2nd peak, and pick the top
    # then assume linear gain
    # then find the best 3rd peak
    # and calculate the residuals
    # pick best 3 assignments
    # dff = pl.DataFrame({"ind":np.arange(len(e)),"e": e})
    # dfph = pl.DataFrame({"ph_ind":np.arange(len(ph)), "ph":ph})
    # for peak0_ind in range(len(e))[:1]:
    #     peak0_e = e[peak0_ind]
    #     gain0 = ph[0]/peak0_e
    #     df0 = dff.with_columns(pred_ph0=pl.col("e")*gain0).filter(pl.col("e")!=peak0_e)
    #     ranked_guesses0 = df0.sort(np.abs((pl.col("e")*gain0-ph[1])))
    #     for peak1_ind in ranked_guesses0["ind"]:
    #         peak1_e = e[peak1_ind]
    #         gain1 = ph[:2]/np.array([peak0_e, peak1_e])
    #         pfit_gain1 = np.polynomial.Polynomial.fit(ph[:2], gain1, deg=1)
    #         # in form gain = m*ph+b
    #         # energy = ph/(m*ph+b)
    #         # so ph = -energy*b/(energy*m-1)
    #         b, m = pfit_gain1.convert().coef
    #         def energy2ph(energy):
    #             return -energy*b/(energy*m-1)
    #         df1 = df0.with_columns(pred_ph1=energy2ph(df0["e"].to_numpy())).filter(pl.col("e")!=peak1_e).sort("pred_ph1")
    #         ranked_guesses1 = df1.sort(np.abs((pl.col("e")*gain0-ph[2])))
    #         df_joined = ranked_guesses1.join_asof(dfph.sort("ph"), left_on="pred_ph1",right_on="ph", strategy="nearest")
    #         if len(df_joined["ph_ind"].unique())==len(df_joined):
    #             print("success??")
    #             print(df_joined)
    return e, line_names, ph


@app.cell
def __(expected_gain, np, pl):
    def rank_3peak_assignments(
        ph,
        e,
        line_names,
        exepcted_gain=6,
        max_fractional_energy_error_3rd_assignment=0.1,
        minimum_gain_fraction_at_ph_30k=0.25,
    ):
        # we explore possible line assignments, and down select based on knowledge of gain curve shape
        # gain = ph/e, and we assume gain starts at zero, decreases with pulse height, and
        # that a 2nd order polynomial is a reasonably good approximation
        # with one assignment we model the gain as constant, and use that to find the most likely
        # 2nd assignments, then we model the gain as linear, and use that to rank 3rd assignments
        dfe = pl.DataFrame({"e0_ind": np.arange(len(e)), "e0": e, "name": line_names})
        dfph = pl.DataFrame({"ph0_ind": np.arange(len(ph)), "ph0": ph})

        #### 1st assignments ####
        # e0 and ph0 are the first assignment
        # 1) exclude assignments when the gain is too far from the input `expected_gain`
        df0 = (
            dfe.join(dfph, how="cross")
            .with_columns(gain0=pl.col("ph0") / pl.col("e0"))
            .filter(np.abs(pl.col("gain0") - expected_gain) / expected_gain < 0.3)
        )
        #### 2nd assignments ####
        # e1 and ph1 are the 2nd assignment
        df1 = (
            df0.join(df0, how="cross")
            .rename({"e0_right": "e1", "ph0_right": "ph1"})
            .drop("e0_ind_right", "ph0_ind_right", "gain0_right")
        )
        # 1) keep only assignments with e0<e1 and ph0<ph1 to avoid looking at the same pair in reverse
        df1 = df1.filter(pl.col("e0") < pl.col("e1")).filter(pl.col("ph0") < pl.col("ph1"))
        # 2) the gain slope must be negative
        df1 = (
            df1.with_columns(gain1=pl.col("ph1") / pl.col("e1"))
            .with_columns(
                gain_slope=(pl.col("gain1") - pl.col("gain0"))
                / (pl.col("ph1") - pl.col("ph0"))
            )
            .filter(pl.col("gain_slope") < 0)
        )
        # 3) the gain slope should not have too large a magnitude
        df1 = df1.with_columns(
            gain_at_0=pl.col("gain0") - pl.col("ph0") * pl.col("gain_slope")
        )
        df1 = df1.with_columns(
            gain_frac_at_ph30k=(1 + 30000 * pl.col("gain_slope") / pl.col("gain_at_0"))
        )
        df1 = df1.filter(pl.col("gain_frac_at_ph30k") > minimum_gain_fraction_at_ph_30k)

        #### 3rd assignments ####
        # e2 and ph2 are the 3rd assignment
        df2 = df1.join(df0.select(e2="e0", ph2="ph0"), how="cross")
        df2 = df2.with_columns(
            gain_at_ph2=pl.col("gain_at_0") + pl.col("gain_slope") * pl.col("ph2")
        )
        df2 = df2.with_columns(e_at_ph2=pl.col("ph2") / pl.col("gain_at_ph2")).filter(
            pl.col("e1") < pl.col("e2")
        )
        # 1) rank 3rd assignments by energy error at ph2 assuming gain = gain_slope*ph+gain_at_0
        # where gain_slope and gain are calculated from assignments 1 and 2
        df2 = df2.with_columns(e_err_at_ph2=pl.col("e_at_ph2") - pl.col("e2")).sort(
            by=np.abs(pl.col("e_err_at_ph2"))
        )
        # 2) return a dataframe downselected to the assignments and the ranking criteria
        # 3) throw away assignments with large (default 10%) energy errors
        df3peak = df2.select("e0", "ph0", "e1", "ph1", "e2", "ph2", "e_err_at_ph2").filter(
            np.abs(pl.col("e_err_at_ph2") / pl.col("e2"))
            < max_fractional_energy_error_3rd_assignment
        )
        return df3peak, dfe
    return rank_3peak_assignments,


@app.cell
def __(e, line_names, ph, rank_3peak_assignments):
    df3peak, _dfe = rank_3peak_assignments(ph, e, line_names)
    df3peak
    return df3peak,


@app.cell
def __(
    assign_pfit_gain,
    df3peak,
    e,
    line_names,
    mo,
    moss,
    np,
    ph,
    pl,
    plt,
):
    def eval_3peak_assignment_pfit_gain(ph_assigned, e_assigned, possible_phs, line_energies, line_names):
        gain_assigned = np.array(ph_assigned)/np.array(e_assigned)
        pfit_gain = np.polynomial.Polynomial.fit(ph_assigned, gain_assigned, deg=2)
        def ph2energy(ph):
            gain = assign_pfit_gain(ph)
            return ph / gain
        def energy2ph(energy):
            import scipy.optimize
        
            sol = scipy.optimize.root_scalar(
                lambda ph: ph2energy(ph) - energy, bracket=[1, 1e5]
            )
            assert sol.converged
            return sol.root   
        predicted_ph = [energy2ph(_e) for _e in line_energies]
        df = pl.DataFrame({"line_energy": line_energies, "line_name": line_names, "predicted_ph": predicted_ph}).sort(by="predicted_ph")
        dfph = pl.DataFrame({"possible_ph":possible_phs, "ph_ind":np.arange(len(possible_phs))}).sort(by="possible_ph")
        # for each e find the closest possible_ph to the calculaed predicted_ph
        # we started with assignments for 3 energies
        # now we have assignments for all energies
        df = df.join_asof(dfph, left_on="predicted_ph", right_on="possible_ph", strategy="nearest")

        # now we evaluate the assignment and create a result object
        residual_e, pfit_gain = moss.rough_cal.find_pfit_gain_residual(
            df["possible_ph"].to_numpy(), df["line_energy"].to_numpy()
        )
        result = moss.rough_cal.BestAssignmentPfitGainResult(
            np.std(residual_e),
            ph_assigned=df["possible_ph"].to_numpy(),
            residual_e=residual_e,
            assignment_inds=df["ph_ind"].to_numpy(),
            pfit_gain=pfit_gain,
            energy_target=df["line_energy"].to_numpy(),
            names_target=df["line_name"].to_list(),
            ph_target=possible_phs)
        return result
        
    eval_3peak_assignment_pfit_gain([df3peak[0]["ph0"][0], df3peak[0]["ph1"][0], df3peak[0]["ph2"][0]],
                                   [df3peak[0]["e0"][0], df3peak[0]["e1"][0], df3peak[0]["e2"][0]], ph, e, line_names).plot()
    mo.mpl.interactive(plt.gcf())
    return eval_3peak_assignment_pfit_gain,


@app.cell
def __(e, line_names, np, ph, pl):
    dfe = pl.DataFrame({"e0_ind": np.arange(len(e)), "e0": e, "name": line_names})
    dfph = pl.DataFrame({"ph0_ind": np.arange(len(ph)), "ph0": ph})
    expected_gain = 6
    max_fractional_energy_error_3rd_assignment = 0.1
    minimum_gain_fraction_at_ph_30k = 0.25
    # we explore possible line assignments, and down select as we god
    # gain = ph/e, and we assume gain starts at zero, decreases with pulse height, and
    # that a 2nd order polynomial is a reasonably good approximation
    # with one assignment we model the gain as constant, and use that to find the most likely
    # 2nd assignments, then we model the gain as linear, and use that to find the most likely
    # 3rd assignments, then try to assign all peaks and do a polynomial fit, and check the residuals
    # to judge the quality of assignment

    #### 1st assignments ####
    # e0 and ph0 are the first assignment
    # 1) exclude assignments when the gain is too far from the input `expected_gain`
    dfgain = (
        dfe.join(dfph, how="cross")
        .with_columns(gain0=pl.col("ph0") / pl.col("e0"))
        .filter(np.abs(pl.col("gain0") - expected_gain) / expected_gain < 0.3)
    )
    #### 2nd assignments ####
    # e1 and ph1 are the 2nd assignment
    dfg = (
        dfgain.join(dfgain, how="cross")
        .rename({"e0_right": "e1", "ph0_right": "ph1"})
        .drop("e0_ind_right", "ph0_ind_right", "gain0_right")
    )
    # 1) keep only assignments with e0<e1 and ph0<ph1 to avoid looking at the same pair in reverse
    dfg = dfg.filter(pl.col("e0") < pl.col("e1")).filter(pl.col("ph0") < pl.col("ph1"))
    # 2) the gain slope must be negative
    dfg = (
        dfg.with_columns(gain1=pl.col("ph1") / pl.col("e1"))
        .with_columns(
            gain_slope=(pl.col("gain1") - pl.col("gain0")) / (pl.col("ph1") - pl.col("ph0"))
        )
        .filter(pl.col("gain_slope") < 0)
    )
    # 3) the gain slope should not have too large a magnitude
    dfg = dfg.with_columns(gain_at_0=pl.col("gain0") - pl.col("ph0") * pl.col("gain_slope"))
    dfg = dfg.with_columns(
        gain_frac_at_ph30k=(1 + 30000 * pl.col("gain_slope") / pl.col("gain_at_0"))
    )
    dfg = dfg.filter(pl.col("gain_frac_at_ph30k") > minimum_gain_fraction_at_ph_30k)

    #### 3rd assignments ####
    # e2 and ph2 are the 3rd assignment
    dfg = dfg.join(dfgain.select(e2="e0", ph2="ph0"), how="cross")
    dfg = dfg.with_columns(
        gain_at_ph2=pl.col("gain_at_0") + pl.col("gain_slope") * pl.col("ph2")
    )
    dfg = dfg.with_columns(e_at_ph2=pl.col("ph2") / pl.col("gain_at_ph2")).filter(
        pl.col("e1") < pl.col("e2")
    )
    # 1) rank 3rd assignments by energy error at ph2 assuming gain = gain_slope*ph+gain_at_0
    # where gain_slope and gain are calculated from assignments 1 and 2
    dfg = dfg.with_columns(e_err_at_ph2=pl.col("e_at_ph2") - pl.col("e2")).sort(
        by=np.abs(pl.col("e_err_at_ph2"))
    )
    # 2) return a dataframe downselected to the assignments and the ranking criteria
    # 3) throw away assignments with large (default 10%) energy errors
    dfg.select("e0", "ph0", "e1", "ph1", "e2", "ph2", "e_err_at_ph2").filter(
        np.abs(pl.col("e_err_at_ph2") / pl.col("e2"))
        < max_fractional_energy_error_3rd_assignment
    )
    return (
        dfe,
        dfg,
        dfgain,
        dfph,
        expected_gain,
        max_fractional_energy_error_3rd_assignment,
        minimum_gain_fraction_at_ph_30k,
    )


@app.cell
def __(dfg, dfph, e, line_names, np, pl):
    # now starting from a dataframe with 3 assignments, ranked by err at 3rd assignment
    # i want to get a quadratic pfit gain
    # calculate e_pred for each ph
    # use join_asof to assign each ph to closest e
    # calcualte residuals
    assign_e = np.array((dfg[0]["e0"][0], dfg[0]["e1"][0], dfg[0]["e2"][0]))
    assign_ph = np.array((dfg[0]["ph0"][0], dfg[0]["ph1"][0], dfg[0]["ph2"][0]))
    assign_gain = assign_ph / assign_e
    assign_pfit_gain = np.polynomial.Polynomial.fit(assign_ph, assign_gain, deg=2)


    def ph2energy(ph):
        gain = assign_pfit_gain(ph)
        return ph / gain


    def energy2ph(energy):
        import scipy.optimize

        sol = scipy.optimize.root_scalar(
            lambda ph: ph2energy(ph) - energy, bracket=[1, 1e5]
        )
        assert sol.converged
        return sol.root


    # pred_e = energy2ph(np.array(ph))
    # dfa = pl.DataFrame({"ph":ph, "pred_e":pred_e})
    # dfa = dfa.sort(by="pred_e").join_asof(dfe.sort(by="e0"), left_on="pred_e", right_on="e0", strategy="nearest")
    # dfa
    pred_ph = [energy2ph(_e) for _e in e]
    dfa = pl.DataFrame({"e": e, "name": line_names, "pred_ph": pred_ph})
    dfa = dfa.sort(by="pred_ph").join_asof(
        dfph.sort(by="ph0"), left_on="pred_ph", right_on="ph0", strategy="nearest"
    )
    dfa
    return (
        assign_e,
        assign_gain,
        assign_pfit_gain,
        assign_ph,
        dfa,
        energy2ph,
        ph2energy,
        pred_ph,
    )


@app.cell
def __(dfa, dfph, mo, moss, np, plt):
    # now I have a df with an assignment, lets evaluate it further
    residual_e, pfit_gain = moss.rough_cal.find_pfit_gain_residual(
        dfa["ph0"].to_numpy(), dfa["e"].to_numpy()
    )
    residual_e, pfit_gain
    result = moss.rough_cal.BestAssignmentPfitGainResult(
        np.std(residual_e),
        ph_assigned=dfa["ph0"].to_numpy(),
        residual_e=residual_e,
        assignment_inds=dfa["ph0_ind"].to_numpy(),
        pfit_gain=pfit_gain,
        energy_target=dfa["e"].to_numpy(),
        names_target=dfa["name"].to_list(),
        ph_target=dfph["ph0"].to_numpy(),
    )
    result.plot()
    mo.mpl.interactive(plt.gcf())
    return pfit_gain, residual_e, result


if __name__ == "__main__":
    app.run()
