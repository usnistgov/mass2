import marimo

__generated_with = "0.7.9"
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
    return moss,


@app.cell
def __():
    import mass 
    import os
    massroot = os.path.split(os.path.split(mass.__file__)[0])[0]
    massroot
    return mass, massroot, os


@app.cell
def __(massroot, os):
    noise_folder = os.path.join(massroot,"tests","ljh_files","20230626","0000")
    pulse_folder = os.path.join(massroot,"tests","ljh_files","20230626","0001")
    return noise_folder, pulse_folder


@app.cell
def __(moss, noise_folder, pulse_folder):
    data = moss.Channels.from_ljh_folder(
        pulse_folder=pulse_folder, noise_folder=noise_folder
    )
    data
    return data,


@app.cell
def __(data, mo, np, pl, plt):
    chx=data.channels[4102].summarize_pulses().with_good_expr_pretrig_mean_and_postpeak_deriv().rough_cal(
            ["MnKAlpha", "MnKBeta", "CuKAlpha", "CuKBeta", "PdLAlpha", "PdLBeta"],
            uncalibrated_col="peak_value",
            calibrated_col="energy_peak_value",
            ph_smoothing_fwhm=50,
        )
    chx.step_summary()
    chx.df.select(pl.exclude("pulse"))
    chx.step_plot(1, bin_edges=np.arange(0,10000,3))
    mo.mpl.interactive(plt.gcf())
    return chx,


@app.cell
def __(chx):
    for _df in chx.df_history:
        print(f"{_df.columns=}")
    return


@app.cell
def __():
    return


@app.cell
def __(data):
    data2 = data.transform_channels(
        lambda channel: channel.summarize_pulses()
        .with_good_expr_pretrig_mean_and_postpeak_deriv()
        .rough_cal(
            ["MnKAlpha", "MnKBeta", "CuKAlpha", "CuKBeta", "PdLAlpha", "PdLBeta"],
            uncalibrated_col="peak_value",
            calibrated_col="energy_peak_value",
            ph_smoothing_fwhm=50,
        )
        .filter5lag(f_3db=10e3)
        .rough_cal(
            ["MnKAlpha", "MnKBeta", "CuKAlpha", "CuKBeta", "PdLAlpha", "PdLBeta"],
            uncalibrated_col="5lagy",
            calibrated_col="energy_5lagy",
            ph_smoothing_fwhm=50,
        )
        .driftcorrect()
        .rough_cal(
            ["MnKAlpha", "MnKBeta", "CuKAlpha", "CuKBeta", "PdLAlpha", "PdLBeta"],
            uncalibrated_col="5lagy_dc",
            calibrated_col="energy_5lagy_dc",
            ph_smoothing_fwhm=50,
        )
    )
    return data2,


@app.cell
def __(data2, mo, plt):
    ch2x = data2.channels[4102]
    for i in range(len(ch2x.steps)):
        ch2x.step_plot(i)
    mo.mpl.interactive(plt.gcf())
    return ch2x, i


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        f"""
        # inspecting the data

        Internally, the data is stored in polars `DataFrame`s. Lets take a look. To access the dataframe for one channel we do `data2.channels[4102].df`. In `marimo` we can get a nice UI element to browse through the data by returning the `DataFrame` as the last element in a cell. marimo's nicest display doesn't work with array columns like our pulse column, so lets leave that out for now.
        """
    )
    return


@app.cell
def __(mo):
    mo.md("# make sure the results are the same!")
    return


@app.cell
def __(mo):
    mo.md("# step plots")
    return


@app.cell
def __(mo):
    mo.md(
        """
        # cuts? good_expr and use_expr
        We're using polars expressions in place of cuts. Each `Channel` can work with two of these, `good_expr` which is intended to isolate clean pulses that will yield good resolution, and `use_expr` which is intended to time slice to seperate out different states of the experiment. However, there is nothing binding these behaviors.

        `good_expr` is stored in the `Channel` and use automatically in many functions, including plots. `use_expr` is passed on a per function basis, and is not generally stored, although some steps will store the `use_expr` provided during that step. Many functions have something like `plot(df.filter(good_expr).filter(use_expr))` in them.
        """
    )
    return


@app.cell
def __():
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        # todos!
         * external trigger
         * check accuracy of psd level and filter vdv
         * start automated tests
         * move drift correct into moss
         * move fitting into moss (maybe seperate package?)
         * open multi ljh example
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        # fine calibration
        first we show off the MultiFit class, then we use it to run a fine calibration step
        then we use the same multifit spec to calibrate our data, and make the debug plot
        """
    )
    return


@app.cell
def __(data3, mo, moss, plt):
    multifit = moss.MultiFit(default_fit_width=80, default_bin_size=0.6)
    multifit = multifit.with_line("MnKAlpha").with_line("CuKAlpha").with_line("PdLAlpha")
    multifit_with_results = multifit.fit_ch(data3.channels[4102], "energy_5lagy_dc")
    multifit_with_results.plot_results()
    mo.mpl.interactive(plt.gcf())
    return multifit, multifit_with_results


@app.cell
def __(multifit_with_results):
    pd_result, mn_result, cu_result = multifit_with_results.results
    assert mn_result.params["fwhm"].value < 3.34
    assert cu_result.params["fwhm"].value < 3.7
    return cu_result, mn_result, pd_result


@app.cell
def __(data3, multifit):
    data4= data3.transform_channels(lambda ch: ch.multifit_spline_cal(multifit, previous_cal_step_index=5, calibrated_col="energy2_5lagy_dc"))
    return data4,


@app.cell
def __(data4, mo, plt):
    data4.channels[4102].step_plot(6)
    mo.mpl.interactive(plt.gcf())
    return


@app.cell
def __(data4):
    steps_dict = {ch_num:ch.steps for ch_num, ch in data4.channels.items()}
    return steps_dict,


@app.cell
def __(moss, steps_dict):
    moss.misc.pickle_object(steps_dict, filename="example_steps_dict.pkl")
    return


@app.cell
def __(data4):
    data4.dfg().select("timestamp", "energy2_5lagy_dc", "state_label", "ch_num").write_parquet("example_result.parquet")
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        # "multi-ljh analysis"
        we can easily concatenate a `Channel`s or `Channels`s with `Channel.contcat_df`, `Channel.concat_ch`, and `channels.concat_data`. For now the steps and df history are dropped, since it's not quite clear how to use them helpfully. Internally this relies on polars ability to concat `DataFrame`s without allocation.
        """
    )
    return


@app.cell
def __(ch):
    # here we concatenate two channels and check that the length has double
    ch_concat = ch.concat_ch(ch)
    assert 2*len(ch.df) == len(ch_concat.df)
    return ch_concat,


@app.cell
def __(data4):
    # here we concatenate two `Channels` objects and check that the length of the resulting dfg (remember, this is the df of good pulses) has doubled
    data_concat = data4.concat_data(data4)
    assert 2*len(data4.dfg())==len(data_concat.dfg())
    return data_concat,


@app.cell
def __(mo):
    mo.md(r"# final coadded spectrum")
    return


@app.cell
def __(data4, mo, plt):
    _result = data4.linefit("MnKAlpha",col="energy2_5lagy_dc")
    _result.plotm()
    mo.mpl.interactive(plt.gcf())
    return


@app.cell
def __(data2):
    ch6=data2.channels[4102]
    return ch6,


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run()
