import marimo

__generated_with = "0.15.0"
app = marimo.App(width="medium", app_title="MASS v2 intro")


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


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    # Load data
    Here we load the data from two ljh representing pulse data taken with a TES array looking at a Gd-153 gamma ray source. 

    *	The source is Gd-153
    *	The two strongest peaks in these spectra will be the 97 keV and 103 keV peaks.
    *	Full decay information can be found here: https://www.nndc.bnl.gov/nudat3/DecayRadiationServlet?nuc=153Gd&unc=NDS (it’s the electron capture decay that you want, not the internal transition).
    *	Across all pixels, the median energy resolution that I got was 71eV.
    *	If you want to focus on just one channel initially, try ch 5 (achieved energy resolution 71 eV) or ch 2 (56 eV achieved).
    *	All energy resolutions are at the 97 keV line.

    We use the `pulsedata` package to get the file names for the pulse data and noise data files. The `pulsedata` package contains a few small (< 100 MB) files needed for the example notebooks to work, so it only includes two channels of this dataset.
    """
    )
    return


@app.cell
def _(mass2, mo, pulsedata):
    _p = pulsedata.pulse_noise_ljh_pairs["gamma_20241005"]
    data = mass2.Channels.from_ljh_folder(
        pulse_folder=_p.pulse_folder, noise_folder=_p.noise_folder
    )
    print(data)
    mo.md("Here we see thae we've loaded two channels into a `Channels` object, which contains two `Channel` objects. The channel numbers are 2 and 5.")
    return (data,)


@app.cell
def _(data, mo):
    mo.vstack([mo.md("""# View the raw data for one channel\nThe data is represented as a polars `DataFrame` with one row per pulse. The ljh files have yielded 3 columns, so that we have 3 values representing each pulse. These are:

    * `timestamp` - wall clock time based on the clock on the PC taking the data, often innacurate by many milliseconds, but convenient for comparison to other parts of an experiment that change on times scales greater than 50 ms
    * `pulse` - unsigned int 16 values representing the current vs time of the pulses, each pulse is caused by a gamma ray and our goal is to determine the energy of each pulse. Since this is umux data, 4096 units corresponds to 1 phi0. The value of zero current, and the current gain are both not recorded in an ljh file.
    * `subframecount` - a timestamp based on the data aqusition system clock. May be used for microsecond level timing with external events when using the "external trigger" feature.

    Next we will plot the first few pulses using the `data.ch0` property to access the lowest channel number in `data`. And we plot the noise spectrum for one channel"""),data.ch0.df])
    return


@app.cell
def _(data, mass2, plt):
    plt.figure()
    _pulses = data.ch0.df["pulse"].limit(20).to_numpy().T
    plt.plot(_pulses)
    plt.ylabel("signal (arb)")
    plt.ylabel("sample number")
    plt.title(f"the first 20 pulses of channel {data.ch0.header.description}")
    mass2.show() # makes the plot interactive in marimo
    return


@app.cell
def _(data, mass2):
    data.ch0.noise.spectrum().plot()
    mass2.show()
    return


@app.cell
def _(mo):
    mo.md(
        """
    # Analysis Step 1: summary quantities and `good_expr`
    The variables `data` is the conventional name for a `Channels` object. It contains a list of `Channel` objects, conventinally assigned to a variable `ch` when accessed individualy. One `Channel` represents a single pixel, whiles a `Channels` is a collection of pixels, like a whole array.

    `summarize_data` computes many "summary quantities" for each pulse that may be used to estimate energy, to classify pulses based on how they were generated, and may be used to identify and correct for correlations between pulse height and things that should not effect energy, such as the quiescent current level aka the "pretrigger mean". Below we call `summarize_data` and show the new dataframe, there are now many new columns representing the summary quantities.

    Both `mass2` and `marimo` rely upon treating variables as immutable, so we do not modify `data`, instead we call `data2=dostuff(data)` and only the variable `data2` contains the new summary quantities while `data` remains unchanged. We have used python to enforce immutability to the extent that is possible, don't bend over backwards to mutate objects as you may get unexpected results.

    Since TES arrays consist of many TESs, `mass2` is designed to do the same set of operations to each of many `Channel`, with algorithms that learn analysis parameters specific to each channel. The primary method for this is the `data.map` function, which takes a function with the signature `f(ch: Channel) -> Channel` and applies it to each `Channel`.

    We also call `with_good_expr_pretrig_rms_and_postpeak_deriv` which uses the std deviation of the noise traces to set thresholds for acceptable amounts of tail from a previous pulse and detect a 2nd pulse in the same record. It creates a `good_expr`, which is a polars expression that can be used to filter the `DataFrame` to select only clean pulses. Further steps will usually use the `good_expr` by default when learning neccesary parameters.

    We call `with_good_expr_nsigma_range_outlier_resistant` to exclude pulses with very large `pretrig_mean` excursions.

    Then we call `filter5lag` which learns and applies an optimal filter using the 5lag method. This creates the columns `5lagy` (the pulse height) and `5lagyx` the arrival time of the pulse measured at a subsample level. The quality of this filter depends on the extend to which the `good_expr` selects clean pulses, but it's somewhat forgiving.
    """
    )
    return


@app.cell
def _(data, mass2):
    def analysis_step1(ch: mass2.Channel) -> mass2.Channel: # type annotation helps autocompletions later
        return (ch.summarize_pulses()
                .with_good_expr_pretrig_rms_and_postpeak_deriv()
            .with_good_expr_nsigma_range_outlier_resistant(col_nsigma_pairs=[("pretrig_mean",100)])
               .filter5lag()
               )
    data2 = data.map(analysis_step1)
    data2 = data2.with_experiment_state_by_path()
    data2.ch0.df
    return (data2,)


@app.cell
def _(mo):
    mo.md(
        r"""
    # Overview of data

    Here we plot `pretrig_mean` and `5lagy` vs `timestamp` so we can see that the data is basically steady in time. If it is not, then the analysis is much harder and great efforts should be put into the data aquisition system to fix the instability.

    Then we plot the uncalibrated spectrum of `5lagy` to get a sense of how many lines are in the data, and we can use the width of those lines to set the `BLAH` parameter for `rough_calibration`.
    """
    )
    return


@app.cell
def _(data2, mass2):
    data2.ch0.plot_scatter("timestamp","pretrig_mean")
    mass2.show()
    return


@app.cell
def _(data2, mass2):
    data2.ch0.plot_scatter("timestamp","5lagy")
    mass2.show()
    return


@app.cell
def _(data2, mass2, np):
    data2.ch0.plot_hist(col="5lagy", bin_edges=np.arange(0,16000, 10))
    mass2.show()
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    # Analsyis Step 2: Calibration and Corrections

    Here we call `driftcorrect` which attempts to remove correlation between `pretrig_mean` and `5lagy`. Then we call `rough_cal_combinatoric` which identifies peaks and assigns them to lines. In this dataset the strongest two lines by far are 97 keV and 103 keV, so it will be easy for the algorithm to assign them. Then we do a second drift correct, this time trying to remove the correlation between sub-sample arrival time of the pulse and energy. We use the same algorithm as for the `pretrig_mean` correlation removal, but different columns as arguments.

    Then we plot a calibrated histogram of the whole dataset, and some more debug plots. One checks for correlation between `5lagx` and `energy_5lagy_dc`. The others are auto generated by each analysis step we've done. We have a GUI to view some debug plots that are auto generated.
    """
    )
    return


@app.cell
def _(data2, mass2, pl):
    line_names = [97000, 103000]
    def analysis_step2(ch: mass2.Channel) -> mass2.Channel: # type annotation helps autocompletions later
        return (ch
            .driftcorrect(indicator_col="pretrig_mean",
                          uncorrected_col="5lagy")
            .rough_cal_combinatoric(line_names, 
                                         uncalibrated_col="5lagy_dc",
                                         calibrated_col="energy_5lagy_dc",
                                    ph_smoothing_fwhm=50 # choose something close to the width of the lines in the uncalibrated plot, and small compared to the spacings
                                         )
            .driftcorrect(indicator_col="5lagx",
                         uncorrected_col="5lagy_dc",
                          corrected_col="5lagy_dc_pc",
                         use_expr=pl.col("energy_5lagy_dc").is_between(90_000,110_000))
            .rough_cal_combinatoric(line_names, 
                                         uncalibrated_col="5lagy_dc_pc",
                                         calibrated_col="energy_5lagy_dc_pc",
                                    ph_smoothing_fwhm=50 # choose something close to the width of the lines in the uncalibrated plot, and small compared to the spacings
                                         )            
               )
    data3 = data2.map(analysis_step2, allow_throw=True)
    return (data3,)


@app.cell
def _(data3, mass2, np):
    data3.ch0.plot_hist("energy_5lagy_dc", np.arange(0, 150_000, 20))
    mass2.show()
    return


@app.cell
def _(data3, mo):
    chs = list(data3.channels.keys())
    dropdown_ch = mo.ui.dropdown({str(k): k for k in chs}, value=str(chs[0]), label="ch")
    steps = data3.ch0.steps
    steps[0].description
    steps_d = {f"{i} {steps[i].description}": i for i in range(len(steps))}
    dropdown_step = mo.ui.dropdown(steps_d, value=list(steps_d.keys())[-1], label="step")
    return dropdown_ch, dropdown_step


@app.cell
def _(data3, dropdown_ch, dropdown_step, mass2, mo):
    _ch = data3.channels[dropdown_ch.value]
    _ch.step_plot(dropdown_step.value)
    mo.vstack([dropdown_ch, dropdown_step, mass2.show()])
    return


@app.cell
def _(data3, dropdown_ch, mass2, plt):
    data3.channels[dropdown_ch.value].plot_scatter("5lagx", "energy_5lagy_dc_pc")
    plt.grid()
    plt.ylim(96000, 104000)
    plt.xlim(-1.5, 1)
    mass2.show()
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    # Fits to determine energy resolution

    We fine 60 eV for Ch 2 and 76 eV for Ch5, pretty close to Dan B.'s results.
    """
    )
    return


@app.cell
def _(data3, dropdown_ch, mass2):
    _result = data3.channels[dropdown_ch.value].linefit(97000, 
                                            col="energy_5lagy_dc_pc",   
                                            dlo=200, 
                                            dhi=200,
                                            binsize=15)
    _result.plotm()
    mass2.show()
    return


if __name__ == "__main__":
    app.run()
