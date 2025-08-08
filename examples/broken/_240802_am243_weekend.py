import marimo

__generated_with = "0.14.16"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import moss
    import numpy as np
    import pylab as plt
    from pathlib import Path
    import polars as pl
    import moss.mass_add_lines_truebq
    import lmfit
    return Path, lmfit, mo, moss, np, pl, plt


@app.cell
def _(mo):
    mo.md("### Demo of steps--won't work if you aren't user https://github.com/ggggggggg")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # set key parameters
    Changing any of these will re-evaluate some or all of notebook, so it can be slow. But this is how you tweak stuff.
    I moved most of the numerical parameters here for ease of seeing what you may want to tweak.
    """
    )
    return


@app.cell
def _(Path, np):
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
    npost = 700
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
def _():
    max_frontload = 0.04
    min_last_minus_first = -1000
    return max_frontload, min_last_minus_first


@app.cell
def _(mo):
    phi0_dac_units = 1880
    mo.md("###define phi0 in arb units for pretrig mean correction")
    return (phi0_dac_units,)


@app.cell
def _(
    bin_path,
    energy_of_highest_peak_ev,
    max_frontload,
    min_frames_from_last,
    min_frames_until_next,
    min_last_minus_first,
    mo,
    npost,
    npre,
    phi0_dac_units,
    threshold,
    trigger_filter,
):
    mo.md(
        f"""
    # print key parameters
    this is a workaround for not having a great pdf output yet that includes code
    we just list all the hand defined paramters manually and print them

    * {bin_path=}
    * {trigger_filter=}
    * {threshold=}
    * {min_frames_until_next=}
    * {min_frames_from_last=}
    * {energy_of_highest_peak_ev=}
    * {npre=}
    * {npost=}
    * {max_frontload=}
    * {min_last_minus_first=}
    * {phi0_dac_units=}
    """
    )
    return


@app.cell
def _(bin_path, mo, moss, threshold, trigger_filter):
    bin = moss.TrueBqBin.load(bin_path)
    trigger_result = bin.trigger(trigger_filter, threshold, limit_hours=200)
    # I notice that my memory usage increases significaintly when running bin.trigger
    # the function is written to only load a small amount of data at a time from a large mmap'd file
    # I believe the memory fills due to how the OS optimizes the mmap access.... it's better to use as much ram as available
    # so as we iterate through the large file, the OS just doesn't release the memory because it doesn't need to
    # it should be able to release that memory as needed to keep the computer running well
    trigger_result.plot(decimate=10, n_limit=100000, offset=0, x_axis_time_s=True)
    mo.vstack([mo.md("#triggering check plot"), moss.show()])
    return (trigger_result,)


@app.cell
def _():
    # long_noise = trigger_result.get_noise(
    #     n_dead_samples_after_pulse_trigger=100000, n_record_samples=500000
    # )
    # long_noise.spectrum().plot()
    # mo.vstack([mo.md("#noise plot with long records for low frequency response"),moss.show()])
    return


@app.cell
def _(mo, npost, npre, trigger_result):
    ch = trigger_result.to_channel_mmap(
        noise_n_dead_samples_after_pulse_trigger=100000,
        npre=npre,
        npost=npost,
        invert=True,
    )
    mo.vstack([mo.md("""#call triggering with cache
    then show initial dataframe
    results in `ch` the initial Channel object"""), ch.df.limit(10)])
    return (ch,)


@app.cell
def _(
    ch,
    energy_of_highest_peak_ev,
    min_frames_from_last,
    min_frames_until_next,
    mo,
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
    ch2 = ch2.with_columns(ch2.df.select(pretrig_mean_orig=pl.col("pretrig_mean"),
                           pretrig_mean=pl.col("pretrig_mean") % phi0_dac_units))
    mo.md("""#first analysis steps
    1. summarize
    2. optimal filter
    3. rough calibration (just largest peak)
    4.calculate frames_until_next and frames_from_last
    results in `ch2` the 2nd update of the Channel object""")
    return (ch2,)


@app.cell
def _(ch2, mo, moss, np, npre, pl):
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
        a, b = npre, npre+20
        pulse_pt_sub = pulse - np.mean(pulse[:npre])
        area_front = np.sum(pulse_pt_sub[a:b])
        area_rest = np.sum(pulse_pt_sub[b:])
        return area_front/area_rest

    def last_minus_first(pulse):
        return pulse[-1]-pulse[0]
    ch3 = ch3.with_columns(pl.Series("frontload", [frontload(pulses[i, :]) for i in range(pulses.shape[0])]))
    ch3 = ch3.with_columns(ch3.df.select(last_minus_first=pl.col("pulse").arr.last()-pl.col("pulse").arr.first()))
    mo.md("""#second analysis steps
    1. calculate an average pulse to plot (different than one used in filtering it seems)
    2. calculate residual_rms
    3. calculate frontloat
    results in `ch3`, the 3rd update of the Channel""")
    return avg_pulse, ch3


@app.cell
def _(ch3, max_frontload, mo, moss, np, plt):
    ch3.plot_hist("frontload", np.linspace(0, 2, 501))
    plt.axvline(max_frontload, color="k", label="max frontload")
    plt.yscale("log")
    plt.legend()
    mo.vstack([mo.md("#frontload histogram (check for good frontload choice)"), moss.show()])
    return


@app.cell
def _(
    ch3,
    max_frontload,
    min_frames_from_last,
    min_frames_until_next,
    min_last_minus_first,
    mo,
    np,
    pl,
):
    cat_cond = {
        "first_and_last": True,
        "clean": pl.all_horizontal(pl.all().is_not_null()),
        "large_residual_rms": pl.col("residual_rms") > 12,
        "spikey": pl.col("frontload") > max_frontload,
        "unlock": pl.col("last_minus_first") < min_last_minus_first,
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
    mo.md("""#categorization
    assign pulses to categories based on things we calculated about the pulse
    this isn't perfect, it's just the best I've done""")
    return cat_cond, ch4


@app.cell
def _(ch, mo, moss, plt, pulses_with_subtracted_pretrigger_mean):
    plt.plot(pulses_with_subtracted_pretrigger_mean(ch.df["pulse"][:20].to_numpy().T))
    plt.title("first 20 pulses")
    plt.xlabel("framecount")
    plt.ylabel("signal (arb)")
    mo.vstack([mo.md("# first 20 pulses plot"), moss.show()])
    return


@app.cell
def _(avg_pulse, mo, moss, plt):
    plt.plot(avg_pulse)
    plt.xlabel("framecount")
    plt.ylabel("signal (arb)")
    mo.vstack([mo.md("#average pulse plot"), moss.show()])
    return


@app.cell
def _(ch2, mo, moss, np, plt):
    ch2.plot_hist("energy_5lagy", np.arange(0, 6500000, 2000))
    plt.yscale("log")
    mo.vstack([mo.md("## rough histogram plot without any categorization"), moss.show()])
    return


@app.cell
def _(ch4, mo, moss):
    ch4.plot_scatter("5lagx", "energy_5lagy", color_col="category")
    mo.vstack([mo.md("## scatter 5lagx (subsample arrival time) vs energy\ntypically I zoom in on these to look for correlations to indicate resolution could be improved with a correction, and also to build some intuition about the dataset."), moss.show()])
    return


@app.cell
def _(ch4, mo, moss):
    ch4.plot_scatter("frontload", "energy_5lagy", color_col="category")
    mo.vstack([mo.md("#scatter energy vs frontload (spikeyness)\ntypically I zoom in on these to look for correlations to indicate resolution could be improved with a correction, and also to build some intuition about the dataset."), moss.show()])
    return


@app.cell
def _(ch3, moss):
    ch3.plot_scatter("pretrig_mean", "energy_5lagy", color_col="residual_rms_range")
    moss.show()
    return


@app.cell
def _(mo, np):
    def pulses_with_subtracted_pretrigger_mean(pulses, n_pretrigger=10):
        """
        Subtract the pretrigger mean from each pulse in the 2D array.

        Parameters:
        pulses (np.ndarray): A 2D array of pulses where pulses[:, i] represents the i-th pulse.
        n_pretrigger (int): The number of samples to use for calculating the pretrigger mean.

        Returns:
        np.ndarray: A 2D array with the pretrigger mean subtracted from each pulse.
        """
        # Compute the pretrigger mean for each pulse (column-wise)
        pretrigger_means = np.mean(pulses[:n_pretrigger, :], axis=0)

        # Expand dimensions of pretrigger_means to allow broadcasting
        result = pulses - pretrigger_means[np.newaxis, :]

        return result
    mo.md("#### define a helper function")
    return (pulses_with_subtracted_pretrigger_mean,)


@app.cell
def _(ch2, mo, moss):
    ch2.plot_scatter("framecount", "pretrig_mean")
    mo.vstack([mo.md("#scatter pretrig mean (corrected) vs framecount (time)\ntypically I zoom in on these to look for correlations to indicate resolution could be improved with a correction, and also to build some intuition about the dataset."), moss.show()])
    return


@app.cell
def _(ch2, mo, moss):
    ch2.plot_scatter("framecount", "pretrig_mean_orig")
    mo.vstack([mo.md("#scatter debug pretrig_mean_orig vs framecount\ntypically I zoom in on these to look for correlations to indicate resolution could be improved with a correction, and also to build some intuition about the dataset."), moss.show()])
    return


@app.cell
def _(ch2, mo, moss):
    ch2.plot_scatter("pretrig_mean", "energy_5lagy")
    mo.vstack([mo.md("#scatter pretrig_mean rms vs energy\ntypically I zoom in on these to look for correlations to indicate resolution could be improved with a correction, and also to build some intuition about the dataset."), moss.show()])
    return


@app.cell
def _(ch4, mo, moss):
    ch4.plot_scatter("residual_rms", "energy_5lagy", color_col="category")
    mo.vstack([mo.md("#scatter debug plot rms vs energy\ntypically I zoom in on these to look for correlations to indicate resolution could be improved with a correction, and also to build some intuition about the dataset."), moss.show()])
    return


@app.cell
def _(cat_cond, mo):
    dropdown_pulse_category = mo.ui.dropdown(
        cat_cond.keys(),
        value=list(cat_cond.keys())[0],
        label="choose pulse category to plot",
    )
    mo.md("###boiler plate for pulse category viewer")
    return (dropdown_pulse_category,)


@app.cell
def _(ch4, mo, moss, pl, plt, pulses_with_subtracted_pretrigger_mean):
    _elo, _ehi = 5.5e6, 6e6
    _pulses = (
        ch4.df.lazy()
        .filter(pl.col("category") == "clean", pl.col("energy_5lagy").is_between(_elo, _ehi))
        .limit(50)
        .select("pulse")
        .collect()
        .to_series()
        .to_numpy()
    )
    plt.plot(pulses_with_subtracted_pretrigger_mean(_pulses.T, 30))
    plt.suptitle(f"pulses which are between {_elo:g} and {_ehi:g} eV and clean")
    plt.xlabel("sample number")
    plt.ylabel("signal (dac units)")
    mo.vstack([mo.md("#pulse in energy range viewer"), moss.show()])
    return


@app.cell
def _(
    ch4,
    dropdown_pulse_category,
    mo,
    moss,
    pl,
    plt,
    pulses_with_subtracted_pretrigger_mean,
):
    _cat = dropdown_pulse_category.value
    _pulses = (
        ch4.df.lazy()
        .filter(pl.col("category") == _cat)
        .limit(50)
        .select("pulse")
        .collect()
        .to_series()
        .to_numpy()
    )
    print(f"{_cat=} {_pulses.shape}")
    plt.plot(pulses_with_subtracted_pretrigger_mean(_pulses.T))
    plt.suptitle(_cat)
    plt.xlabel("sample number")
    plt.ylabel("signal (dac units)")
    mo.vstack([mo.md("#pulse type viewer\nwith pretrigger mean subtracted"), dropdown_pulse_category, moss.show()])
    return


@app.cell
def _(ch4, mo):
    stepplotter = (
        ch4.mo_stepplots()
    )  # can't show stepplotter in same cell it's defined, just how marimo works
    mo.md("###boiler plate for the step plot cell")
    return (stepplotter,)


@app.cell
def _(mo, stepplotter):
    mo.vstack([mo.md("#step plots"), stepplotter.show()])
    return


@app.cell
def _(ch4, min_frames_from_last, mo, moss, np, pl, plt):
    # live time clean hist
    def livetime_clean_energies(cat="clean"):
        df = (
            ch4.df.lazy()
            .filter(pl.col("category").is_in([cat]))
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
    mo.vstack([mo.md("#coadded spectrum with livetime\nbeware unlocks not neccesarily fully accounted for"), moss.show()])
    return bin_centers, bin_edges, bin_size, livetime_clean_energies


@app.cell
def _(
    bin_centers,
    bin_edges,
    bin_size,
    cat_cond,
    livetime_clean_energies,
    mo,
    moss,
    plt,
):
    for cat in list(cat_cond.keys())[::-1]:
        _energies, _ = livetime_clean_energies(cat)
        _, _counts = moss.misc.hist_of_series(_energies, bin_edges=bin_edges)
        plt.plot(bin_centers, _counts, label=cat)
    # plt.plot(bin_centers, counts)

    plt.xlabel("energy / eV")
    plt.ylabel(f"counts / {bin_size:.1f} eV bin")
    plt.yscale("log")
    plt.legend()

    mo.vstack([mo.md("# histogram by category plot"), moss.show()])
    return


@app.cell
def _(ch4, lmfit, mo, moss, pl):
    _result = ch4.linefit(
        "Am241Q",
        "energy_5lagy",
        use_expr=pl.col("category") == "clean",
        dlo=1e5,
        dhi=2e5,
        binsize=1000,
        params_update=lmfit.create_params(fwhm={"value": 4000, "min": 500}, dph_de={
                                          "vary": False, "min": 0.9, "max": 1.1, "value": 1}, peak_ph=5.64e6)
    )
    _result.plotm()
    mo.vstack([mo.md("#Am241 fit"), moss.show()])
    return


@app.cell
def _(ch4, lmfit, mo, moss, pl):
    _result = ch4.linefit(
        "Am243Q",
        "energy_5lagy",
        use_expr=pl.col("category") == "clean",
        dlo=1e5,
        dhi=2e5,
        binsize=1000,
        params_update=lmfit.create_params(fwhm={"value": 4000, "min": 500}, dph_de={"vary": False, "min": 0.9, "max": 1.1, "value": 1})
    )
    _result.plotm()

    mo.vstack([mo.md("#Am243 fit"), moss.show()])
    return


@app.cell
def _(ch4, estimate_dataframe_size, human_readable_size, mo, pl):
    import os
    from datetime import datetime

    # the dataframe without pulse data aka list mode data
    _df = ch4.df.select(pl.exclude("pulse"))
    # Get the current date and time
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M")
    # Generate the filename
    _filename = f"{ch4.header.description}_{current_time}.parquet"
    _df.write_parquet(_filename)
    # Function to convert file size to human-readable format

    # Check the file size
    _file_size = os.path.getsize(_filename)
    # Convert to human-readable format
    _readable_size = human_readable_size(_file_size)
    # get full path
    _absolute_path = os.path.abspath(_filename)
    mo.vstack([mo.md(f"""#final dataframe and save list mode data
    {_readable_size} data saved to {_absolute_path}

    naivley it should should be {estimate_dataframe_size(_df)}, but may be smaller with compression"""), _df])
    return


@app.cell(hide_code=True)
def _(mo, pl):
    def human_readable_size(size_in_bytes):
        for unit in ['B', 'kB', 'MB', 'GB', 'TB']:
            if size_in_bytes < 1024:
                return f"{size_in_bytes:.2f} {unit}"
            size_in_bytes /= 1024
        s = f"{size_in_bytes:.2f} PB"  # In case it's enormous!
        print(s)
        return s

    def estimate_dataframe_size(df: pl.DataFrame) -> str:
        """
        Estimate the memory/storage size of a Polars DataFrame containing only
        primitive, enum, and categorical columns.

        Parameters:
        - df: A Polars DataFrame.

        Returns:
        - A human-readable string representing the estimated size (e.g., "1.23 MB").
        """
        # Bytes per type mapping
        type_sizes = {
            pl.Float64: 8,
            pl.Float32: 4,
            pl.Int64: 8,
            pl.Int32: 4,
            pl.Int16: 2,
            pl.Int8: 1,
            pl.UInt64: 8,
            pl.UInt32: 4,
            pl.UInt16: 2,
            pl.UInt8: 1,
        }

        total_bytes = 0

        for column in df.columns:
            dtype = df[column].to_physical().dtype
            num_elements = len(df[column])

            if dtype in type_sizes:
                total_bytes += num_elements * type_sizes[dtype]
            else:
                raise ValueError(f"Unsupported dtype: {dtype}")

        # Convert to human-readable format

        return human_readable_size(total_bytes)

    def write_and_measure_parquet(df: pl.DataFrame):
        """
        Write a Polars DataFrame to a temporary folder as a Parquet file,
        measure its size, and clean up afterward.

        Parameters:
        - df: A Polars DataFrame.

        Returns:
        - The size of the Parquet file in a human-readable format.
        """
        import tempfile
        import os
        # Use a temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Define the temporary file path
            parquet_path = os.path.join(temp_dir, "dummy_data.parquet")

            # Write the DataFrame to Parquet
            df.write_parquet(parquet_path)

            # Measure the file size
            file_size_bytes = os.path.getsize(parquet_path)

            # Return the human-readable size
            readable_size = human_readable_size(file_size_bytes)
            print(f"Parquet file size: {readable_size}")
            return readable_size
    mo.md("helper code for checking file size is reasonable")
    return estimate_dataframe_size, human_readable_size


@app.cell
def _(ch4, mo):
    mo.vstack([mo.md("#final dataframe viewer"), ch4.df])
    return


if __name__ == "__main__":
    app.run()
