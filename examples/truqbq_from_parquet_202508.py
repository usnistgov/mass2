import marimo

__generated_with = "0.15.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import mass2
    import numpy as np
    import pylab as plt
    from pathlib import Path
    import polars as pl
    import tempfile
    import os
    import mass2.core.mass_add_lines_truebq
    import pulsedata
    return mass2, np, os, pl, plt, pulsedata, tempfile


@app.cell
def _(mass2, pl, pulsedata):
    def from_parquet(filename) -> mass2.Channel:
        df = pl.read_parquet(filename)
        # with just the parquet, we don't know all this other info
        # lets just fill it in with None where we can
        # and see what we can do
        ch = mass2.Channel(df,
                           header=mass2.ChannelHeader(
                               description=filename,
                               data_source=filename,
                               ch_num=1,
                               frametime_s=1e-5, df=None, n_presamples=None, n_samples=None),
                           npulses=len(df),
                           good_expr=pl.col("category") == "clean")
        return ch
    ch5um = from_parquet(pulsedata.parquet["truebq_202508_5um_Pu239.parquet"])
    ch20um = from_parquet(pulsedata.parquet["truebq_202508_20um_Pu239.parquet"])
    ch20um_blank = from_parquet(pulsedata.parquet["truebq_202508_20um_BLANK.parquet"])
    return ch20um, ch20um_blank, ch5um


@app.cell
def _(ch5um):
    ch = ch5um
    ch.good_df()

    return


@app.cell
def _(ch5um, mass2, np, plt):
    bin_edges = np.arange(0, 10e6, 500)
    bin_centers, counts5um = ch5um.plot_hists("energy_5lagy_dc", bin_edges, group_by_col="concat_state")
    plt.yscale("log")
    plt.title("5um foil, large tail to Pu239 line from alpha escape")
    mass2.show()
    return bin_centers, bin_edges, counts5um


@app.cell
def _(bin_edges, ch20um, mass2, plt):
    _, counts20um = ch20um.plot_hists("energy_5lagy", bin_edges, group_by_col="concat_state")
    plt.yscale("log")
    plt.title("20um foil, few alpha escape, clean Pu239 line")
    mass2.show()

    return (counts20um,)


@app.cell
def _(bin_edges, ch20um_blank, mass2, plt):
    _, counts20um_blank = ch20um_blank.plot_hists("energy_5lagy", bin_edges, group_by_col="concat_state")
    plt.yscale("log")
    plt.title("20um BLANK, no Pu239 line")
    mass2.show()
    return (counts20um_blank,)


@app.cell
def _(mass2):
    model1 = mass2.calibration.algorithms.get_model(5.244e6, has_tails=True)
    model2 = mass2.calibration.algorithms.get_model(5.254e6).spect.model(prefix="B", has_linear_background=False, has_tails=True)
    model = model1+model2
    params = model.make_params()
    params["Bintegral"].set(1000)
    params["integral"].set(10000)
    params["fwhm"].set(4000, max=6000)

    params["Bfwhm"].set(expr="fwhm")
    params["Btail_frac"].set(expr="tail_frac")
    params["Btail_tau"].set(expr="tail_tau")
    params["Bpeak_ph"].set(expr="peak_ph+Bshift")
    params.add("Bshift", 0.011e6, min=0.01e6, vary=True)
    params["tail_frac"].set(vary=True)
    params["tail_tau"].set(vary=True)
    params["Bdph_de"].set(1, vary=False)
    params["dph_de"].set(1, vary=False)
    return model, params


@app.cell
def _(ch20um, mass2, model, np, params):
    bin_edges_fit = np.arange(5_210_000, 5_275_000, 250.0)
    bin_centers_fit, counts_noco = ch20um.hist("energy_5lagy", bin_edges_fit)
    result = model.fit(counts_noco, params, bin_centers=bin_centers_fit)
    result.plotm()
    mass2.show()
    return


@app.cell
def _(
    bin_centers,
    counts20um,
    counts20um_blank,
    counts5um,
    np,
    os,
    pl,
    tempfile,
):
    # Define dtype and create structured array
    dtype = [
        ('bin_centers', float),
        ('counts20um_noCo', int),
        ('counts20um_withCo', int),
        ('counts20um_blank_noCo', int),
        ('counts20um_blank_withCo', int),
        ('counts5um_noCo', int),
        ('counts5um_withCo', int),
    ]

    data = np.zeros(len(bin_centers), dtype=dtype)
    data['bin_centers'] = bin_centers
    data['counts20um_noCo'] = counts20um["0"].astype(int)
    data['counts20um_withCo'] = counts20um["1"].astype(int)
    data['counts20um_blank_noCo'] = counts20um_blank["0"].astype(int)
    data['counts20um_blank_withCo'] = counts20um_blank["1"].astype(int)
    data['counts5um_noCo'] = counts5um["0"].astype(int)
    data['counts5um_withCo'] = counts5um["1"].astype(int)

    # Replace with some local directory if you want to check out these files.
    # But for testing purposes, we want to stash them somewhere temporary:
    with tempfile.TemporaryDirectory(prefix="trueBq_demo_notebook_") as output_dir:
        # Uncomment the following line to write somewhere specific, instead of using the temporary dir
        # output_dir = "."
        print(f"Writing output CSV and npy files to {output_dir=}")

        # Save structured array as .npy
        npy_path = os.path.join(output_dir, '202508truebq_Pu239_5um_20um_20umblank_all_data.npy')
        np.save(npy_path, data)

        # Convert structured array to Polars DataFrame directly
        df = pl.from_numpy(data)

        # Save to CSV
        csv_path = os.path.join(output_dir, '202508truebq_Pu239_5um_20um_20umblank_all_data.csv')
        df.write_csv(csv_path)
    return


if __name__ == "__main__":
    app.run()
