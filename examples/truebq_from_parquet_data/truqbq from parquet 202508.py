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
    import mass2.core.mass_add_lines_truebq
    import lmfit
    import pulsedata
    return Path, lmfit, mass2, mo, np, pl, plt, pulsedata


@app.cell
def _(mass2, pl, pulsedata):
    def from_parquet(filename) -> mass2.Channel:
        df = pl.read_parquet(filename)
        # with just the parquet, we don't know all this other info
        # lets just fill it in with None where we can
        # and see what we can do
        ch = mass2.Channel(
            df,
            header=mass2.ChannelHeader(
                description=filename,
                ch_num=1,
                frametime_s=1e-5,
                df=None,
                n_presamples=None,
                n_samples=None,
            ),
            npulses=len(df),
            good_expr=pl.col("category") == "clean",
        )
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
    bin_centers, counts5um = ch5um.plot_hists(
        "energy_5lagy_dc", bin_edges, group_by_col="concat_state"
    )
    plt.yscale("log")
    plt.title("5um foil, large tail to Pu239 line from alpha escape")
    mass2.show()
    return bin_centers, bin_edges, counts5um


@app.cell
def _(bin_edges, ch20um, mass2, plt):
    _, counts20um = ch20um.plot_hists(
        "energy_5lagy", bin_edges, group_by_col="concat_state"
    )
    plt.yscale("log")
    plt.title("20um foil, few alpha escape, clean Pu239 line")
    mass2.show()
    return (counts20um,)


@app.cell
def _(bin_edges, ch20um_blank, mass2, plt):
    _, counts20um_blank = ch20um_blank.plot_hists(
        "energy_5lagy", bin_edges, group_by_col="concat_state"
    )
    plt.yscale("log")
    plt.title("20um BLANK, no Pu239 line")
    mass2.show()
    return (counts20um_blank,)


@app.cell
def _(mass2):
    model1 = mass2.calibration.algorithms.get_model(5.244e6, has_tails=True)
    model2 = mass2.calibration.algorithms.get_model(5.254e6).spect.model(
        prefix="B", has_linear_background=False, has_tails=True
    )
    model = model1 + model2
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
def _(bin_centers, counts20um, counts20um_blank, counts5um, np, pl):
    # Define dtype and create structured array
    dtype = [
        ("bin_centers", float),
        ("counts20um_noCo", int),
        ("counts20um_withCo", int),
        ("counts20um_blank_noCo", int),
        ("counts20um_blank_withCo", int),
        ("counts5um_noCo", int),
        ("counts5um_withCo", int),
    ]

    data = np.zeros(len(bin_centers), dtype=dtype)
    data["bin_centers"] = bin_centers
    data["counts20um_noCo"] = counts20um[0].astype(int)
    data["counts20um_withCo"] = counts20um[1].astype(int)
    data["counts20um_blank_noCo"] = counts20um_blank[0].astype(int)
    data["counts20um_blank_withCo"] = counts20um_blank[1].astype(int)
    data["counts5um_noCo"] = counts5um[0].astype(int)
    data["counts5um_withCo"] = counts5um[1].astype(int)

    # Save structured array as .npy
    np.save("202508truebq_Pu239_5um_20um_20umblank_all_data.npy", data)

    # Convert structured array to Polars DataFrame directly
    df = pl.from_numpy(data)

    # Save to CSV
    df.write_csv("202508truebq_Pu239_5um_20um_20umblank_all_data.csv")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    # TODO
    1. Try the GEANT4 fitting with Ryan's basis funcs.
    2. Live time hist
    3. All data category accounting + example plot
    """
    )
    return


@app.cell
def _(Path, pl):
    def _():
        from pathlib import Path
        import polars as pl

        folder = Path(
            r"\\elwood.nist.gov\682\internal\PL846_04\TrueBq IMS\Geant4\TrueBq fit\Pu-239 Exp1\Geant4_300eV"
        )
        files = sorted(folder.glob("*.out"))

        data = {}
        energy_ref = None

        for f in files:
            df = pl.read_csv(
                f,
                separator="\t",
                skip_rows=14,
                has_header=False,  # no header, treat all as data
                new_columns=["E_MeV", f.stem],  # assign names directly
                truncate_ragged_lines=True,
            )

            e = df["E_MeV"]
            c = df[f.stem]

            if energy_ref is None:
                energy_ref = e
                data["E_MeV"] = energy_ref
            else:
                if not e.equals(energy_ref):
                    raise ValueError(f"Energy axis mismatch in {f}")

            data[f.stem] = c

        df_all = pl.DataFrame(data)

        out_path = folder / "merged_results.parquet"
        df_all.write_parquet(out_path, compression="zstd")
        return df_all


    def col_name(element, thickness):
        return f"{element}_{thickness}_300eV_h1_0"


    # df_all = _()
    merged_results_path = (
        Path(
            r"\\elwood.nist.gov\682\internal\PL846_04\TrueBq IMS\Geant4\TrueBq fit\Pu-239 Exp1\Geant4_300eV"
        )
        / "merged_results.parquet"
    )
    df_all = pl.read_parquet(merged_results_path)
    print(df_all)
    return (df_all,)


@app.cell
def _(df_all):
    print(df_all.columns)
    return


@app.cell
def _(df_all, pl):
    def get_isotope_data(isotope: str, df: pl.DataFrame = df_all):
        iso = isotope[:2] + "-" + isotope[2:]
        cols = [c for c in df.columns if c.startswith(iso)]
        thk = [int(c.split("_")[1].replace("um", "")) for c in cols]
        cols, thk = zip(*sorted(zip(cols, thk), key=lambda x: x[1]))
        return list(cols), list(thk)


    def get_available_isotopes(df: pl.DataFrame = df_all):
        return sorted({c.split("_")[0].replace("-", "") for c in df.columns} - {"E"})


    def iterate_isotopes_at_thickness(df: pl.DataFrame = df_all, thickness: int = 30):
        for iso in get_available_isotopes(df):
            col = f"{iso[:2]}-{iso[2:]}_{thickness}um_300eV_h1_0"
            yield iso, col


    get_available_isotopes()
    return (
        get_available_isotopes,
        get_isotope_data,
        iterate_isotopes_at_thickness,
    )


@app.cell
def _(df_all, get_available_isotopes, get_isotope_data, pl, plt):
    def plot_isotope(isotope: str, df: pl.DataFrame = df_all):
        cols, thk = get_isotope_data(isotope, df)
        x = df["E_MeV"].to_numpy()
        plt.figure()
        for c, t in zip(cols, thk):
            plt.plot(x, df[c].to_numpy(), label=f"{t}")
        plt.yscale("log")
        plt.xlabel("Energy [MeV]")
        plt.ylabel("Intensity (a.u.)")
        plt.title(isotope)
        plt.legend()
        plt.show()


    # plot Am241
    for isotope_str in get_available_isotopes(df_all):
        plot_isotope(isotope_str, df_all)

    plt.show()
    return


@app.cell
def _(ch20um, df_all, iterate_isotopes_at_thickness, mass2, np, pl, plt):
    bin_edges_full = np.arange(0, 6000000, 1000)
    bin_centers_full, counts_full = ch20um.hist(
        "energy_5lagy", bin_edges_full, use_expr=pl.col("concat_state") == 0
    )
    plt.figure()
    for _isotope_str, col in iterate_isotopes_at_thickness(thickness=30):
        count_theory = df_all[col].to_numpy()
        plt.plot(
            df_all["E_MeV"].to_numpy(),
            count_theory / np.amax(count_theory),
            label=_isotope_str,
        )
    plt.plot(bin_centers_full / 1e6, counts_full / np.amax(counts_full), label="data", lw=2)
    plt.legend()
    plt.xlabel("energy (MeV)")
    plt.ylabel("counts")
    mass2.show()
    return


@app.cell
def _(ch20um2, df_all, iterate_isotopes_at_thickness, mass2, np, pl, plt):
    def _():
        bin_edges_full = np.arange(0, 6000000, 1000)
        bin_centers_full, counts_full = ch20um2.hist(
            "energy2_5lagy", bin_edges_full, use_expr=pl.col("concat_state") == 0
        )
        plt.figure()
        for _isotope_str, col in iterate_isotopes_at_thickness(thickness=30):
            count_theory = df_all[col].to_numpy()
            plt.plot(
                df_all["E_MeV"].to_numpy(),
                count_theory / np.amax(count_theory),
                label=_isotope_str,
            )
        plt.plot(bin_centers_full / 1e6, counts_full / np.amax(counts_full), label="data", lw=2)
        plt.legend()
        plt.xlabel("energy (MeV)")
        plt.ylabel("counts")
        return mass2.show()


    _()
    return


@app.cell
def _(ch20um, np, pl):
    # recal_energy_5lagy = np.array([5.24599, 5.58497, 5.62697])*1e6
    # recal_energy_theory = np.array([5.24396, 5.59299, 5.5637])*1e6
    recal_energy_5lagy = np.array([5.24599, 5.6268])*1e6
    recal_energy_theory = np.array([5.24396, 5.637])*1e6
    from numpy.polynomial import Polynomial
    pfit = np.polyfit(recal_energy_5lagy, recal_energy_theory/recal_energy_5lagy,deg=1)
    ch20um.rough_cal(uncalibrated_col="energy_5lagy",line_names=recal_energy_theory)
    energy_5lagy = ch20um.df["energy_5lagy"].to_numpy()
    energy2_5lagy = energy_5lagy*np.polyval(pfit, energy_5lagy)
    ch20um2 = ch20um.with_columns(pl.DataFrame({"energy2_5lagy":energy2_5lagy}))
    return ch20um2, pfit, recal_energy_5lagy, recal_energy_theory


@app.cell
def _(np, pfit, plt, recal_energy_5lagy, recal_energy_theory):
    z = np.arange(10,6e6,10000)
    plt.plot(z, np.polyval(pfit, z))
    plt.plot(recal_energy_5lagy, recal_energy_theory/recal_energy_5lagy,".")
    return


@app.cell
def _(ch20um2):
    ch20um2.df
    return


@app.cell
def _(mass2, np):
    mass2.calibration.line_models._smear_exponential_tail
    from numpy.typing import ArrayLike, NDArray

    def yolo_smear_exponential_tail(
        x: ArrayLike,
        y: ArrayLike,
        P_resolution: float,
        P_tailfrac: float,
        P_tailtau: float,
        P_tailshare_hi: float = 0.0,
        P_tailtau_hi: float = 1,
    ) -> NDArray:
        """Smear a clean spectrum with exponential low- and/or high-energy tails.

        Parameters
        ----------
        x : ArrayLike
            Grid for the spectrum (monotonic, evenly spaced).
        y : ArrayLike
            Spectrum values at x. Assume to be zero outside x.
        P_resolution : float
            Instrument resolution (FWHM).
        P_tailfrac : float
            Fraction of events in tails.
        P_tailtau : float
            Low-energy tail exponential scale length (same units as x).
        P_tailshare_hi : float, optional
            Fraction of tail events in the high-E tail.
        P_tailtau_hi : float, optional
            High-energy tail exponential scale length.

        Returns
        -------
        NDArray
            The smeared spectrum on the same x grid.
        """
        x = np.asarray(x)
        y = np.asarray(y)

        if P_tailfrac <= 1e-6:
            return y

        # Padding to avoid wrap-around in FFT
        energy_step = x[1] - x[0]
        nlow = int((P_resolution + max(P_tailtau * 6, P_tailtau_hi)) / energy_step + 0.5)
        nhi = int((P_resolution + max(P_tailtau, P_tailtau_hi * 6)) / energy_step + 0.5)
        nlow = min(10 * len(x), nlow)
        nhi = min(10 * len(x), nhi)
        x_wide = np.arange(-nlow, nhi + len(x)) * energy_step + x[0]
        if len(x_wide) > 100000:
            msg = f"you're trying to FFT data of length {len(x_wide)} (bad fit param?)"
            raise ValueError(msg)

        # Interpolate y onto padded grid, fill with 0 outside
        y_wide = np.interp(x_wide, x, y, left=0.0, right=0.0)

        # FFT-based convolution
        freq = np.fft.rfftfreq(len(x_wide), d=energy_step)  # 1/energy
        ft = np.fft.rfft(y_wide)

        filter_effect_fourier = np.ones_like(ft) - P_tailfrac
        P_tailfrac_hi = P_tailfrac * P_tailshare_hi
        P_tailfrac_lo = P_tailfrac - P_tailfrac_hi
        if P_tailfrac_lo > 1e-6:
            filter_effect_fourier += P_tailfrac_lo / (1 - 2j * np.pi * freq * P_tailtau)
        if P_tailfrac_hi > 1e-6:
            filter_effect_fourier += P_tailfrac_hi / (1 + 2j * np.pi * freq * P_tailtau_hi)
        ft *= filter_effect_fourier
        if P_resolution > 1e-12:
            sigma = P_resolution / 2.3548
            gaussian_ft = np.exp(-0.5 * (2*np.pi*freq*sigma)**2)
            # print(f"{sigma=} {P_resolution=} {x[0]=}")
            ft *= gaussian_ft
        smoothspectrum = np.fft.irfft(ft, n=len(x_wide))

        # Avoid negatives from numerical artifacts
        smoothspectrum[smoothspectrum < 0] = 0
        return smoothspectrum[nlow : nlow + len(x)]


    return (yolo_smear_exponential_tail,)


@app.cell
def _(ch20um2, df_all, np, pl):
    x_fit_lo, x_fit_hi = 5.24599*0.95, 5.6268*1.05
    x_all = df_all["E_MeV"].to_numpy()
    x_fit_inds = np.searchsorted(x_all, [x_fit_lo, x_fit_hi])
    x_edges_fit = df_all["E_MeV"].to_numpy()[x_fit_inds[0]:x_fit_inds[1]]
    counts_fit = x_centers_fit, counts_fit = ch20um2.hist(
            "energy2_5lagy", x_edges_fit*1e6, use_expr=pl.col("concat_state") == 0
        )
    return counts_fit, x_centers_fit, x_fit_inds


@app.cell
def _(
    df_all,
    lmfit,
    np,
    x_centers_fit,
    x_fit_inds,
    yolo_smear_exponential_tail,
):

    # assumes _smear_exponential_tail (the x,y-only version) is already defined

    ISOTOPES = ["Am241", "Pu238", "Pu239", "Pu240", "Pu241", "Pu242"]

    def isotope_model_function(
        x,
        Am241=0.1,
        Pu238=0.1,
        Pu239=0.1,           # 30 um
        Pu239_12um=0.1,      # 12 um
        Pu239_8um=0.1,       # 8 um
        Pu240=0.1,
        Pu241=0.1,
        Pu242=0.1,
        P_resolution=0.2,
        P_tailfrac=0.05,
        P_tailtau=0.3,
        P_tailshare_hi=0.0,
        P_tailtau_hi=1.0,
        x_fit_ind_lo=x_fit_inds[0],
        x_fit_ind_hi=x_fit_inds[1],
    ):
        assert np.allclose(x, x_centers_fit)
        """
        Combine isotope spectra (with separate Pu-239 thickness components)
        and apply smearing.
        """

        # start with zeros
        y_total = np.zeros_like(x, dtype=float)

        # map isotope -> (amplitude, key(s))
        spec_map = {
            "Am-241": [(Am241, "Am-241_30um_300eV_h1_0")],
            "Pu-238": [(Pu238, "Pu-238_30um_300eV_h1_0")],
            "Pu-239": [
                (Pu239,      "Pu-239_30um_300eV_h1_0"),
                (Pu239_12um, "Pu-239_12um_300eV_h1_0"),
                (Pu239_8um,  "Pu-239_8um_300eV_h1_0"),
            ],
            "Pu-240": [(Pu240, "Pu-240_30um_300eV_h1_0")],
            "Pu-241": [(Pu241, "Pu-241_30um_300eV_h1_0")],
            "Pu-242": [(Pu242, "Pu-242_30um_300eV_h1_0")],
        }

        # build clean spectrum
        for iso, specs in spec_map.items():
            for amp, key in specs:
                if amp == 0:
                    continue
                y_iso = df_all[key].to_numpy()[x_fit_ind_lo:x_fit_ind_hi-1]
                y_total += amp * y_iso

        # apply smearing
        y_smeared = yolo_smear_exponential_tail(
            x, y_total,
            P_resolution=P_resolution,
            P_tailfrac=P_tailfrac,
            P_tailtau=P_tailtau,
            P_tailshare_hi=P_tailshare_hi,
            P_tailtau_hi=P_tailtau_hi,
        )
        return y_smeared


    # Wrap in lmfit.Model
    IsotopeModel = lmfit.Model(isotope_model_function, independent_vars=["x"])
    return (IsotopeModel,)


@app.cell
def _(IsotopeModel):
    params_fit = IsotopeModel.make_params()
    params_fit["Am241"].set(0.001, vary=True, min=0)
    params_fit["Pu238"].set(0.001, vary=True, min=0)
    params_fit["Pu239"].set(0.04, vary=True, min=0)       # 30um
    params_fit["Pu239_12um"].set(0.00, vary=False, min=0)  # 12um
    # params_fit["Pu239_12um"].set(0.01, vary=False, min=0)  # 12um
    params_fit["Pu239_8um"].set(0.0000, vary=False, min=0)   # 8um
    params_fit["Pu240"].set(0.001, vary=True, min=0)
    params_fit["Pu241"].set(0.001, vary=True, min=0)
    params_fit["Pu242"].set(0.001, vary=True, min=0)

    params_fit["P_resolution"].set(4000, vary=False, min=0)
    params_fit["P_tailfrac"].set(1, vary=False, min=0, max=1)
    params_fit["P_tailtau"].set(4000, vary=False, min=0)
    params_fit["P_tailshare_hi"].set(0.0, vary=False, min=0)
    params_fit["P_tailtau_hi"].set(1.0, vary=False, min=0)
    return (params_fit,)


@app.cell
def _(IsotopeModel, counts_fit, params_fit, plt, x_centers_fit):
    # Evaluate the model at initial parameter values
    y_initial = IsotopeModel.eval(
        params=params_fit,
        x=x_centers_fit
    )

    # Plot
    plt.figure(figsize=(8,5))
    plt.plot(x_centers_fit, y_initial, label="Initial model")
    plt.plot(x_centers_fit, counts_fit, label="data")
    plt.xlabel("Energy (or whatever units x has)")
    plt.ylabel("Counts / intensity")
    plt.title("Initial isotope model spectrum")
    plt.legend()
    plt.grid(True)
    plt.yscale("log")
    plt.show()
    return


@app.cell
def _(IsotopeModel, counts_fit, params_fit, x_centers_fit):
    result1 = IsotopeModel.fit(
        data=counts_fit,
        params=params_fit,
        x=x_centers_fit
    )
    return (result1,)


@app.cell
def _(result1):
    print(result1.fit_report())
    return


@app.cell
def _(mass2, plt, result1):
    result1.plot()
    plt.yscale("log")
    mass2.show()
    return


@app.cell
def _(counts_fit, mass2, plt, result1, x_centers_fit):
    def plot_fit_and_components(result, x, data, isotope_names):
        """
        Plot the data, best-fit model, and individual isotope contributions.

        Parameters
        ----------
        result : lmfit.model.ModelResult
            The result object from Model.fit().
        x : array-like
            X-axis values (energy bins, etc).
        data : array-like
            The observed spectrum (counts).
        isotope_names : list of str
            Isotopes included in the model, e.g. ["Am241", "Pu238", ...].
            Pu239 will automatically expand into [Pu239, Pu239_12um, Pu239_8um].
        """
        # Evaluate the full best-fit model
        y_best = result.eval(x=x)

        plt.figure(figsize=(8, 5))
        plt.plot(x, data, "k.", label="Data")
        plt.plot(x, y_best, "r-", label="Best fit")

        # Expand isotope list so Pu239 has thicknesses
        expanded_isotopes = []
        for iso in isotope_names:
            if iso == "Pu239":
                expanded_isotopes.extend(["Pu239", "Pu239_12um", "Pu239_8um"])
            else:
                expanded_isotopes.append(iso)

        # Loop through isotopes/thicknesses
        for iso in expanded_isotopes:
            params_iso = result.params.copy()
            # Zero out all isotope amplitudes except the one we care about
            for other in expanded_isotopes:
                if other != iso:
                    params_iso[other].set(value=0)
            # Evaluate contribution
            y_iso = result.model.eval(params=params_iso, x=x)
            plt.plot(x, y_iso, "--", label=f"{iso} contribution")

        plt.xlabel("Energy (MeV)")
        plt.ylabel("Counts")
        plt.legend()
        plt.tight_layout()
        plt.yscale("log")
        plt.ylim(1e-3, plt.ylim()[1])
        return mass2.show()


    isotopes = ["Am241", "Pu238", "Pu239", "Pu240", "Pu241", "Pu242"]

    plot_fit_and_components(result1, x_centers_fit, counts_fit, isotopes)
    return


if __name__ == "__main__":
    app.run()
