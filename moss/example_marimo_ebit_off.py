import marimo

__generated_with = "0.7.11"
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
    off_paths = moss.ljhutil.find_ljh_files(str(pulsedata.off["ebit_20240722_0006"]), ext=".off")
    off = mass.off.OffFile(off_paths[0])
    return off, off_paths


@app.cell
def __(mass, moss, pl):
    def from_off_paths(cls, off_paths):
        channels = {}
        for path in off_paths:
            ch = from_off(moss.Channel, mass.off.OffFile(path))
            channels[ch.header.ch_num] =ch
        return moss.Channels(channels, "from_off_paths")

    def from_off(cls, off):
        df = pl.from_numpy(off._mmap)
        df = df.select(pl.from_epoch("unixnano", time_unit="ns").dt.cast_time_unit("us").alias("timestamp")).with_columns(df).select(pl.exclude("unixnano"))
        header = moss.ChannelHeader(f"{off}",
                       off.header["ChannelNumberMatchingName"],
                      off.framePeriodSeconds,
        off._mmap["recordPreSamples"][0],
        off._mmap["recordSamples"][0],
        pl.DataFrame(off.header))
        ch = cls(df,
                         header)
        return ch
    return from_off, from_off_paths


@app.cell
def __(from_off_paths, moss, off_paths):
    data = from_off_paths(moss.Channels, off_paths)
    data
    return data,


@app.cell
def __(data, mo, off_paths, pathlib):
    data2 = data.with_experiment_state_by_path(pathlib.Path(off_paths[0]).parent /"20240722_run0006_experiment_state.txt")
    mo.plain(data2.channels[1].df)
    return data2,


@app.cell
def __(data2, mo):
    ch = data2.channels[1]
    mo.plain(ch.df)
    return ch,


@app.cell
def __(ch, mo, pl, plt):
    ch2 = ch.rough_cal(
            ["MnKAlpha", "MnKBeta", "CuKAlpha", "CuKBeta", "AlKAlpha", "ClKAlpha", "KKAlpha","ScKAlpha","MgKAlpha"],
            uncalibrated_col="filtValue",
            calibrated_col="energy_peak_value",
            ph_smoothing_fwhm=50,
        use_expr=pl.col("state_label")=="START"
        )
    ch2.step_plot(0)
    mo.mpl.interactive(plt.gcf())
    return ch2,


@app.cell
def __(moss, np):
    def find_local_maxima(pulse_heights, gaussian_fwhm):
        """Smears each pulse by a gaussian of gaussian_fhwm and finds local maxima,
        returns a list of their locations in pulse_height units (sorted by number of
        pulses in peak) AND their peak values as: (peak_locations, peak_intensities)

        Args:
            pulse_heights (np.array(dtype=float)): a list of pulse heights (eg p_filt_value)
            gaussian_fwhm = fwhm of a gaussian that each pulse is smeared with, in same units as pulse heights
        """
        # kernel density estimation (with a gaussian kernel)
        n = 128 * 1024
        gaussian_fwhm = float(gaussian_fwhm)
        # The above ensures that lo & hi are floats, so that (lo-hi)/n is always a float in python2
        sigma = gaussian_fwhm / (np.sqrt(np.log(2) * 2) * 2)
        tbw = 1.0 / sigma / (np.pi * 2)
        lo = np.min(pulse_heights) - 3 * gaussian_fwhm
        hi = np.max(pulse_heights) + 3 * gaussian_fwhm
        hist, bins = np.histogram(pulse_heights, np.linspace(lo, hi, n + 1))
        tx = np.fft.rfftfreq(n, (lo - hi) / n)
        ty = np.exp(-tx**2 / 2 / tbw**2)
        x = (bins[1:] + bins[:-1]) / 2
        y = np.fft.irfft(np.fft.rfft(hist) * ty)

        flag = (y[1:-1] > y[:-2]) & (y[1:-1] > y[2:])
        lm = np.arange(1, n - 1)[flag]
        lm = lm[np.argsort(-y[lm])]
        bin_centers, step_size = moss.misc.midpoints_and_step_size(bins)
        return np.array(x[lm]), np.array(y[lm]), (hist, bin_centers, y)
    return find_local_maxima,


@app.cell
def __(moss, np, plt):
    from dataclasses import dataclass

    @dataclass(frozen=True)
    class SmoothedLocalMaximaResult:
        fwhm_pulse_height_units: float
        bin_centers: np.ndarray
        counts: np.ndarray
        smoothed_counts: np.ndarray
        local_maxima_inds: np.ndarray # inds into bin_centers

        def inds_sorted_by_peak_height(self):
            return self.local_maxima_inds[np.argsort(-self.peak_height())]

        def inds_sorted_by_prominence(self):
            return self.local_maxima_inds[np.argsort(-self.prominence())]

        def ph_sorted_by_prominence(self):
            return self.bin_centers[self.inds_sorted_by_prominence()]

        def ph_sorted_by_peak_height(self):
            return self.bin_centers[self.inds_sorted_by_peak_height()]

        def peak_height(self):
            return self.smoothed_counts[self.local_maxima_inds]

        def prominence(self):
            peak_height = self.peak_height()
            return np.diff(peak_height, prepend=0)-np.diff(peak_height,append=0)

        def plot(self, assignment_result=None, n_highlight=10, plot_counts=False, ax=None):
            if ax is None:
                plt.figure()
                ax = plt.gca()
            inds_prominence = self.inds_sorted_by_prominence()[:n_highlight]
            inds_peak_height = self.inds_sorted_by_peak_height()[:n_highlight]
            if plot_counts:
                ax.plot(self.bin_centers, self.counts, label="counts")
            ax.plot(self.bin_centers, self.smoothed_counts, label="smoothed_counts")
            ax.plot(self.bin_centers[self.local_maxima_inds],
                   self.smoothed_counts[self.local_maxima_inds],".",
                   label="peaks") 
            if assignment_result is not None:
                inds_assigned = np.searchsorted(self.bin_centers, assignment_result.ph_assigned)
                inds_unassigned = np.searchsorted(self.bin_centers, assignment_result.ph_unassigned())
                bin_centers_assigned = self.bin_centers[inds_assigned]
                bin_centers_unassigned = self.bin_centers[inds_unassigned]
                smoothed_counts_assigned = self.smoothed_counts[inds_assigned]
                smoothed_counts_unassigned = self.smoothed_counts[inds_unassigned]
                ax.plot(bin_centers_assigned, smoothed_counts_assigned,"o",label="assigned")
                ax.plot(bin_centers_unassigned, smoothed_counts_unassigned,"o",label="unassigned")
                for name, x, y in zip(assignment_result.names_target, bin_centers_assigned, smoothed_counts_assigned):
                    ax.annotate(str(name), (x,y), rotation=30)
                ax.set_title(f"SmoothedLocalMaximaResult rms_residual={assignment_result.rms_residual:.2f} eV")

            else:
                ax.plot(self.bin_centers[inds_prominence],
                self.smoothed_counts[inds_prominence],"o",
                label=f"{n_highlight} most prominent")
                ax.plot(self.bin_centers[inds_peak_height],
                self.smoothed_counts[inds_peak_height],"v",
                label=f"{n_highlight} highest")     
                ax.set_title("SmoothedLocalMaximaResult")

            ax.legend()
            ax.set_xlabel("pulse height")
            ax.set_ylabel("intensity")
            ax.set_ylim(1/self.fwhm_pulse_height_units, ax.get_ylim()[1])


            return ax

    def smooth_hist_with_gauassian_by_fft(hist, fwhm_in_bin_number_units):
        sigma = fwhm_in_bin_number_units / (np.sqrt(np.log(2) * 2) * 2)
        tbw = 1.0 / sigma / (np.pi * 2)
        tx = np.fft.rfftfreq(len(hist))
        ty = np.exp(-tx**2 / 2 / tbw**2)
        y = np.fft.irfft(np.fft.rfft(hist) * ty)
        return y

    def hist_smoothed(pulse_heights, fwhm_pulse_height_units, bin_edges=None):
        if bin_edges is None:
            n = 128 * 1024
            lo = np.min(pulse_heights) - 3 * fwhm_pulse_height_units
            hi = np.max(pulse_heights) + 3 * fwhm_pulse_height_units
            bin_edges =  np.linspace(lo, hi, n + 1)
        bin_centers, step_size = moss.misc.midpoints_and_step_size(bin_edges)    
        counts,_ = np.histogram(pulse_heights, bin_edges)
        fwhm_in_bin_number_units = fwhm_pulse_height_units/step_size
        smoothed_counts = smooth_hist_with_gauassian_by_fft(counts, fwhm_in_bin_number_units)
        return smoothed_counts, bin_edges, counts

    def local_maxima(y):
        flag = (y[1:-1] > y[:-2]) & (y[1:-1] > y[2:])
        local_maxima_inds = np.nonzero(flag)[0]
        return local_maxima_inds

    def peakfind_local_maxima_of_smoothed_hist(pulse_heights, fwhm_pulse_height_units, bin_edges=None):
        smoothed_counts, bin_edges, counts = hist_smoothed(pulse_heights, fwhm_pulse_height_units, bin_edges)
        bin_centers, step_size = moss.misc.midpoints_and_step_size(bin_edges)    
        local_maxima_inds = local_maxima(smoothed_counts)
        return SmoothedLocalMaximaResult(fwhm_pulse_height_units,bin_centers, counts, smoothed_counts, local_maxima_inds)
    return (
        SmoothedLocalMaximaResult,
        dataclass,
        hist_smoothed,
        local_maxima,
        peakfind_local_maxima_of_smoothed_hist,
        smooth_hist_with_gauassian_by_fft,
    )


@app.cell
def __(ch2, mass, peakfind_local_maxima_of_smoothed_hist, pl):
    line_names = ["AlKAlpha", "MgKAlpha", "ClKAlpha", "ScKAlpha", "CoKAlpha","MnKAlpha","VKAlpha","CuKAlpha"]
    uncalibrated_col, calibrated_col = "filtValue", "energy_filtValue"
    use_expr = pl.col("state_label")=="START"
    (names, ee) = mass.algorithms.line_names_and_energies(line_names)
    uncalibrated = ch2.good_series(uncalibrated_col, use_expr=use_expr).to_numpy()
    pfresult = peakfind_local_maxima_of_smoothed_hist(uncalibrated, fwhm_pulse_height_units=50)
    return (
        calibrated_col,
        ee,
        line_names,
        names,
        pfresult,
        uncalibrated,
        uncalibrated_col,
        use_expr,
    )


@app.cell
def __(assignment_result, mo, pfresult, plt):
    pfresult.plot(assignment_result)
    mo.mpl.interactive(plt.gcf())
    return


@app.cell
def __(
    ee,
    find_best_residual_among_all_possible_assignments2,
    mo,
    names,
    pfresult,
    plt,
):
    _n=len(ee) + 3
    assignment_result = find_best_residual_among_all_possible_assignments2(pfresult.ph_sorted_by_prominence()[:_n], ee, names)
    assignment_result.plot()
    mo.mpl.interactive(plt.gcf())
    return assignment_result,


@app.cell
def __(dataclass, np, plt):
    import itertools

    @dataclass(frozen=True)
    class BestAssignmentPfitGainResult:
        rms_residual: float
        ph_assigned: np.ndarray
        residual_e: np.ndarray
        assignment_inds: np.ndarray
        pfit_gain: np.polynomial.Polynomial
        energy_target: np.ndarray
        names_target: list[str] # list of strings with names for the energies in energy_target
        ph_target: np.ndarry # longer than energy target by 0-3

        def ph_unassigned(self):
            return np.array(list(set(self.ph_target)-set(self.ph_assigned)))

        def plot(self, ax=None):
            if ax is None:
                plt.figure()
                ax=plt.gca()
            gain = self.ph_assigned/self.energy_target
            ax.plot(self.ph_assigned, self.ph_assigned/self.energy_target, "o")
            ax.plot(self.ph_assigned, self.pfit_gain(self.ph_assigned))
            ax.set_xlabel("pulse_height")
            ax.set_ylabel("gain")
            ax.set_title(f"BestAssignmentPfitGainResult rms_residual={self.rms_residual:.2f} eV")
            assert len(self.names_target) == len(self.ph_assigned)
            for name, x, y in zip(self.names_target, self.ph_assigned, gain):
                ax.annotate(str(name), (x,y))

    def find_pfit_gain_residual(ph, e):
        assert len(ph) == len(e)
        gain = ph/e
        pfit_gain = np.polynomial.Polynomial.fit(ph, gain, deg=2)
        def ph2energy(ph):
            return ph/pfit_gain(ph)
        predicted_e = ph2energy(ph)
        residual_e = e-predicted_e
        return residual_e, pfit_gain

    def find_best_residual_among_all_possible_assignments2(ph, e, names):
        best_rms_residual, best_ph_assigned, best_residual_e, best_assignment_inds, best_pfit = find_best_residual_among_all_possible_assignments(ph,e)
        return BestAssignmentPfitGainResult(best_rms_residual, best_ph_assigned, best_residual_e, best_assignment_inds, best_pfit, e, names, ph)

    def find_best_residual_among_all_possible_assignments(ph, e):
        assert len(ph) >= len(e)
        ph=np.sort(ph)
        assignments_inds = itertools.combinations(np.arange(len(ph)), len(e))
        best_rms_residual = np.inf
        best_ph_assigned = None
        best_residual_e = None
        best_pfit = None
        for i, assignment_inds in enumerate(assignments_inds):
            assignment_inds = np.array(assignment_inds)
            ph_assigned = np.array(ph[assignment_inds])
            residual_e, pfit_gain = find_pfit_gain_residual(ph_assigned,e)
            rms_residual = np.std(residual_e)
            if rms_residual < best_rms_residual:
                best_rms_residual=rms_residual
                best_ph_assigned = ph_assigned
                best_residual_e = residual_e
                best_assignment_inds = assignment_inds
                best_pfit = pfit_gain
        return best_rms_residual, best_ph_assigned, best_residual_e, best_assignment_inds, best_pfit
    return (
        BestAssignmentPfitGainResult,
        find_best_residual_among_all_possible_assignments,
        find_best_residual_among_all_possible_assignments2,
        find_pfit_gain_residual,
        itertools,
    )


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run()
