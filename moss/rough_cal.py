from dataclasses import dataclass
import numpy as np
import pylab as plt
import moss
import polars as pl

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
    kernel = smooth_hist_with_gauassian_by_fft_compute_kernel(len(hist), fwhm_in_bin_number_units)
    y = np.fft.irfft(np.fft.rfft(hist) * kernel)
    return y

def smooth_hist_with_gauassian_by_fft_compute_kernel(nbins, fwhm_in_bin_number_units):
    sigma = fwhm_in_bin_number_units / (np.sqrt(np.log(2) * 2) * 2)
    tbw = 1.0 / sigma / (np.pi * 2)
    tx = np.fft.rfftfreq(nbins)
    kernel = np.exp(-tx**2 / 2 / tbw**2)
    return kernel

def hist_smoothed(pulse_heights, fwhm_pulse_height_units, bin_edges=None):
    if bin_edges is None:
        n = 128 * 1024
        # force the use of float64 here, otherwise the bin spacings from
        # linspace can be uneven. seems platform dependent. windows doesn't need the forces float64
        # but linux, at least github CI, does
        lo = (np.min(pulse_heights) - 3 * fwhm_pulse_height_units).astype(np.float64)
        hi = (np.max(pulse_heights) + 3 * fwhm_pulse_height_units).astype(np.float64)
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
    ph_target: np.ndarray # longer than energy target by 0-3

    def ph_unassigned(self):
        return np.array(list(set(self.ph_target)-set(self.ph_assigned)))

    def plot(self, ax=None):
        if ax is None:
            plt.figure()
            ax=plt.gca()
        gain = self.ph_assigned/self.energy_target
        ax.plot(self.ph_assigned, self.ph_assigned/self.energy_target, "o")
        ph_large_range = np.linspace(0, self.ph_assigned[-1]*1.1,51)
        ax.plot(ph_large_range, self.pfit_gain(ph_large_range))
        ax.set_xlabel("pulse_height")
        ax.set_ylabel("gain")
        ax.set_title(f"BestAssignmentPfitGainResult rms_residual={self.rms_residual:.2f} eV")
        assert len(self.names_target) == len(self.ph_assigned)
        for name, x, y in zip(self.names_target, self.ph_assigned, gain):
            ax.annotate(str(name), (x,y))

    def phzerogain(self):
        # the pulse height at which the gain is zero
        # for now I'm counting on the roots being ordered, we want the positive root where gain goes zero
        # since our function is invalid outside that range
        return self.pfit_gain.roots()[1]

    def ph2energy(self, ph):
        return ph/self.pfit_gain(ph)
    
    def energy2ph(self, energy):
        import scipy.optimize
        sol = scipy.optimize.root_scalar(lambda ph: self.ph2energy(ph)-energy, 
                                   bracket=[1,self.phzerogain()*0.9])
        assert sol.converged
        return sol.root
    
    def predicted_energies(self):
        return self.ph2energy(self.ph_assigned)

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


def drift_correct_entropy(slope, indicator_zero_mean, uncorrected, bin_edges, fwhm_in_bin_number_units):
    corrected = uncorrected * (1 + indicator_zero_mean * slope)
    smoothed_counts, bin_edges, counts = hist_smoothed(corrected, fwhm_in_bin_number_units, bin_edges)
    w = smoothed_counts > 0
    return -(np.log(smoothed_counts[w]) * smoothed_counts[w]).sum()

def minimize_entropy_linear(indicator, uncorrected, bin_edges, fwhm_in_bin_number_units):
    import scipy.optimize
    indicator_mean = np.mean(indicator)
    indicator_zero_mean = indicator-indicator_mean

    def entropy_fun(slope):
        return drift_correct_entropy(slope, indicator_zero_mean, uncorrected, bin_edges, fwhm_in_bin_number_units)

    result = scipy.optimize.minimize_scalar(entropy_fun, bracket=[0,0.1])
    return result, indicator_mean

@dataclass(frozen=True)
class RoughCalibrationStep(moss.CalStep):
    pfresult: SmoothedLocalMaximaResult
    assignment_result: BestAssignmentPfitGainResult
    ph2energy: callable

    def calc_from_df(self, df):
        # only works with in memory data, but just takes it as numpy data and calls function
        # is much faster than map_elements approach, but wouldn't work with out of core data without some extra book keeping
        inputs_np = [df[input].to_numpy() for input in self.inputs]
        out = self.ph2energy(inputs_np[0])
        df2 = pl.DataFrame({self.output[0]: out}).with_columns(df)
        return df2

    def dbg_plot_old(self, df, bin_edges=np.arange(0, 10000, 1), axis=None, plotkwarg={}):
        series = moss.good_series(df, col=self.output[0], good_expr=self.good_expr, use_expr=self.use_expr)
        axis = moss.misc.plot_hist_of_series(series, bin_edges)
        axis.plot(self.line_energies, np.zeros(len(self.line_energies)), "o")
        for line_name, energy in zip(self.line_names, self.line_energies):
            axis.annotate(line_name, (energy, 0), rotation=90)
        np.set_printoptions(precision=2)
        energy_residuals = self.predicted_energies-self.line_energies
        axis.set_title(f"RoughCalibrationStep dbg_plot\n{energy_residuals=}")
        return axis
    
    def dbg_plot(self, df, axs=None):
        if axs is None:
            fig, axs = plt.subplots(2, 1, figsize=(11,6))
        self.assignment_result.plot(ax=axs[0])
        self.pfresult.plot(self.assignment_result, ax=axs[1])
        plt.tight_layout()
    
    def energy2ph(self, energy):
        return self.assignment_result.energy2ph(energy)
    
    @classmethod
    def learn(cls, ch, line_names, uncalibrated_col, calibrated_col, 
        ph_smoothing_fwhm, n_extra=3,
        use_expr=True
    ):
        import mass

        (names, ee) = mass.algorithms.line_names_and_energies(line_names)
        uncalibrated = ch.good_series(uncalibrated_col, use_expr=use_expr).to_numpy()
        pfresult = moss.rough_cal.peakfind_local_maxima_of_smoothed_hist(uncalibrated, 
                                                                         fwhm_pulse_height_units=ph_smoothing_fwhm)
        assignment_result = moss.rough_cal.find_best_residual_among_all_possible_assignments2(
            pfresult.ph_sorted_by_prominence()[:len(ee)+n_extra], ee, names)


        step = cls(
            [uncalibrated_col],
            [calibrated_col],
            ch.good_expr,
            use_expr=use_expr,
            pfresult=pfresult,
            assignment_result=assignment_result, 
            ph2energy=assignment_result.ph2energy)
        return step