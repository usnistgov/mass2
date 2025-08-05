from dataclasses import dataclass
import numpy as np
import pylab as plt  # type: ignore
import mass2 as mass
from mass2 import moss
import polars as pl
from matplotlib.axes._axes import Axes
from mass2.moss.channel import Channel
from numpy import float32, float64, ndarray
from polars.dataframe.frame import DataFrame
from numpy.polynomial import Polynomial
from scipy.optimize._optimize import OptimizeResult  # type: ignore
import typing
from typing import List, Optional, Tuple, Union
import itertools


def rank_3peak_assignments(
    ph,
    e,
    line_names,
    max_fractional_energy_error_3rd_assignment=0.1,
    min_gain_fraction_at_ph_30k=0.25,
):
    # we explore possible line assignments, and down select based on knowledge of gain curve shape
    # gain = ph/e, and we assume gain starts at zero, decreases with pulse height, and
    # that a 2nd order polynomial is a reasonably good approximation
    # with one assignment we model the gain as constant, and use that to find the most likely
    # 2nd assignments, then we model the gain as linear, and use that to rank 3rd assignments
    dfe = pl.DataFrame({"e0_ind": np.arange(len(e)), "e0": e, "name": line_names})
    dfph = pl.DataFrame({"ph0_ind": np.arange(len(ph)), "ph0": ph})
    # dfph should know about peak_area and use it to weight choices somehow

    # 1st assignments ####
    # e0 and ph0 are the first assignment
    df0 = (
        dfe.join(dfph, how="cross")
        .with_columns(gain0=pl.col("ph0") / pl.col("e0"))
    )
    # 2nd assignments ####
    # e1 and ph1 are the 2nd assignment
    df1 = (
        df0.join(df0, how="cross")
        .rename({"e0_right": "e1", "ph0_right": "ph1"})
        .drop("e0_ind_right", "ph0_ind_right", "gain0_right")
    )
    # 1) keep only assignments with e0<e1 and ph0<ph1 to avoid looking at the same pair in reverse
    df1 = df1.filter((pl.col("e0") < pl.col("e1")).and_(pl.col("ph0") < pl.col("ph1")))
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
    df1 = df1.filter(pl.col("gain_frac_at_ph30k") > min_gain_fraction_at_ph_30k)

    # 3rd assignments ####
    # e2 and ph2 are the 3rd assignment
    df2 = df1.join(df0.select(e2="e0", ph2="ph0"), how="cross")
    df2 = df2.with_columns(
        gain_at_ph2=pl.col("gain_at_0") + pl.col("gain_slope") * pl.col("ph2")
    )
    df2 = df2.with_columns(e_at_ph2=pl.col("ph2") / pl.col("gain_at_ph2"))
    df2 = df2.filter((pl.col("e1") < pl.col("e2")).and_(pl.col("ph1") < pl.col("ph2")))
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


@dataclass(frozen=True)
class BestAssignmentPfitGainResult:
    rms_residual: float
    ph_assigned: np.ndarray
    residual_e: np.ndarray
    assignment_inds: np.ndarray
    pfit_gain: np.polynomial.Polynomial
    energy_target: np.ndarray
    names_target: list[str]  # list of strings with names for the energies in energy_target
    ph_target: np.ndarray  # longer than energy target by 0-3

    def ph_unassigned(self) -> ndarray:
        return np.array(list(set(self.ph_target)-set(self.ph_assigned)))

    def plot(self, ax: Optional[Axes] = None):
        if ax is None:
            plt.figure()
            ax = plt.gca()
        gain = self.ph_assigned/self.energy_target
        ax.plot(self.ph_assigned, self.ph_assigned/self.energy_target, "o")
        ph_large_range = np.linspace(0, self.ph_assigned[-1]*1.1, 51)
        ax.plot(ph_large_range, self.pfit_gain(ph_large_range))
        ax.set_xlabel("pulse_height")
        ax.set_ylabel("gain")
        ax.set_title(f"BestAssignmentPfitGainResult rms_residual={self.rms_residual:.2f} eV")
        assert len(self.names_target) == len(self.ph_assigned)
        for name, x, y in zip(self.names_target, self.ph_assigned, gain):
            ax.annotate(str(name), (x, y))

    def phzerogain(self) -> float64:
        # the pulse height at which the gain is zero
        # for now I'm counting on the roots being ordered, we want the positive root where gain goes zero
        # since our function is invalid outside that range
        return self.pfit_gain.roots()[1]

    def ph2energy(self, ph: Union[ndarray, float]) -> Union[float64, ndarray]:
        return ph/self.pfit_gain(ph)

    def energy2ph(self, energy):
        if self.pfit_gain.degree() == 2:
            return self._energy2ph_deg2(energy)
        elif self.pfit_gain.degree() == 1:
            return self._energy2ph_deg1(energy)
        elif self.pfit_gain.degree() == 0:
            return self._energy2ph_deg0(energy)
        else:
            raise Exception("degree out of range")

    def _energy2ph_deg2(self, energy):
        # ph2energy is equivalent to this with y=energy, x=ph
        # y = x/(c + b*x + a*x^2)
        # so
        # y*c + (y*b-1)*x + a*x^2 = 0
        # and given that we've selected for well formed calibrations,
        # we know which root we want
        cba = self.pfit_gain.convert().coef
        c, bb, a = cba*energy
        b = bb-1
        ph = (-b-np.sqrt(b**2-4*a*c))/(2*a)
        return ph

    def _energy2ph_deg1(self, energy):
        # ph2energy is equivalent to this with y=energy, x=ph
        # y = x/(b + a*x)
        # so
        # x = y*b/(1-y*a)
        # and given that we've selected for well formed calibrations,
        # we know which root we want
        b, a = self.pfit_gain.convert().coef
        y = energy
        ph = y*b/(1-y*a)
        return ph

    def _energy2ph_deg0(self, energy):
        # ph2energy is equivalent to this with y=energy, x=ph
        # y = x/(a)
        # so
        # x = y*a
        a, = self.pfit_gain.convert().coef
        y = energy
        ph = y*a
        return ph

    def predicted_energies(self):
        return self.ph2energy(self.ph_assigned)


@dataclass(frozen=True)
class SmoothedLocalMaximaResult:
    fwhm_pulse_height_units: float
    bin_centers: np.ndarray
    counts: np.ndarray
    smoothed_counts: np.ndarray
    local_maxima_inds: np.ndarray  # inds into bin_centers
    local_minima_inds: np.ndarray  # inds into bin_centers

    def inds_sorted_by_peak_height(self) -> ndarray:
        return self.local_maxima_inds[np.argsort(-self.peak_height())]

    def inds_sorted_by_prominence(self) -> ndarray:
        return self.local_maxima_inds[np.argsort(-self.prominence())]

    def ph_sorted_by_prominence(self) -> ndarray:
        return self.bin_centers[self.inds_sorted_by_prominence()]

    def ph_sorted_by_peak_height(self):
        return self.bin_centers[self.inds_sorted_by_peak_height()]

    def peak_height(self) -> ndarray:
        return self.smoothed_counts[self.local_maxima_inds]

    def prominence(self) -> ndarray:
        assert len(self.local_minima_inds) == len(self.local_maxima_inds)+1, "peakfind_local_maxima_of_smoothed_hist must ensure this "
        prominence = np.zeros_like(self.local_maxima_inds, dtype=float64)
        for i in range(len(self.local_maxima_inds)):
            sc_max = self.smoothed_counts[self.local_maxima_inds[i]]
            sc_min_before = self.smoothed_counts[self.local_minima_inds[i]]
            sc_min_after = self.smoothed_counts[self.local_minima_inds[i+1]]
            prominence[i] = (2*sc_max - sc_min_before - sc_min_after)/2
        assert np.all(prominence >= 0), "prominence should be non-negative"
        return prominence

    def plot(self, assignment_result: Optional[BestAssignmentPfitGainResult] = None, n_highlight: int = 10,
             plot_counts: bool = False, ax: Optional[Axes] = None) -> Axes:
        if ax is None:
            plt.figure()
            ax = plt.gca()
        inds_prominence = self.inds_sorted_by_prominence()[:n_highlight]
        inds_peak_height = self.inds_sorted_by_peak_height()[:n_highlight]
        if plot_counts:
            ax.plot(self.bin_centers, self.counts, label="counts")
        ax.plot(self.bin_centers, self.smoothed_counts, label="smoothed_counts")
        ax.plot(self.bin_centers[self.local_maxima_inds],
                self.smoothed_counts[self.local_maxima_inds], ".",
                label="peaks")
        if assignment_result is not None:
            inds_assigned = np.searchsorted(self.bin_centers, assignment_result.ph_assigned)
            inds_unassigned = np.searchsorted(self.bin_centers, assignment_result.ph_unassigned())
            bin_centers_assigned = self.bin_centers[inds_assigned]
            bin_centers_unassigned = self.bin_centers[inds_unassigned]
            smoothed_counts_assigned = self.smoothed_counts[inds_assigned]
            smoothed_counts_unassigned = self.smoothed_counts[inds_unassigned]
            ax.plot(bin_centers_assigned, smoothed_counts_assigned, "o", label="assigned")
            ax.plot(bin_centers_unassigned, smoothed_counts_unassigned, "o", label="unassigned")
            for name, x, y in zip(assignment_result.names_target, bin_centers_assigned, smoothed_counts_assigned):
                ax.annotate(str(name), (x, y), rotation=30)
            ax.set_title(f"SmoothedLocalMaximaResult rms_residual={assignment_result.rms_residual:.2f} eV")

        else:
            ax.plot(self.bin_centers[inds_prominence],
                    self.smoothed_counts[inds_prominence], "o",
                    label=f"{n_highlight} most prominent")
            ax.plot(self.bin_centers[inds_peak_height],
                    self.smoothed_counts[inds_peak_height], "v",
                    label=f"{n_highlight} highest")
            ax.set_title("SmoothedLocalMaximaResult")

        ax.legend()
        ax.set_xlabel("pulse height")
        ax.set_ylabel("intensity")
        # print(f"{np.amax(self.smoothed_counts)=} {np.amin(self.smoothed_counts)=} ")
        # ax.set_ylim(1/self.fwhm_pulse_height_units, ax.get_ylim()[1])

        return ax


def smooth_hist_with_gauassian_by_fft(hist: ndarray, fwhm_in_bin_number_units: float64) -> ndarray:
    kernel = smooth_hist_with_gauassian_by_fft_compute_kernel(len(hist), fwhm_in_bin_number_units)
    y = np.fft.irfft(np.fft.rfft(hist) * kernel)
    return y


def smooth_hist_with_gauassian_by_fft_compute_kernel(nbins: int, fwhm_in_bin_number_units: float64) -> ndarray:
    sigma = fwhm_in_bin_number_units / (np.sqrt(np.log(2) * 2) * 2)
    tbw = 1.0 / sigma / (np.pi * 2)
    tx = np.fft.rfftfreq(nbins)
    kernel = np.exp(-tx**2 / 2 / tbw**2)
    return kernel


def hist_smoothed(pulse_heights: ndarray, fwhm_pulse_height_units: int,
                  bin_edges: Optional[ndarray] = None) -> Tuple[ndarray, ndarray, ndarray]:
    pulse_heights = pulse_heights.astype(np.float64)
    # convert to float64 to avoid warpping subtraction and platform specific behavior regarding uint16s
    # linux CI will throw errors, while windows does not, but maybe is just silently wrong?
    assert len(pulse_heights > 10), "not enough pulses"
    if bin_edges is None:
        n = 128 * 1024
        lo = (np.min(pulse_heights) - 3 * fwhm_pulse_height_units).astype(np.float64)
        hi = (np.max(pulse_heights) + 3 * fwhm_pulse_height_units).astype(np.float64)
        bin_edges = np.linspace(lo, hi, n + 1)

    _, step_size = moss.misc.midpoints_and_step_size(bin_edges)
    counts, _ = np.histogram(pulse_heights, bin_edges)
    fwhm_in_bin_number_units = fwhm_pulse_height_units/step_size
    smoothed_counts = smooth_hist_with_gauassian_by_fft(counts, fwhm_in_bin_number_units)
    return smoothed_counts, bin_edges, counts


def local_maxima(y: ndarray) -> ndarray:
    local_maxima_inds = []
    local_minima_inds = []
    increasing = False
    for i in range(len(y)-1):
        if increasing and (y[i+1] < y[i]):
            local_maxima_inds.append(i)
            increasing = False
        if not increasing and (y[i+1] > y[i]):
            local_minima_inds.append(i)
            increasing = True
    # increasing starts false, so we always start with a miniumum
    return np.array(local_maxima_inds), np.array(local_minima_inds)


def peakfind_local_maxima_of_smoothed_hist(pulse_heights: ndarray, fwhm_pulse_height_units: int,
                                           bin_edges: Optional[ndarray] = None) -> SmoothedLocalMaximaResult:
    assert len(pulse_heights > 10), "not enough pulses"
    smoothed_counts, bin_edges, counts = hist_smoothed(pulse_heights, fwhm_pulse_height_units, bin_edges)
    bin_centers, _step_size = moss.misc.midpoints_and_step_size(bin_edges)
    local_maxima_inds, local_minima_inds = local_maxima(smoothed_counts)
    # require a minimum before and after a maximum (the first check is redundant with behavior of local_maxima)
    if local_maxima_inds[0] < local_minima_inds[0]:
        local_maxima_inds = local_maxima_inds[1:]
    if local_maxima_inds[-1] > local_minima_inds[-1]:
        local_maxima_inds = local_maxima_inds[:-1]
    return SmoothedLocalMaximaResult(fwhm_pulse_height_units, bin_centers,
                                     counts, smoothed_counts, local_maxima_inds,
                                     local_minima_inds)


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
    bin_centers, _step_size = moss.misc.midpoints_and_step_size(bins)
    return np.array(x[lm]), np.array(y[lm]), (hist, bin_centers, y)


def rank_assignments(ph, e):
    # ph is a list of peak heights longer than e
    # e is a list of known peak energies
    # we want to find the set of peak heights from ph that are closest to being locally linear with the energies in e
    e = np.array(sorted(list(e)))
    ph = np.array(sorted(list(ph)))
    pha = np.array(list(itertools.combinations(ph, len(e))))
    pha_inds = np.array(list(itertools.combinations(np.arange(len(ph)), len(e))))
    # pha[i,:] is one choice of len(e) values from ph to assign to e
    # we use linear interpolation of the form y = y0 + (y1-y0)*(x-x0)/(x1-x0)
    # on each set of 3 values
    # with y = e and x = ph
    # x is pha[:,1:-1], x0 is pha[:,:-2], x1 is pha[:,2:]
    x = pha[:, 1:-1]
    x0 = pha[:, :-2]
    x1 = pha[:, 2:]
    y0 = e[:-2]
    y1 = e[2:]
    x_m_x0_over_x1_m_x0 = (x-x0)/(x1-x0)
    y = y0+(y1-y0)*x_m_x0_over_x1_m_x0
    y_expected = e[1:-1]
    rms_e_residual = moss.misc.root_mean_squared(y-y_expected, axis=1)
    # prefer negative slopes for gain
    # gain_first = ph[0]/e[0]
    # gain_last = ph[-1]/e[-1]
    return rms_e_residual, pha, pha_inds


def find_optimal_assignment(ph, e):
    # ph is a list of peak heights longer than e
    # e is a list of known peak energies
    # we want to find the set of peak heights from ph that are closest to being locally linear with the energies in e

    # when given 3 or less energies to match, use the largest peaks in peak order
    assert len(e) >= 1
    if len(e) <= 2:
        return 0, np.array(sorted(ph[:len(e)]))

    rms_e_residual, pha, _pha_inds = rank_assignments(ph, e)
    ind = np.argmin(rms_e_residual)
    return rms_e_residual[ind], pha[ind]


def find_optimal_assignment_height_info(ph, e, line_heights_allowed):
    # ph is a list of peak heights longer than e
    # e is a list of known peak energies
    # we want to find the set of peak heights from ph that are closest to being locally linear with the energies in e

    # when given 3 or less energies to match, use the largest peaks in peak order
    assert len(e) >= 1
    if len(e) <= 2:
        return 0, np.array(sorted(ph[:len(e)]))

    rms_e_residual, pha, pha_inds = rank_assignments(ph, e)
    best_ind = None
    best_rms_residual = np.inf
    print(f"{e=}")
    for i_assign in range(len(rms_e_residual)):
        rms_e_candidate = rms_e_residual[i_assign]
        # pha[i,:] is one choice of len(e) values from ph to assign to e
        pha_inds_candidate = pha_inds[i_assign, :]
        if rms_e_candidate > best_rms_residual:
            continue
        # check if peaks mathch height info
        print(f"{pha_inds_candidate=}")
        print(f"{line_heights_allowed=}")
        failed_line_height_check = False
        for j in range(len(e)):
            if pha_inds_candidate[j] not in line_heights_allowed[j]:
                failed_line_height_check = True
                print("not allowed")
                a = pha_inds_candidate[j]
                b = line_heights_allowed[j]
                print(f"{a=} {b=}")
                break
        if failed_line_height_check:
            continue
        print("is new best!")
        best_rms_residual = rms_e_candidate
        best_ind = i_assign
    if best_ind is None:
        raise Exception("no assignment found satisfying peak height info")
    return rms_e_residual[best_ind], pha[best_ind, :]


def find_optimal_assignment2(ph, e, line_names):
    rms_e_residual, pha = find_optimal_assignment(ph, e)
    gain = pha/e
    deg = min(len(e)-1, 2)
    if deg == 0:
        pfit_gain = np.polynomial.Polynomial(gain)
    else:
        pfit_gain = np.polynomial.Polynomial.fit(pha,
                                                 gain,
                                                 deg=min(len(e)-1, 2))
    result = moss.rough_cal.BestAssignmentPfitGainResult(
        rms_e_residual,
        ph_assigned=pha,
        residual_e=None,
        assignment_inds=None,
        pfit_gain=pfit_gain,
        energy_target=e,
        names_target=line_names,
        ph_target=ph,
    )
    return result


def find_optimal_assignment2_height_info(ph, e, line_names, line_heights_allowed):
    rms_e_residual, pha = find_optimal_assignment_height_info(ph, e, line_heights_allowed)
    gain = pha/e
    deg = min(len(e)-1, 2)
    if deg == 0:
        pfit_gain = np.polynomial.Polynomial(gain)
    else:
        pfit_gain = np.polynomial.Polynomial.fit(pha,
                                                 gain,
                                                 deg=min(len(e)-1, 2))
    result = moss.rough_cal.BestAssignmentPfitGainResult(
        rms_e_residual,
        ph_assigned=pha,
        residual_e=None,
        assignment_inds=None,
        pfit_gain=pfit_gain,
        energy_target=e,
        names_target=line_names,
        ph_target=ph,
    )
    return result


def find_pfit_gain_residual(ph: ndarray, e: Tuple[float, float, float, float, float, float]) -> Tuple[ndarray, Polynomial]:
    assert len(ph) == len(e)
    gain = ph/e
    pfit_gain = np.polynomial.Polynomial.fit(ph, gain, deg=2)

    def ph2energy(ph):
        return ph/pfit_gain(ph)
    predicted_e = ph2energy(ph)
    residual_e = e-predicted_e
    return residual_e, pfit_gain


def find_best_residual_among_all_possible_assignments2(ph: ndarray, e: ndarray, names: list[str]) -> BestAssignmentPfitGainResult:
    best_rms_residual, best_ph_assigned, best_residual_e, best_assignment_inds, best_pfit = \
        find_best_residual_among_all_possible_assignments(ph, e)
    return BestAssignmentPfitGainResult(float(best_rms_residual), best_ph_assigned, best_residual_e,
                                        best_assignment_inds, best_pfit, e, names, ph)


def find_best_residual_among_all_possible_assignments(ph: ndarray, e: Tuple[float, float, float, float, float, float]
                                                      ) -> Tuple[float64, ndarray, ndarray, ndarray, Polynomial]:
    assert len(ph) >= len(e)
    ph = np.sort(ph)
    assignments_inds = itertools.combinations(np.arange(len(ph)), len(e))
    best_rms_residual = np.inf
    best_ph_assigned = None
    best_residual_e = None
    best_pfit = None
    for i, assignment_inds in enumerate(assignments_inds):
        assignment_inds = np.array(assignment_inds)
        ph_assigned = np.array(ph[assignment_inds])
        residual_e, pfit_gain = find_pfit_gain_residual(ph_assigned, e)
        rms_residual = moss.misc.root_mean_squared(residual_e)
        if rms_residual < best_rms_residual:
            best_rms_residual = rms_residual
            best_ph_assigned = ph_assigned
            best_residual_e = residual_e
            best_assignment_inds = assignment_inds
            best_pfit = pfit_gain
    return best_rms_residual, best_ph_assigned, best_residual_e, best_assignment_inds, best_pfit


def drift_correct_entropy(slope: float64, indicator_zero_mean: ndarray, uncorrected: ndarray,
                          bin_edges: ndarray, fwhm_in_bin_number_units: int) -> float64:
    corrected = uncorrected * (1 + indicator_zero_mean * slope)
    smoothed_counts, bin_edges, _counts = hist_smoothed(corrected, fwhm_in_bin_number_units, bin_edges)
    w = smoothed_counts > 0
    return -(np.log(smoothed_counts[w]) * smoothed_counts[w]).sum()


def minimize_entropy_linear(indicator: ndarray, uncorrected: ndarray, bin_edges: ndarray,
                            fwhm_in_bin_number_units: int) -> Tuple[OptimizeResult, float32]:
    import scipy.optimize
    indicator_mean = np.mean(indicator)
    indicator_zero_mean = indicator-indicator_mean

    def entropy_fun(slope):
        return drift_correct_entropy(slope, indicator_zero_mean, uncorrected, bin_edges, fwhm_in_bin_number_units)

    result = scipy.optimize.minimize_scalar(entropy_fun, bracket=[0, 0.1])
    return result, indicator_mean


def eval_3peak_assignment_pfit_gain(
    ph_assigned, e_assigned, possible_phs, line_energies, line_names
):
    assert len(np.unique(ph_assigned)) == len(ph_assigned), "assignments must be unique"
    assert len(np.unique(e_assigned)) == len(e_assigned), "assignments must be unique"
    assert all(np.diff(ph_assigned) > 0), "assignments must be sorted"
    assert all(np.diff(e_assigned) > 0), "assignments must be sorted"
    gain_assigned = np.array(ph_assigned) / np.array(e_assigned)
    pfit_gain_3peak = np.polynomial.Polynomial.fit(ph_assigned, gain_assigned, deg=2)
    if pfit_gain_3peak.deriv(1)(0) > 0:
        # well formed calibration have negative derivative at zero pulse height
        return np.inf, "pfit_gain_3peak deriv at 0 should be <0"
    if pfit_gain_3peak(1e5) < 0:
        # well formed calibration have positive gain at 1e5
        return np.inf, "pfit_gain_3peak should be above zero at 100k ph"
    if any(np.iscomplex(pfit_gain_3peak.roots())):
        # well formed calibrations have real roots
        return np.inf, "pfit_gain_3peak must have real roots"

    def ph2energy(ph):
        gain = pfit_gain_3peak(ph)
        return ph / gain

    cba = pfit_gain_3peak.convert().coef

    def energy2ph(energy):
        # ph2energy is equivalent to this with y=energy, x=ph
        # y = x/(c + b*x + a*x^2)
        # so
        # y*c + (y*b-1)*x + a*x^2 = 0
        # and given that we've selected for well formed calibrations,
        # we know which root we want
        c, bb, a = cba*energy
        b = bb-1
        ph = (-b-np.sqrt(b**2-4*a*c))/(2*a)
        return ph

    predicted_ph = predicted_ph = [energy2ph(_e) for _e in line_energies]
    df = pl.DataFrame(
        {
            "line_energy": line_energies,
            "line_name": line_names,
            "predicted_ph": predicted_ph,
        }
    ).sort(by="predicted_ph")
    dfph = pl.DataFrame(
        {"possible_ph": possible_phs, "ph_ind": np.arange(len(possible_phs))}
    ).sort(by="possible_ph")
    # for each e find the closest possible_ph to the calculaed predicted_ph
    # we started with assignments for 3 energies
    # now we have assignments for all energies
    df = df.join_asof(
        dfph, left_on="predicted_ph", right_on="possible_ph", strategy="nearest"
    )
    n_unique = len(df["possible_ph"].unique())
    if n_unique < len(df):
        # assigned multiple energies to same pulseheight, not a good cal
        return np.inf, "assignments should be unique"

    # now we evaluate the assignment and create a result object
    residual_e, pfit_gain = moss.rough_cal.find_pfit_gain_residual(
        df["possible_ph"].to_numpy(), df["line_energy"].to_numpy()
    )
    if pfit_gain(1e5) < 0:
        # well formed calibration have positive gain at 1e5
        return np.inf, "pfit_gain should be above zero at 100k ph"
    if any(np.iscomplex(pfit_gain.roots())):
        # well formed calibrations have real roots
        return np.inf, "pfit_gain should not have complex roots"
    rms_residual_e = moss.misc.root_mean_squared(residual_e)
    result = moss.rough_cal.BestAssignmentPfitGainResult(
        rms_residual_e,
        ph_assigned=df["possible_ph"].to_numpy(),
        residual_e=residual_e,
        assignment_inds=df["ph_ind"].to_numpy(),
        pfit_gain=pfit_gain,
        energy_target=df["line_energy"].to_numpy(),
        names_target=df["line_name"].to_list(),
        ph_target=possible_phs,
    )
    return rms_residual_e, result


@dataclass(frozen=True)
class RoughCalibrationStep(moss.CalStep):
    pfresult: SmoothedLocalMaximaResult
    assignment_result: BestAssignmentPfitGainResult
    ph2energy: typing.Callable
    success: bool

    def calc_from_df(self, df: DataFrame) -> DataFrame:
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
        if self.success:
            self.dbg_plot_success(df, axs)
        else:
            self.dbg_plot_failure(df, axs)

    def dbg_plot_success(self, df: DataFrame, axs: Union[None, ndarray] = None):
        if axs is None:
            _, axs = plt.subplots(2, 1, figsize=(11, 6))
        self.assignment_result.plot(ax=axs[0])
        self.pfresult.plot(self.assignment_result, ax=axs[1])
        plt.tight_layout()

    def dbg_plot_failure(self, df: DataFrame, axs: Union[None, ndarray] = None):
        if axs is None:
            _, axs = plt.subplots(2, 1, figsize=(11, 6))
        self.pfresult.plot(self.assignment_result, ax=axs[1])
        plt.tight_layout()

    def energy2ph(self, energy: float64) -> float:
        return self.assignment_result.energy2ph(energy)

    @classmethod
    def learn_combinatoric(cls, ch: Channel, line_names: List[str], uncalibrated_col: str, calibrated_col: str,
                           ph_smoothing_fwhm: int, n_extra: int,
                           use_expr: bool = True
                           ) -> "RoughCalibrationStep":

        (names, ee) = mass.algorithms.line_names_and_energies(line_names)
        uncalibrated = ch.good_series(uncalibrated_col, use_expr=use_expr).to_numpy()
        assert len(uncalibrated) > 10, "not enough pulses"
        pfresult = moss.rough_cal.peakfind_local_maxima_of_smoothed_hist(uncalibrated,
                                                                         fwhm_pulse_height_units=ph_smoothing_fwhm)
        assignment_result = moss.rough_cal.find_optimal_assignment2(
            pfresult.ph_sorted_by_prominence()[:len(ee)+n_extra], ee, names)

        step = cls(
            [uncalibrated_col],
            [calibrated_col],
            ch.good_expr,
            use_expr=use_expr,
            pfresult=pfresult,
            assignment_result=assignment_result,
            ph2energy=assignment_result.ph2energy,
            success=True)
        return step

    @classmethod
    def learn_combinatoric_height_info(cls, ch: Channel, line_names: List[str], line_heights_allowed: List[List[int]],
                                       uncalibrated_col: str, calibrated_col: str,
                                       ph_smoothing_fwhm: int, n_extra: int,
                                       use_expr: bool = True
                                       ) -> "RoughCalibrationStep":

        (names, ee) = mass.algorithms.line_names_and_energies(line_names)
        uncalibrated = ch.good_series(uncalibrated_col, use_expr=use_expr).to_numpy()
        assert len(uncalibrated) > 10, "not enough pulses"
        pfresult = moss.rough_cal.peakfind_local_maxima_of_smoothed_hist(uncalibrated,
                                                                         fwhm_pulse_height_units=ph_smoothing_fwhm)
        assignment_result = moss.rough_cal.find_optimal_assignment2_height_info(
            pfresult.ph_sorted_by_prominence()[:len(ee)+n_extra], ee, names, line_heights_allowed)

        step = cls(
            [uncalibrated_col],
            [calibrated_col],
            ch.good_expr,
            use_expr=use_expr,
            pfresult=pfresult,
            assignment_result=assignment_result,
            ph2energy=assignment_result.ph2energy,
            success=True)
        return step

    @classmethod
    def learn_3peak(cls, ch: Channel,
                    line_names: list[str | float64],
                    uncalibrated_col: str = "filtValue",
                    calibrated_col: Optional[str] = None,
                    use_expr: bool | pl.Expr = True,
                    max_fractional_energy_error_3rd_assignment: float = 0.1,
                    min_gain_fraction_at_ph_30k: float = 0.25,
                    fwhm_pulse_height_units: float = 75,
                    n_extra_peaks: int = 10,
                    acceptable_rms_residual_e: float = 10) -> "RoughCalibrationStep":
        if calibrated_col is None:
            calibrated_col = f"energy_{uncalibrated_col}"
        (line_names, line_energies) = mass.algorithms.line_names_and_energies(line_names)
        uncalibrated = ch.good_series(uncalibrated_col, use_expr=use_expr).to_numpy()
        pfresult = moss.rough_cal.peakfind_local_maxima_of_smoothed_hist(
            uncalibrated, fwhm_pulse_height_units=fwhm_pulse_height_units
        )
        possible_phs = pfresult.ph_sorted_by_prominence()[:len(line_names)+n_extra_peaks]
        df3peak, _dfe = rank_3peak_assignments(
            possible_phs,
            line_energies,
            line_names,
            max_fractional_energy_error_3rd_assignment,
            min_gain_fraction_at_ph_30k,
        )
        best_rms_residual = np.inf
        best_assignment_result = None
        for assignment_row in df3peak.select("e0", "ph0", "e1", "ph1", "e2", "ph2", "e_err_at_ph2").iter_rows():
            e0, ph0, e1, ph1, e2, ph2, _e_err_at_ph2 = assignment_row
            rms_residual, assignment_result = eval_3peak_assignment_pfit_gain(
                [ph0, ph1, ph2], [e0, e1, e2], possible_phs, line_energies, line_names
            )
            if rms_residual < best_rms_residual:
                best_rms_residual = rms_residual
                best_assignment_result = assignment_result
                if rms_residual < acceptable_rms_residual_e:
                    break
        if not np.isinf(best_rms_residual):
            success = True
            ph2energy = best_assignment_result.ph2energy
            # df3peak_on_failure = None
        else:
            success = False

            def nanenergy(ph):
                return ph * np.nana
            ph2energy = nanenergy
            # df3peak_on_failure = df3peak
        # df3peak_on_failure = df3peak

        step = cls(
            [uncalibrated_col],
            [calibrated_col],
            ch.good_expr,
            use_expr=use_expr,
            pfresult=pfresult,
            assignment_result=best_assignment_result,
            ph2energy=ph2energy,
            success=success)
        return step
