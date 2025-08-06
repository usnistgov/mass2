"""
This file is intended to include algorithms that could be generally useful
for calibration. Mostly they are pulled out of the former
mass.calibration.young module.
"""

import itertools
import operator
import numpy as np

from collections.abc import Iterable

import matplotlib.pyplot as plt
import matplotlib.transforms as mtrans
from matplotlib.ticker import MaxNLocator

from ..common import isstr
from mass2.calibration.fluorescence_lines import STANDARD_FEATURES
import mass2 as mass


def line_names_and_energies(line_names):
    """Given a list of line_names, return (names, energies) in eV.

    Can also accept energies in eV directly and return (names, energies).
    """
    if len(line_names) <= 0:
        return [], []

    energies = [STANDARD_FEATURES.get(name_or_energy, name_or_energy)
                for name_or_energy in line_names]
    # names = [str(name_or_energy) for name_or_energy in line_names]
    return zip(*sorted(zip(line_names, energies), key=operator.itemgetter(1)))


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

    return np.array(x[lm]), np.array(y[lm])


def find_opt_assignment(peak_positions, line_names, nextra=2, nincrement=3, nextramax=8, maxacc=0.015):
    """Tries to find an assignment of peaks to line names that is reasonably self consistent and smooth

    Args:
        peak_positions (np.array(dtype=float)): a list of peak locations in arb units,
            e.g. p_filt_value units
        line_names (list[str or float)]): a list of calibration lines either as number (which is
            energies in eV), or name to be looked up in STANDARD_FEATURES
        nextra (int): the algorithm starts with the first len(line_names) + nextra peak_positions
        nincrement (int): each the algorithm fails to find a satisfactory peak assignment, it uses
            nincrement more lines
        nextramax (int): the algorithm stops incrementint nextra past this value, instead
            failing with a ValueError saying "no peak assignment succeeded"
        maxacc (float): an empirical number that determines if an assignment is good enough.
            The default number works reasonably well for tupac data
    """
    name_e, e_e = line_names_and_energies(line_names)

    n_sel_pp = len(line_names) + nextra  # number of peak_positions to use to line up to line_names
    nmax = len(line_names) + nextramax

    while True:
        sel_positions = np.asarray(peak_positions[:n_sel_pp], dtype="float")
        energies = np.asarray(e_e, dtype="float")
        assign = np.array(list(itertools.combinations(sel_positions, len(line_names))))
        assign.sort(axis=1)
        fracs = np.divide(energies[1:-1] - energies[:-2], energies[2:] - energies[:-2])
        est_pos = assign[:, :-2] * (1 - fracs) + assign[:, 2:] * fracs
        acc_est = np.linalg.norm(np.divide(est_pos - assign[:, 1:-1],
                                           assign[:, 2:] - assign[:, :-2]), axis=1)

        opt_assign_i = np.argmin(acc_est)
        acc = acc_est[opt_assign_i]
        opt_assign = assign[opt_assign_i]

        if acc > maxacc * np.sqrt(len(energies)):
            n_sel_pp += nincrement
            if n_sel_pp > nmax:
                msg = f"no peak assignment succeeded: acc {acc:g}, maxacc*sqrt(len(energies)) "\
                    f"{maxacc * np.sqrt(len(energies)):g}"
                raise ValueError(msg)
            else:
                continue
        else:
            return name_e, energies, list(opt_assign)


def build_fit_ranges_ph(line_names, excluded_line_names, approx_ecal, fit_width_ev):
    """Call build_fit_ranges() to get (lo,hi) for fitranges in energy units,
    then convert to ph using approx_ecal"""
    e_e, fit_lo_hi_energy, slopes_de_dph = build_fit_ranges(
        line_names, excluded_line_names, approx_ecal, fit_width_ev)
    fit_lo_hi_ph = []
    for lo, hi in fit_lo_hi_energy:
        lo_ph = approx_ecal.energy2ph(lo)
        hi_ph = approx_ecal.energy2ph(hi)
        fit_lo_hi_ph.append((lo_ph, hi_ph))

    return e_e, fit_lo_hi_ph, slopes_de_dph


def build_fit_ranges(line_names, excluded_line_names, approx_ecal, fit_width_ev):
    """Returns a list of (lo,hi) where lo and hi have units of energy of
    ranges to fit in for each energy in line_names.

    Args:
        line_names (list[str or float]): list or line names or energies
        excluded_line_names (list[str or float]): list of line_names or energies to
            avoid when making fit ranges
        approx_ecal: an EnergyCalibration object containing an approximate calibration
        fit_width_ev (float): full size in eV of fit ranges
    """
    _names, e_e = line_names_and_energies(line_names)
    _excl_names, excl_e_e = line_names_and_energies(excluded_line_names)
    half_width_ev = fit_width_ev / 2.0
    all_e = np.sort(np.hstack((e_e, excl_e_e)))
    assert (len(all_e) == len(np.unique(all_e)))
    fit_lo_hi_energy = []
    slopes_de_dph = []

    for e in e_e:
        slope_de_dph = approx_ecal.energy2dedph(e)
        if any(all_e < e):
            nearest_below = all_e[all_e < e][-1]
        else:
            nearest_below = -np.inf
        if any(all_e > e):
            nearest_above = all_e[all_e > e][0]
        else:
            nearest_above = np.inf
        lo = max(e - half_width_ev, (e + nearest_below) / 2.0)
        hi = min(e + half_width_ev, (e + nearest_above) / 2.0)
        fit_lo_hi_energy.append((lo, hi))
        slopes_de_dph.append(slope_de_dph)

    return e_e, fit_lo_hi_energy, slopes_de_dph


class FailedFit:
    def __init__(self, hist, bins):
        self.hist = hist
        self.bins = bins


class FailedToGetModelException(Exception):
    pass


def get_model(lineNameOrEnergy, has_linear_background=True, has_tails=False):
    if isinstance(lineNameOrEnergy, mass.GenericLineModel):
        line = lineNameOrEnergy.spect
    elif isinstance(lineNameOrEnergy, mass.SpectralLine):
        line = lineNameOrEnergy
    elif isinstance(lineNameOrEnergy, str):
        if lineNameOrEnergy in mass.spectra:
            line = mass.spectra[lineNameOrEnergy]
        elif lineNameOrEnergy in mass.STANDARD_FEATURES:
            energy = mass.STANDARD_FEATURES[lineNameOrEnergy]
            line = mass.SpectralLine.quick_monochromatic_line(lineNameOrEnergy, energy, 0.001, 0)
        else:
            raise FailedToGetModelException(
                f"failed to get line from lineNameOrEnergy={lineNameOrEnergy}")
    else:
        try:
            energy = float(lineNameOrEnergy)
        except Exception:
            raise FailedToGetModelException(
                f"lineNameOrEnergy = {lineNameOrEnergy} is not convertable to float or "
                "a str in mass.spectra or mass.STANDARD_FEATURES")
        line = mass.SpectralLine.quick_monochromatic_line(
            f"{lineNameOrEnergy}eV", float(lineNameOrEnergy), 0.001, 0)
    return line.model(has_linear_background=has_linear_background, has_tails=has_tails)


# support both names as they were both used historically
getmodel = get_model


def multifit(ph, line_names, fit_lo_hi, binsize_ev, slopes_de_dph, hide_deprecation=False):
    """
    Args:
        ph (np.array(dtype=float)): list of pulse heights
        line_names: names of calibration  lines
        fit_lo_hi (list[list[float]]): a list of (lo,hi) with units of ph, used as
            edges of histograms for fitting
        binsize_ev (list[float]): list of binsizes in eV for calibration lines
        slopes_de_dph (list[float]): - list of slopes de_dph (e in eV)
        hide_deprecation: whether to suppress deprecation warnings
    """
    name_e, e_e = line_names_and_energies(line_names)
    results = []
    peak_ph = []
    eres = []

    for i, name in enumerate(name_e):
        lo, hi = fit_lo_hi[i]
        dP_dE = 1 / slopes_de_dph[i]
        binsize_ph = binsize_ev[i] * dP_dE
        result = singlefit(ph, name, lo, hi, binsize_ph, dP_dE)
        results.append(result)
        peak_ph.append(result.best_values["peak_ph"])
        eres.append(result.best_values["fwhm"])
    return {"results": results, "peak_ph": peak_ph,
            "eres": eres, "line_names": name_e, "energies": e_e}


def singlefit(ph, name, lo, hi, binsize_ph, approx_dP_dE):
    nbins = (hi - lo) / binsize_ph
    if nbins > 5000:
        raise Exception("too damn many bins, dont like running out of memory")
    counts, bin_edges = np.histogram(ph, np.arange(lo, hi, binsize_ph))
    e = bin_edges[:-1] + 0.5 * (bin_edges[1] - bin_edges[0])
    model = getmodel(name)
    guess_params = model.guess(counts, bin_centers=e, dph_de=approx_dP_dE)
    if "Gaussian" not in model.name:
        guess_params["dph_de"].set(approx_dP_dE, vary=False)
    result = model.fit(counts, guess_params, bin_centers=e, minimum_bins_per_fwhm=1.5)
    result.energies = e
    return result
