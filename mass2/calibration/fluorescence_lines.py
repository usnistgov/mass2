# -*- coding: UTF-8 -*-
"""
fluorescence_lines.py

Tools for fitting and simulating X-ray fluorescence lines.
"""

from dataclasses import dataclass
from functools import cached_property
from typing import Optional
import numpy.typing as npt
from enum import Enum
import importlib.resources as pkg_resources

import numpy as np
import scipy as sp
import pylab as plt
import yaml

from . import line_models
from mass2.calibration.nist_xray_database import NISTXrayDBFile
from mass2.mathstat.special import voigt

import logging
LOG = logging.getLogger("mass")

FWHM_OVER_SIGMA = (8 * np.log(2))**0.5

_rng = np.random.default_rng()


def LineEnergies() -> dict[str, float]:
    """
    A dictionary to know a lot of x-ray fluorescence line energies, based on Deslattes' database.

    It is built on facts from mass2.calibration.nist_xray_database module.

    It is a dictionary from peak name to energy, with several alternate names
    for the lines:

    E = Energies()
    print E["MnKAlpha"]
    print E["MnKAlpha"], E["MnKA"], E["MnKA1"], E["MnKL3"]
    """
    db = NISTXrayDBFile()
    alternate_line_names = {v: k for (k, v) in db.LINE_NICKNAMES.items()}
    data = {}

    for fullname, L in db.lines.items():
        element, linename = fullname.split(" ", 1)
        allnames = [linename]
        if linename in alternate_line_names:
            siegbahn_linename = alternate_line_names[linename]
            long_linename = siegbahn_linename.replace("A", "Alpha"). \
                replace("B", "Beta").replace("G", "Gamma")

            allnames.append(siegbahn_linename)
            allnames.append(long_linename)

            if siegbahn_linename.endswith("1"):
                allnames.append(siegbahn_linename[:-1])
                allnames.append(long_linename[:-1])

        for name in allnames:
            key = "".join((element, name))
            data[key] = L.peak

    return data


# Some commonly-used standard energy features.
STANDARD_FEATURES = LineEnergies()


class AmplitudeType(Enum):
    LORENTZIAN_PEAK_HEIGHT = "Peak height of Lorentzians"
    LORENTZIAN_INTEGRAL_INTENSITY = "Integrated intensity of Lorentzians"
    VOIGT_PEAK_HEIGHT = "Peak height of Voigts"


spectra = {}


@dataclass(frozen=True)
class SpectralLine:
    """An abstract base class for modeling spectral lines as a sum
    of Voigt profiles (i.e., Gaussian-convolved Lorentzians).

    Call addline to create a new subclass properly.

    The API follows scipy.stats.stats.rv_continuous and is kind of like rv_frozen.
    Calling this object with an argument evalutes the pdf at the argument, it does not
    return an rv_frozen.

    But so far we ony define `rvs` and `pdf`.
    """

    element: str
    material: str
    linetype: str
    nominal_peak_energy: float
    energies: npt.NDArray[np.float64]
    lorentzian_fwhm: npt.NDArray[np.float64]
    reference_amplitude: npt.NDArray[np.float64]
    reference_amplitude_type: AmplitudeType = AmplitudeType.LORENTZIAN_INTEGRAL_INTENSITY
    reference_measurement_type: str = "unknown"
    intrinsic_sigma: float = 0.0
    reference_plot_instrument_gaussian_fwhm: float = 0.0
    reference_short: str = "unknown"
    position_uncertainty: float = 0.0
    is_default_material: bool = True

    @cached_property
    def peak_energy(self):
        try:
            peak_energy = sp.optimize.brent(
                lambda x: -self.pdf(x, instrument_gaussian_fwhm=0),
                brack=np.array((0.5, 1, 1.5)) * self.nominal_peak_energy)
        except ValueError:
            peak_energy = self.nominal_peak_energy
        return peak_energy

    @property
    def cumulative_amplitudes(self):
        return self.lorentzian_integral_intensity.cumsum()

    @cached_property
    def lorentzian_integral_intensity(self):
        if self.reference_amplitude_type == AmplitudeType.VOIGT_PEAK_HEIGHT:
            sigma = self.reference_plot_instrument_gaussian_fwhm / FWHM_OVER_SIGMA
            return np.array([
                ph / voigt(0, 0, fwhm / 2.0, sigma)
                for (ph, fwhm) in zip(self.reference_amplitude, self.lorentzian_fwhm)
            ])
        if self.reference_amplitude_type == AmplitudeType.LORENTZIAN_PEAK_HEIGHT:
            return self.reference_amplitude * (0.5 * np.pi * self.lorentzian_fwhm)

        if self.reference_amplitude_type == AmplitudeType.LORENTZIAN_INTEGRAL_INTENSITY:
            return self.reference_amplitude

    @cached_property
    def normalized_lorentzian_integral_intensity(self):
        x = self.lorentzian_integral_intensity
        return x / np.sum(x)

    @cached_property
    def lorentz_amplitude(self):
        return self.lorentzian_integral_intensity / self.lorentzian_fwhm

    def __call__(self, x, instrument_gaussian_fwhm):
        """Make the class callable, returning the same value as the self.pdf method."""
        return self.pdf(x, instrument_gaussian_fwhm)

    def pdf(self, x, instrument_gaussian_fwhm):
        """Spectrum (units of fraction per eV) as a function of <x>, the energy in eV"""
        gaussian_sigma = self._gaussian_sigma(instrument_gaussian_fwhm)
        x = np.asarray(x, dtype=float)
        result = np.zeros_like(x)
        for energy, fwhm, ampl in zip(self.energies, self.lorentzian_fwhm,
                                      self.lorentzian_integral_intensity):
            result += ampl * voigt(x, energy, hwhm=fwhm * 0.5, sigma=gaussian_sigma)
            # mass.voigt() is normalized to have unit integrated intensity
        return result

    def components(self, x, instrument_gaussian_fwhm):
        """List of spectrum components as a function of <x>, the energy in eV"""
        gaussian_sigma = self._gaussian_sigma(instrument_gaussian_fwhm)
        x = np.asarray(x, dtype=float)
        components = []
        for energy, fwhm, ampl in zip(self.energies, self.lorentzian_fwhm,
                                      self.lorentzian_integral_intensity):
            components.append(ampl * voigt(x, energy, hwhm=fwhm * 0.5, sigma=gaussian_sigma))
        return components

    def plot(self, x=None, instrument_gaussian_fwhm=0, axis=None, components=True, label=None, setylim=True):
        """Plot the spectrum.
        x - np array of energy in eV to plot at (sensible default)
        axis - axis to plot on (default creates new figure)
        components - True plots each voigt component in addition to the spectrum
        label - a string to label the plot with (optional)"""
        gaussian_sigma = self._gaussian_sigma(instrument_gaussian_fwhm)
        if x is None:
            width = max(2 * gaussian_sigma, 3 * np.amax(self.lorentzian_fwhm))
            lo = np.amin(self.energies) - width
            hi = np.amax(self.energies) + width
            x = np.linspace(lo, hi, 500)
        if axis is None:
            plt.figure()
            axis = plt.gca()
        if components:
            for component in self.components(x, instrument_gaussian_fwhm):
                axis.plot(x, component, "--")
        pdf = self.pdf(x, instrument_gaussian_fwhm)
        axis.plot(x, pdf, "k", lw=2, label=label)
        axis.set_xlabel("Energy (eV)")
        axis.set_ylabel(f"Counts per {float(x[1] - x[0]):.2} eV bin")
        axis.set_xlim(x[0], x[-1])
        if setylim:
            axis.set_ylim(0, np.amax(pdf) * 1.05)
        axis.set_title(f"{self.shortname} with resolution {instrument_gaussian_fwhm:.2f} eV FWHM")
        return axis

    def plot_like_reference(self, axis=None):
        if self.reference_plot_instrument_gaussian_fwhm is None:
            fwhm = 0.001
        else:
            fwhm = self.reference_plot_instrument_gaussian_fwhm
        axis = self.plot(
            axis=axis, instrument_gaussian_fwhm=fwhm)
        return axis

    def rvs(self, size, instrument_gaussian_fwhm, rng=None):
        """The CDF and PPF (cumulative distribution and percentile point functions) are hard to
        compute.  But it's easy enough to generate the random variates themselves, so we
        override that method."""
        if rng is None:
            rng = _rng
        gaussian_sigma = self._gaussian_sigma(instrument_gaussian_fwhm)
        # Choose from among the N Lorentzian lines in proportion to the line amplitudes
        iline = self.cumulative_amplitudes.searchsorted(
            rng.uniform(0, self.cumulative_amplitudes[-1], size=size))
        # Choose Lorentzian variates of the appropriate width (but centered on 0)
        lor = rng.standard_cauchy(size=size) * self.lorentzian_fwhm[iline] * 0.5
        # If necessary, add a Gaussian variate to mimic finite resolution
        if gaussian_sigma > 0.0:
            lor += rng.standard_normal(size=size) * gaussian_sigma
        # Finally, add the line centers.
        results = lor + self.energies[iline]
        # We must check for non-positive results and replace them by recursive call
        # to self.rvs().
        not_positive = results <= 0.0
        if np.any(not_positive):
            Nbad = not_positive.sum()
            results[not_positive] = self.rvs(
                size=Nbad, instrument_gaussian_fwhm=instrument_gaussian_fwhm)
        return results

    @property
    def shortname(self):
        if self.is_default_material:
            return f"{self.element}{self.linetype}"
        else:
            return f"{self.element}{self.linetype}_{self.material}"

    @property
    def reference(self):
        return lineshape_references[self.reference_short]

    def _gaussian_sigma(self, instrument_gaussian_fwhm):
        """combined intrinstic_sigma and insturment_gaussian_fwhm in quadrature and return the result
        """
        assert instrument_gaussian_fwhm >= 0
        return ((instrument_gaussian_fwhm / FWHM_OVER_SIGMA)**2 + self.intrinsic_sigma**2)**0.5

    def __repr__(self):
        return f"SpectralLine: {self.shortname}"

    def model(self, has_linear_background=True, has_tails=False, prefix="", qemodel=None):
        """Generate a LineModel instance from a SpectralLine"""
        model_class = line_models.GenericLineModel
        name = f"{self.element}{self.linetype}"
        m = model_class(name=name, spect=self, has_linear_background=has_linear_background,
                        has_tails=has_tails, prefix=prefix, qemodel=qemodel)
        return m

    def fitter(self):
        fitter_class = line_models.GenericLineModel
        f = fitter_class(self)
        f.name = f"{self.element}{self.linetype}"
        return f

    def minimum_fwhm(self, instrument_gaussian_fwhm):
        """for the narrowest lorentzian in the line model, calculate the combined fwhm including
        the lorentzian, intrinstic_sigma, and instrument_gaussian_fwhm"""
        fwhm2 = np.amin(self.lorentzian_fwhm)**2 + instrument_gaussian_fwhm**2 + \
            (self.intrinsic_sigma * FWHM_OVER_SIGMA)**2
        return np.sqrt(fwhm2)

    @classmethod
    def quick_monochromatic_line(cls, name, energy, lorentzian_fwhm, intrinsic_sigma=0.0):
        """
        Create a quick monochromatic line. Intended for use in calibration when we know a line energy, but not a lineshape model.
        Returns and instrance of SpectralLine with most fields having contents like "unknown: quick_line". The line will have
        a single lorentzian element with the given energy, fwhm, and intrinsic_sigma values.
        """
        energy = float(energy)
        element = name
        material = "unknown: quick_line"
        energies = np.array([energy])
        if lorentzian_fwhm <= 0 and intrinsic_sigma <= 0:
            intrinsic_sigma = 1e-4
        lorentzian_fwhm = np.array([lorentzian_fwhm])
        linetype = "Gaussian"
        reference_short = "unkown: quick_line"
        reference_amplitude = np.array([1.0])
        reference_amplitude_type = AmplitudeType.LORENTZIAN_INTEGRAL_INTENSITY
        nominal_peak_energy = energy
        position_uncertainty = 0.0
        reference_measurement_type = "unkown: quick_line"
        return cls(
            element=element,
            material=material,
            linetype=linetype,
            energies=energies,
            lorentzian_fwhm=lorentzian_fwhm,
            intrinsic_sigma=intrinsic_sigma,
            reference_short=reference_short,
            reference_amplitude=reference_amplitude,
            reference_amplitude_type=reference_amplitude_type,
            nominal_peak_energy=nominal_peak_energy,
            position_uncertainty=position_uncertainty,
            reference_measurement_type=reference_measurement_type)

    @classmethod
    def addline(cls, element, linetype, material, reference_short, reference_plot_instrument_gaussian_fwhm,  # noqa: PLR0917
                nominal_peak_energy, energies, lorentzian_fwhm, reference_amplitude,
                reference_amplitude_type, ka12_energy_diff=None,
                position_uncertainty=np.nan, intrinsic_sigma=0, reference_measurement_type=None,
                is_default_material=True, allow_replacement=False):

        # require exactly one method of specifying the amplitude of each component
        assert reference_amplitude_type in {
            AmplitudeType.LORENTZIAN_PEAK_HEIGHT,
            AmplitudeType.LORENTZIAN_INTEGRAL_INTENSITY,
            AmplitudeType.VOIGT_PEAK_HEIGHT
        }
        # require the reference exists in lineshape_references
        assert reference_short in lineshape_references

        # require kalpha lines to have ka12_energy_diff
        if linetype.startswith("KAlpha"):
            ka12_energy_diff = float(ka12_energy_diff)
        # require reference_plot_instrument_gaussian_fwhm to be a float or None
        assert reference_plot_instrument_gaussian_fwhm is None or isinstance(
            reference_plot_instrument_gaussian_fwhm, float)

        line = cls(
            element=element,
            material=material,
            linetype=linetype,
            nominal_peak_energy=float(nominal_peak_energy),
            energies=np.array(energies),
            lorentzian_fwhm=np.array(lorentzian_fwhm),
            reference_amplitude=reference_amplitude,
            reference_amplitude_type=reference_amplitude_type,
            reference_measurement_type=reference_measurement_type,
            intrinsic_sigma=intrinsic_sigma,
            reference_plot_instrument_gaussian_fwhm=reference_plot_instrument_gaussian_fwhm,
            reference_short=reference_short,
            position_uncertainty=float(position_uncertainty),
            is_default_material=is_default_material,
        )
        name = line.shortname
        if name in spectra.keys() and (not allow_replacement):
            raise ValueError(f"spectrum {name} already exists")

        # Add this SpectralLine to spectra dict AND make it be a variable in the module
        spectra[name] = line
        globals()[name] = line
        return line


@dataclass(frozen=True)
class LineshapeReference:
    """Description of our source of information on a line shape. Might be a reference to the literature,
    or notes on conversations. They are stored in a YAML file mass2/data/fluorescence_line_references.yaml
    """
    tag: str
    description: str
    url: str

    @classmethod
    def load(cls, filename: Optional[str] = None) -> dict:
        references = {}
        if filename is None:
            filename = pkg_resources.files("mass2").joinpath("data", "fluorescence_line_references.yaml")
        with open(filename, "r", encoding="utf-8") as file:
            d = yaml.safe_load(file)
            for item in d:
                url = item.get("URL", "")
                references[item["tag"]] = LineshapeReference(item["tag"], item["description"], url)
        return references

    def __str__(self) -> str:
        lines = [f'lineshape_references["{self.tag}"]:']
        lines.append(self.description.rstrip("\n"))
        if len(self.url) > 1:
            lines.append(f"url: {self.url}")
        return "\n".join(lines)


lineshape_references = LineshapeReference.load()

addline = SpectralLine.addline


addline(
    element="Fe",
    material="metal",
    linetype="LAlpha",
    reference_short="Rough Estimate",
    nominal_peak_energy=STANDARD_FEATURES["FeLAlpha"],
    energies=np.array([STANDARD_FEATURES["FeLAlpha"], STANDARD_FEATURES["FeLBeta"]]),
    lorentzian_fwhm=np.array(np.array([3, 3])),
    reference_amplitude=np.array(np.array([2, 1])),
    reference_amplitude_type=AmplitudeType.LORENTZIAN_PEAK_HEIGHT,
    ka12_energy_diff=np.abs(STANDARD_FEATURES["FeLAlpha"] - STANDARD_FEATURES["FeLBeta"]),
    reference_plot_instrument_gaussian_fwhm=0.2,  # a total guess
    position_uncertainty=1.5
)

addline(
    element="Mg",
    material="metal",
    linetype="KAlpha",
    reference_short="Menesguen 2022",
    nominal_peak_energy=1253.635,
    energies=np.array((1253.666, 1253.401, 1253.422, 1258.457, 1261.887,
                       1262.156, 1263.758, 1271.034, 1272.601, 1274.187)),
    lorentzian_fwhm=np.array((.380, .374, .859, 1.006, .485,
                              .585, .797, 1.213, .88, .852)),
    reference_amplitude=np.array((100, 49.75, 19.7, 1.99, 8,
                                  6, 8.32, 1.23, .23, .71)),
    reference_amplitude_type=AmplitudeType.LORENTZIAN_INTEGRAL_INTENSITY,
    ka12_energy_diff=0.266,
    reference_plot_instrument_gaussian_fwhm=0.2,  # a total guess
    position_uncertainty=0.016
)

addline(
    element="Al",
    material="metal",
    linetype="KAlpha",
    reference_short="Menesguen 2022",
    nominal_peak_energy=1486.690,
    energies=np.array((1486.709, 1486.305, 1486.314, 1492.305, 1496.227,
                       1496.635, 1498.397, 1506.719, 1510.212)),
    lorentzian_fwhm=np.array((.392, .392, 1.12, 1.16, .647,
                              .685, .932, 1.49, 1.07)),
    reference_amplitude=np.array((100, 49.4, 16.1, 1.21, 7.54,
                                  3.7, 5.77, 0.70, 0.51)),
    reference_amplitude_type=AmplitudeType.LORENTZIAN_INTEGRAL_INTENSITY,
    ka12_energy_diff=0.404,
    reference_plot_instrument_gaussian_fwhm=0.2,  # a total guess
    position_uncertainty=0.005
)

addline(
    element="Si",
    material="Si wafer",
    linetype="KAlpha",
    reference_short="Menesguen 2022",
    nominal_peak_energy=1739.961,
    energies=np.array((1739.973, 1739.361, 1739.537, 1746.330, 1750.687,
                       1751.276, 1753.113, 1762.677, 1766.607)),
    lorentzian_fwhm=np.array((.441, .458, 1.12, 1.18, .80, .74, .931, 1.83, 1.07)),
    reference_amplitude=np.array((100, 50.35, 12.41, .78, 5.65, 2.03, 4.45, 0.29, 0.18)),
    reference_amplitude_type=AmplitudeType.LORENTZIAN_INTEGRAL_INTENSITY,
    ka12_energy_diff=.612,
    reference_plot_instrument_gaussian_fwhm=0.2,  # a total guess
    position_uncertainty=0.010
)


# With the publication of Ménesguen 2022, we want to keep our previous line shapes around for
# cross-checking. You can find them as, e.g., mass.MgKAlpha_v1
addline(
    element="Mg",
    material="metal_v1",
    linetype="KAlpha",
    reference_short="Klauber 1993",
    reference_plot_instrument_gaussian_fwhm=None,
    nominal_peak_energy=1253.687,
    energies=np.array((-.265, 0, 4.740, 8.210, 8.487, 10.095, 17.404, 20.430)) + 1253.687,
    lorentzian_fwhm=np.array((.541, .541, 1.1056, .6264, .7349, 1.0007, 1.4311, .8656)),
    reference_amplitude=np.array((0.5, 1, .02099, .07868, .04712, .09071, .01129, .00538)),
    reference_amplitude_type=AmplitudeType.LORENTZIAN_INTEGRAL_INTENSITY,
    ka12_energy_diff=2.2,
    is_default_material=False
)

addline(
    element="Al",
    material="metal_v1992",
    linetype="KAlpha",
    reference_short="Schweppe 1992 Al",
    reference_plot_instrument_gaussian_fwhm=None,
    nominal_peak_energy=1486.88931733,
    energies=np.array((1486.706, 1486.293, 1492.3, 1496.4, 1498.4)),
    lorentzian_fwhm=np.array((0.43, 0.43, 1.34, 0.96, 1.255)),
    reference_amplitude=np.array((1, .5, .02, .12, .06)),
    reference_amplitude_type=AmplitudeType.LORENTZIAN_INTEGRAL_INTENSITY,
    ka12_energy_diff=3.0,
    position_uncertainty=0.010,
    is_default_material=False
)

addline(
    element="Al",
    material="AlO",
    linetype="KAlpha",
    reference_short="Wollman 2000",
    reference_plot_instrument_gaussian_fwhm=None,
    nominal_peak_energy=1486.930456,
    energies=np.array((1486.94, 1486.52, 1492.94, 1496.85, 1498.70, 1507.4, 1510.9)),
    lorentzian_fwhm=np.array((0.43, 0.43, 1.34, 0.96, 1.25, 1.5, 0.9)),
    reference_amplitude=np.array((1.0, 0.5, 0.033, 0.12, 0.11, 0.07, 0.05)),
    reference_amplitude_type=AmplitudeType.LORENTZIAN_PEAK_HEIGHT,
    ka12_energy_diff=3.,
    is_default_material=False
)

addline(
    element="Si",
    material="Si crystal",
    linetype="KAlpha",
    reference_short="Deslattes Notebook Si",
    reference_plot_instrument_gaussian_fwhm=0.245,
    nominal_peak_energy=1739.986,
    energies=np.array((1739.39, 1739.986, 1752.0)),
    lorentzian_fwhm=np.array((0.539, 0.524, 5)),
    reference_amplitude=np.array((3.134e2, 6.121e3, 8e2)),
    reference_amplitude_type=AmplitudeType.LORENTZIAN_INTEGRAL_INTENSITY,
    ka12_energy_diff=.6,
    position_uncertainty=0.040,
    is_default_material=False
)

addline(
    element="S",
    material="MoS2 spray",
    linetype="KAlpha",
    reference_short="Deslattes Notebook S, Cl, K",
    reference_plot_instrument_gaussian_fwhm=0.2414,
    nominal_peak_energy=2307.89,
    energies=np.array((2307.89, 2306.70)),
    lorentzian_fwhm=np.array((0.769, 0.722)),
    reference_amplitude=np.array((0.11951e5, 0.61114e4)),
    reference_amplitude_type=AmplitudeType.LORENTZIAN_INTEGRAL_INTENSITY,
    ka12_energy_diff=1.19,
    position_uncertainty=0.040
)

addline(
    element="Cl",
    material="KCl crystal",
    linetype="KAlpha",
    reference_short="Deslattes Notebook S, Cl, K",
    reference_plot_instrument_gaussian_fwhm=0.266,
    nominal_peak_energy=2622.44,
    energies=np.array((2622.44, 2620.85)),
    lorentzian_fwhm=np.array((0.925, 0.945)),
    reference_amplitude=np.array((0.15153e5, 0.82429e4)),
    reference_amplitude_type=AmplitudeType.LORENTZIAN_INTEGRAL_INTENSITY,
    ka12_energy_diff=1.6,
    position_uncertainty=0.040
)

addline(
    element="K",
    material="KCl crystal",
    linetype="KAlpha",
    reference_short="Deslattes Notebook S, Cl, K",
    reference_plot_instrument_gaussian_fwhm=0.0896,
    nominal_peak_energy=3313.93,
    energies=np.array((3313.93, 3311.17)),
    lorentzian_fwhm=np.array((0.948, 0.939)),
    reference_amplitude=np.array((0.15153e5, 0.82429e4)),
    reference_amplitude_type=AmplitudeType.LORENTZIAN_INTEGRAL_INTENSITY,
    ka12_energy_diff=2.75,
    position_uncertainty=0.020
)

addline(
    element="Sc",
    material="metal",
    linetype="KAlpha",
    reference_short="Dean 2019",
    reference_plot_instrument_gaussian_fwhm=1.9,
    nominal_peak_energy=4090.699,
    energies=np.array((4090.709, 4089.418, 4087.752, 4093.508, 4085.918, 4083.926)),  # Table 4 C_i
    lorentzian_fwhm=np.array((1.15, 2.89, 1.02, 2.01, 1.40, 3.86)),  # Table 4 W_i
    reference_amplitude=np.array((501, 107, 15, 43, 321, 14)),  # Table 4 Fraction
    reference_amplitude_type=AmplitudeType.LORENTZIAN_INTEGRAL_INTENSITY,
    ka12_energy_diff=4.773,
    position_uncertainty=0.010
)

# Keep Chantler 2006 version of scandium in MASS just in case. At resolution 1.9 eV, it differs
# by up to ±10% on either wing of the doublet (around 4082 and 4094 eV).
addline(
    element="Sc",
    material="metal_2006",
    linetype="KAlpha",
    is_default_material=False,
    reference_short="Chantler 2006",
    reference_plot_instrument_gaussian_fwhm=0.52,
    nominal_peak_energy=4090.735,
    energies=np.array((4090.745, 4089.452, 4087.782, 4093.547, 4085.941, 4083.976)),  # Table I C_i
    lorentzian_fwhm=np.array((1.17, 2.65, 1.41, 2.09, 1.53, 3.49)),  # Table I W_i
    reference_amplitude=np.array((8175, 878, 232, 287, 4290, 119)),  # Table I A_i
    reference_amplitude_type=AmplitudeType.VOIGT_PEAK_HEIGHT,
    ka12_energy_diff=5.1,
    position_uncertainty=0.019  # table 3
)

addline(
    element="Sc",
    material="metal",
    linetype="KBeta",
    reference_short="Dean 2020",
    reference_plot_instrument_gaussian_fwhm=1.97,
    nominal_peak_energy=4460.845,
    energies=np.array((4460.972, 4459.841, 4458.467, 4455.979, 4463.544)),  # Table 3 C_i
    lorentzian_fwhm=np.array((1.22, 1.58, 2.51, 4.64, 2.10)),  # Table 3 W_i
    reference_amplitude=np.array((530, 234, 149, 43, 42)),  # Table 3 Fraction
    reference_amplitude_type=AmplitudeType.LORENTZIAN_INTEGRAL_INTENSITY,
    position_uncertainty=0.0092
)

addline(
    # The paper has two sets of TiKAlpha data, I used the set Refit of [21] Kawai et al 1994
    element="Ti",
    material="metal",
    linetype="KAlpha",
    reference_short="Chantler 2006",
    reference_plot_instrument_gaussian_fwhm=0.11,
    nominal_peak_energy=4510.903,
    energies=np.array((4510.918, 4509.954, 4507.763, 4514.002, 4504.910, 4503.088)),  # Table I C_i
    lorentzian_fwhm=np.array((1.37, 2.22, 3.75, 1.70, 1.88, 4.49)),  # Table I W_i
    reference_amplitude=np.array((4549, 626, 236, 143, 2034, 54)),  # Table I A_i
    reference_amplitude_type=AmplitudeType.VOIGT_PEAK_HEIGHT,
    ka12_energy_diff=6.0,
    position_uncertainty=0.003  # table 3, based on difference between two entries
)

addline(
    element="Ti",
    material="metal",
    linetype="KBeta",
    reference_short="Chantler 2013",
    reference_plot_instrument_gaussian_fwhm=1.244 * 2.3548,
    nominal_peak_energy=4931.966,
    energies=np.array((25.37, 30.096, 31.967, 35.59)) + 4900,
    lorentzian_fwhm=np.array((16.3, 4.25, 0.42, 0.47)),
    reference_amplitude=np.array((199, 455, 326, 19.2)),
    reference_amplitude_type=AmplitudeType.LORENTZIAN_INTEGRAL_INTENSITY,
    position_uncertainty=0.022  # figure 6 and seciton 3.1 text

)

addline(
    element="V",
    material="metal",
    linetype="KAlpha",
    reference_short="Chantler 2006",
    reference_plot_instrument_gaussian_fwhm=1.99,  # Table I, other parameters
    nominal_peak_energy=4952.216,
    energies=np.array((4952.237, 4950.656, 4948.266, 4955.269, 4944.672, 4943.014)),  # Table I C_i
    lorentzian_fwhm=np.array((1.45, 2.00, 1.81, 1.76, 2.94, 3.09)),  # Table I W_i
    reference_amplitude=np.array((25832, 5410, 1536, 956, 12971, 603)),  # Table I A_i
    reference_amplitude_type=AmplitudeType.VOIGT_PEAK_HEIGHT,
    ka12_energy_diff=7.5,
    position_uncertainty=0.059  # table 3
)

addline(
    element="V",
    material="metal",
    linetype="KBeta",
    reference_short="Chantler 2013, Section 5",
    reference_plot_instrument_gaussian_fwhm=0.805 * 2.3548,
    nominal_peak_energy=5426.956,
    energies=np.array((18.19, 24.50, 26.992)) + 5400,
    lorentzian_fwhm=np.array((18.86, 5.48, 2.499)),
    reference_amplitude=np.array((258, 236, 507)),
    reference_amplitude_type=AmplitudeType.LORENTZIAN_INTEGRAL_INTENSITY,
)

addline(
    element="Cr",
    material="metal",
    linetype="KAlpha",
    reference_short="Hoelzer 1997, NISTfits.ipf",
    reference_plot_instrument_gaussian_fwhm=None,
    nominal_peak_energy=5414.81,
    energies=5400 + np.array((14.874, 14.099, 12.745, 10.583, 18.304, 5.551, 3.986)),
    lorentzian_fwhm=np.array((1.457, 1.760, 3.138, 5.149, 1.988, 2.224, 4.740)),
    reference_amplitude=np.array((822, 237, 85, 45, 15, 386, 36)),
    reference_amplitude_type=AmplitudeType.LORENTZIAN_PEAK_HEIGHT,
    ka12_energy_diff=9.2,
    position_uncertainty=0.010  # table 4
)

addline(
    element="Cr",
    material="metal",
    linetype="KBeta",
    reference_short="Hoelzer 1997",
    reference_plot_instrument_gaussian_fwhm=None,
    nominal_peak_energy=5946.82,
    energies=5900 + np.array((47.00, 35.31, 46.24, 42.04, 44.93)),  # Table III E_i
    lorentzian_fwhm=np.array((1.70, 15.98, 1.90, 6.69, 3.37)),  # Table III W_i
    reference_amplitude=np.array((670, 55, 337, 82, 151)),  # Table III I_I
    reference_amplitude_type=AmplitudeType.LORENTZIAN_PEAK_HEIGHT,
    position_uncertainty=0.010  # table 4
)

addline(
    element="Mn",
    material="metal",
    linetype="KAlpha",
    reference_short="Hoelzer 1997, NISTfits.ipf",
    reference_plot_instrument_gaussian_fwhm=None,
    nominal_peak_energy=5898.802,
    energies=5800 + np.array((98.853, 97.867, 94.829, 96.532, 99.417, 102.712, 87.743, 86.495)),
    lorentzian_fwhm=np.array((1.715, 2.043, 4.499, 2.663, 0.969, 1.553, 2.361, 4.216)),
    reference_amplitude=np.array((790, 264, 68, 96, 71, 10, 372, 100)),
    reference_amplitude_type=AmplitudeType.LORENTZIAN_PEAK_HEIGHT,
    ka12_energy_diff=11.1,
    position_uncertainty=0.010  # table 4
)

addline(
    element="Mn",
    material="metal",
    linetype="KBeta",
    reference_short="Hoelzer 1997",
    reference_plot_instrument_gaussian_fwhm=None,
    nominal_peak_energy=6490.18,
    energies=6400 + np.array((90.89, 86.31, 77.73, 90.06, 88.83)),  # Table III E_i
    lorentzian_fwhm=np.array((1.83, 9.40, 13.22, 1.81, 2.81)),  # Table III W_i
    reference_amplitude=np.array((608, 109, 77, 397, 176)),  # Table III I_I
    reference_amplitude_type=AmplitudeType.LORENTZIAN_PEAK_HEIGHT,
    position_uncertainty=0.010  # table 4
)

addline(
    element="Fe",
    material="metal",
    linetype="KAlpha",
    reference_short="Hoelzer 1997, NISTfits.ipf",
    reference_plot_instrument_gaussian_fwhm=None,
    nominal_peak_energy=6404.01,
    energies=np.array((6404.148, 6403.295, 6400.653, 6402.077, 6391.190, 6389.106, 6390.275)),
    lorentzian_fwhm=np.array((1.613, 1.965, 4.833, 2.803, 2.487, 4.339, 2.57)),
    reference_amplitude=np.array((697, 376, 88, 136, 339, 60, 102)),
    reference_amplitude_type=AmplitudeType.LORENTZIAN_PEAK_HEIGHT,
    ka12_energy_diff=13.0,
    position_uncertainty=0.010  # table 4
)
# ERROR IN HOLZER PAPER:
# The FWHM in the Table II of Holzer have the Kalpha_22 and _23 widths as 2.339 and 4.433, but
# these disagree with their Figure 1c. Swapping the widths (as you see above) makes the curve
# match the figure, though the exact I_int values still don't match those in Table II.
# To get the I_int values close, we choose width 4.339 and 2.57. These are still just a guess,
# but that's where the above values come from. See conversation with Richard Gnewkow of the
# Technische Universität Berlin on April 23-24, 2020.

addline(
    element="Fe",
    material="metal",
    linetype="KBeta",
    reference_short="Hoelzer 1997",
    reference_plot_instrument_gaussian_fwhm=None,
    nominal_peak_energy=7058.18,
    energies=np.array((7046.90, 7057.21, 7058.36, 7054.75)),  # Table III E_i
    lorentzian_fwhm=np.array((14.17, 3.12, 1.97, 6.38)),  # Table III W_i
    reference_amplitude=np.array((107, 448, 615, 141)),  # Table III I_I
    reference_amplitude_type=AmplitudeType.LORENTZIAN_PEAK_HEIGHT,
    position_uncertainty=0.030  # table 4
)

addline(
    element="Co",
    material="metal",
    linetype="KAlpha",
    reference_short="Hoelzer 1997, NISTfits.ipf",
    reference_plot_instrument_gaussian_fwhm=None,
    nominal_peak_energy=6930.38,
    energies=np.array((6930.425, 6929.388, 6927.676, 6930.941, 6915.713, 6914.659, 6913.078)),
    lorentzian_fwhm=np.array((1.795, 2.695, 4.555, 0.808, 2.406, 2.773, 4.463)),
    reference_amplitude=np.array((809, 205, 107, 41, 314, 131, 43)),
    reference_amplitude_type=AmplitudeType.LORENTZIAN_PEAK_HEIGHT,
    ka12_energy_diff=15.0,
    position_uncertainty=0.010  # table 4
)
# Notice that Co KAlpha in Holzer shows the 4th line as having integrated intensity of 0.088, but
# this is probably a typo (should read 0.008). No effect on the data above, though, because it's a
# derived quantity.

addline(
    element="Co",
    material="metal",
    linetype="KBeta",
    reference_short="Hoelzer 1997",
    reference_plot_instrument_gaussian_fwhm=None,
    nominal_peak_energy=7649.45,
    energies=np.array((7649.60, 7647.83, 7639.87, 7645.49, 7636.21, 7654.13)),  # Table III E_i
    lorentzian_fwhm=np.array((3.05, 3.58, 9.78, 4.89, 13.59, 3.79)),  # Table III W_i
    reference_amplitude=np.array((798, 286, 85, 114, 33, 35)),  # Table III I_I
    reference_amplitude_type=AmplitudeType.LORENTZIAN_PEAK_HEIGHT,
    position_uncertainty=0.010  # table 4
)

addline(
    element="Ni",
    material="metal",
    linetype="KAlpha",
    reference_short="Hoelzer 1997, NISTfits.ipf",
    reference_plot_instrument_gaussian_fwhm=None,
    nominal_peak_energy=7478.26,
    energies=np.array((7478.281, 7476.529, 7461.131, 7459.874, 7458.029)),
    lorentzian_fwhm=np.array((2.013, 4.711, 2.674, 3.039, 4.476)),
    reference_amplitude=np.array((909, 136, 351, 79, 24)),
    reference_amplitude_type=AmplitudeType.LORENTZIAN_PEAK_HEIGHT,
    ka12_energy_diff=17.2,
    position_uncertainty=0.010  # table 4
)

addline(
    element="Ni",
    material="metal",
    linetype="KBeta",
    reference_short="Hoelzer 1997",
    reference_plot_instrument_gaussian_fwhm=None,
    nominal_peak_energy=8264.78,
    energies=np.array((8265.01, 8263.01, 8256.67, 8268.70)),  # Table III E_i
    lorentzian_fwhm=np.array((3.76, 4.34, 13.70, 5.18)),  # Table III W_i
    reference_amplitude=np.array((722, 358, 89, 104)),  # Table III I_I
    reference_amplitude_type=AmplitudeType.LORENTZIAN_PEAK_HEIGHT,
    position_uncertainty=0.010  # table 4
)

addline(
    element="Cu",
    material="metal",
    linetype="KAlpha",
    reference_short="Hoelzer 1997",
    reference_plot_instrument_gaussian_fwhm=None,
    nominal_peak_energy=8047.83,
    energies=np.array((8047.8372, 8045.3672, 8027.9935, 8026.5041)),
    lorentzian_fwhm=np.array((2.285, 3.358, 2.667, 3.571)),
    reference_amplitude=np.array((957, 90, 334, 111)),
    reference_amplitude_type=AmplitudeType.LORENTZIAN_PEAK_HEIGHT,
    ka12_energy_diff=20.0,
    position_uncertainty=0.010  # table 4
)

addline(
    element="Cu",
    material="metal",
    linetype="KBeta",
    reference_short="Hoelzer 1997",
    reference_plot_instrument_gaussian_fwhm=None,
    nominal_peak_energy=8905.42,
    energies=np.array((8905.532, 8903.109, 8908.462, 8897.387, 8911.393)),  # Table III E_i
    lorentzian_fwhm=np.array((3.52, 3.52, 3.55, 8.08, 5.31)),  # Table III W_i
    reference_amplitude=np.array((757, 388, 171, 68, 55)),  # Table III I_I
    reference_amplitude_type=AmplitudeType.LORENTZIAN_PEAK_HEIGHT,
    position_uncertainty=0.030  # table 4
)

addline(
    element="Zn",
    material="metal",
    linetype="KAlpha",
    reference_short="Zn Hack",
    reference_plot_instrument_gaussian_fwhm=None,
    nominal_peak_energy=8638.91,
    energies=[8638.8872, 8636.4172, 8615.9835, 8614.4941],
    lorentzian_fwhm=np.array((2.285, 3.358, 2.667, 3.571)) * 1.1,
    reference_amplitude=np.array((957, 90, 334, 111)),
    reference_amplitude_type=AmplitudeType.LORENTZIAN_PEAK_HEIGHT,
    ka12_energy_diff=23.0,
)

addline(
    element="Zn",
    material="metal",
    linetype="KBeta",
    reference_short="Zn Hack",
    reference_plot_instrument_gaussian_fwhm=None,
    nominal_peak_energy=9572.03,
    energies=np.array((8905.532, 8903.109, 8908.462, 8897.387, 8911.393)) * 1.06 + 133.85,
    lorentzian_fwhm=np.array((3.52, 3.52, 3.55, 8.08, 5.31)) * 1.06,
    reference_amplitude=np.array((757, 388, 171, 68, 55)),  # Table III I_I
    reference_amplitude_type=AmplitudeType.LORENTZIAN_PEAK_HEIGHT,
)

addline(
    element="Se",
    material="metal",
    linetype="KAlpha",
    reference_short="Ito 2020",  # Table III
    reference_plot_instrument_gaussian_fwhm=0.17,
    nominal_peak_energy=11222.383,
    energies=np.array((11222.380, 11217.573, 11181.48, 11178.55)),
    lorentzian_fwhm=np.array((3.633, 3.53, 3.579, 3.02)),
    reference_amplitude=np.array((100, 1.68, 50.83, 1.80)),
    ka12_energy_diff=40.944,
    reference_amplitude_type=AmplitudeType.LORENTZIAN_INTEGRAL_INTENSITY,
)


addline(
    element="Se",
    material="metal",
    linetype="KBeta",
    reference_short="Ito 2020",  # Table IV
    reference_plot_instrument_gaussian_fwhm=0.17,
    nominal_peak_energy=12495.911,
    energies=np.array((12495.911, 12490.094, 12503.11, 12652.840)),
    lorentzian_fwhm=np.array((4.285, 5.70, 6.25, 4.81)),
    reference_amplitude=np.array((100, 63.3, 8.1, 5.24)),
    reference_amplitude_type=AmplitudeType.LORENTZIAN_INTEGRAL_INTENSITY,
)


addline(
    element="Br",
    material="metal",
    linetype="KAlpha",
    reference_short="Steve Smith",
    reference_plot_instrument_gaussian_fwhm=None,
    nominal_peak_energy=11924.36,
    energies=np.array((11924.2, 11877.6)),
    lorentzian_fwhm=np.array((3.60, 3.73)),
    reference_amplitude=np.array((2, 1)),
    ka12_energy_diff=46.6,
    reference_amplitude_type=AmplitudeType.LORENTZIAN_PEAK_HEIGHT,
)


addline(
    element="Y",
    material="metal",
    linetype="KAlpha",
    reference_short="Ito 2020",  # Table III
    reference_plot_instrument_gaussian_fwhm=0.21,
    nominal_peak_energy=14958.389,
    energies=np.array((14958.389, 14883.403)),
    lorentzian_fwhm=np.array((5.464, 5.393)),
    reference_amplitude=np.array((100, 52.23)),
    ka12_energy_diff=74.986,
    reference_amplitude_type=AmplitudeType.LORENTZIAN_INTEGRAL_INTENSITY,
)


addline(
    element="Y",
    material="metal",
    linetype="KBeta",
    reference_short="Ito 2020",  # Table IV
    reference_plot_instrument_gaussian_fwhm=0.21,
    nominal_peak_energy=16737.89,
    energies=np.array((16737.88, 16726.02, 16746.2, 17010.57)),
    lorentzian_fwhm=np.array((5.60, 5.53, 17.0, 5.80)),
    reference_amplitude=np.array((100, 52.6, 1.80, 1.68)),
    reference_amplitude_type=AmplitudeType.LORENTZIAN_INTEGRAL_INTENSITY,
)


addline(
    element="Zr",
    material="metal",
    linetype="KAlpha",
    reference_short="Ito 2020",  # Table III
    reference_plot_instrument_gaussian_fwhm=0.22,
    nominal_peak_energy=15774.87,
    energies=np.array((15774.87, 15690.77)),
    lorentzian_fwhm=np.array((5.865, 5.845)),
    reference_amplitude=np.array((100, 52.53)),
    ka12_energy_diff=84.10,
    reference_amplitude_type=AmplitudeType.LORENTZIAN_INTEGRAL_INTENSITY,
)


addline(
    element="Zr",
    material="metal",
    linetype="KBeta",
    reference_short="Ito 2020",  # Table IV
    reference_plot_instrument_gaussian_fwhm=0.22,
    nominal_peak_energy=17666.578,
    energies=np.array((17667.78, 17654.31, 17680)),
    # The last (Kb'') energy is reported as 15774.87(31), which is clearly a typo.
    # We estimate 17680 by reading Figure 2.
    lorentzian_fwhm=np.array((6.171, 5.89, 10.8)),
    reference_amplitude=np.array((100, 50.49, 5.2)),
    reference_amplitude_type=AmplitudeType.LORENTZIAN_INTEGRAL_INTENSITY,
)


addline(
    element="W",
    material="metal",
    linetype="LAlpha",
    reference_short="Joe Fowler",
    reference_plot_instrument_gaussian_fwhm=None,
    nominal_peak_energy=8398.24,
    energies=np.array((8335.34, 8398.24)),
    lorentzian_fwhm=np.array((6.97, 7.01)),
    reference_amplitude=np.array((.1020, .8980)),
    reference_amplitude_type=AmplitudeType.LORENTZIAN_PEAK_HEIGHT,
)


addline(
    element="W",
    material="metal",
    linetype="LBeta1",
    reference_short="Joe Fowler",
    reference_plot_instrument_gaussian_fwhm=None,
    nominal_peak_energy=9672.58,
    energies=np.array((9672.58,)),
    lorentzian_fwhm=np.array((7.71,)),
    reference_amplitude=np.array((1,)),
    reference_amplitude_type=AmplitudeType.LORENTZIAN_PEAK_HEIGHT,
)


addline(
    element="W",
    material="metal",
    linetype="LBeta2",
    reference_short="Joe Fowler",
    reference_plot_instrument_gaussian_fwhm=None,
    nominal_peak_energy=9964.13,
    energies=np.array((9950.82, 9962.93, 9967.53)) + 1,
    lorentzian_fwhm=np.array((9.16, 9.82, 9.90)),
    reference_amplitude=np.array((.0847, .7726, .1426)),
    reference_amplitude_type=AmplitudeType.LORENTZIAN_PEAK_HEIGHT,
)

addline(
    element="Ir",
    linetype="LAlpha",
    material="metal",
    reference_short='Zschornack',
    nominal_peak_energy=9175.2,
    energies=np.array([9099.6, 9175.2]),
    lorentzian_fwhm=np.array([7.34, 8.10]),
    reference_amplitude=np.array([.123, 1.079]),
    reference_amplitude_type=AmplitudeType.LORENTZIAN_INTEGRAL_INTENSITY,
    reference_plot_instrument_gaussian_fwhm=None
)


addline(
    element="Ir",
    linetype="LBeta1",
    material="metal",
    reference_short='Zschornack',
    nominal_peak_energy=10708.35,
    energies=np.array([10708.35]),
    lorentzian_fwhm=np.array((6.80,)),
    reference_amplitude=np.array((1,)),
    reference_amplitude_type=AmplitudeType.LORENTZIAN_INTEGRAL_INTENSITY,
    reference_plot_instrument_gaussian_fwhm=None
)

addline(
    element="Pt",
    linetype="LAlpha",
    material="metal",
    reference_short='Zschornack',
    nominal_peak_energy=9442.39,
    energies=np.array([9361.96, 9442.39]),
    lorentzian_fwhm=np.array([7.3, 7.4]),
    reference_amplitude=np.array([.130, 1.145]),
    reference_amplitude_type=AmplitudeType.LORENTZIAN_INTEGRAL_INTENSITY,
    reference_plot_instrument_gaussian_fwhm=None
)


addline(
    element="Pt",
    linetype="LBeta1",
    material="metal",
    reference_short='Zschornack',
    nominal_peak_energy=11070.84,
    energies=np.array([11070.84]),
    lorentzian_fwhm=np.array((7.43,)),
    reference_amplitude=np.array((1,)),
    reference_amplitude_type=AmplitudeType.LORENTZIAN_INTEGRAL_INTENSITY,
    reference_plot_instrument_gaussian_fwhm=None
)

addline(
    element="Au",
    linetype="LAlpha",
    material="metal",
    reference_short='Zschornack',
    nominal_peak_energy=9713.44,
    energies=np.array([9628.05, 9713.44]),
    lorentzian_fwhm=np.array([7.61, 8.60]),
    reference_amplitude=np.array([.1377, 1.214]),
    reference_amplitude_type=AmplitudeType.LORENTZIAN_INTEGRAL_INTENSITY,
    reference_plot_instrument_gaussian_fwhm=None
)


addline(
    element="Au",
    linetype="LBeta1",
    material="metal",
    reference_short='Zschornack',
    nominal_peak_energy=11442.5,
    energies=np.array([11442.5]),
    lorentzian_fwhm=np.array((8.5,)),
    reference_amplitude=np.array((1,)),
    reference_amplitude_type=AmplitudeType.LORENTZIAN_INTEGRAL_INTENSITY,
    reference_plot_instrument_gaussian_fwhm=None
)

addline(
    element="Pb",
    linetype="LAlpha",
    material="metal",
    reference_short='Zschornack',
    nominal_peak_energy=10551.7,
    energies=np.array([10449.6, 10551.6]),
    lorentzian_fwhm=np.array([9.35, 9.50]),
    reference_amplitude=np.array([.1633, 1.438]),
    reference_amplitude_type=AmplitudeType.LORENTZIAN_INTEGRAL_INTENSITY,
    reference_plot_instrument_gaussian_fwhm=None
)


addline(
    element="Pb",
    linetype="LBeta1",
    material="metal",
    reference_short='Zschornack',
    nominal_peak_energy=12613.80,
    energies=np.array([12613.80]),
    lorentzian_fwhm=np.array((8.40,)),
    reference_amplitude=np.array((1,)),
    reference_amplitude_type=AmplitudeType.LORENTZIAN_INTEGRAL_INTENSITY,
    reference_plot_instrument_gaussian_fwhm=None
)

addline(
    element="Bi",
    linetype="LAlpha",
    material="metal",
    reference_short='Zschornack',
    nominal_peak_energy=10838.94,
    energies=np.array([10731.06, 10838.94]),
    lorentzian_fwhm=np.array([8.67, 9.80]),
    reference_amplitude=np.array([.1726, 1.519]),
    reference_amplitude_type=AmplitudeType.LORENTZIAN_INTEGRAL_INTENSITY,
    reference_plot_instrument_gaussian_fwhm=None
)


addline(
    element="Bi",
    linetype="LBeta1",
    material="metal",
    reference_short='Zschornack',
    nominal_peak_energy=13023.65,
    energies=np.array([13023.65]),
    lorentzian_fwhm=np.array((9.603,)),
    reference_amplitude=np.array((1,)),
    reference_amplitude_type=AmplitudeType.LORENTZIAN_INTEGRAL_INTENSITY,
    reference_plot_instrument_gaussian_fwhm=None
)

addline(
    element="Nb",
    linetype="KBeta",
    material="Nb2O5",
    reference_short="Ravel 2018",
    reference_plot_instrument_gaussian_fwhm=1.2,
    nominal_peak_energy=18625.4,
    energies=np.array((18625.4, 18609.9)),
    lorentzian_fwhm=np.array((6.7, 6.7)),
    reference_amplitude=np.array((1, 0.5)),
    reference_amplitude_type=AmplitudeType.LORENTZIAN_INTEGRAL_INTENSITY,
)


addline(
    element="Nb",
    linetype="KBeta24",
    material="Nb2O5",
    reference_short="Ravel 2018",
    reference_plot_instrument_gaussian_fwhm=1.2,
    nominal_peak_energy=18952.79,
    energies=np.array((18952.79, 18968.0, 18982.7)),
    lorentzian_fwhm=np.array((8.67, 1.9, 5.2)),
    reference_amplitude=np.array((14.07, 0.066, 0.359)),
    reference_amplitude_type=AmplitudeType.LORENTZIAN_INTEGRAL_INTENSITY,
)


addline(
    element="Mo",
    material="metal",
    linetype="KAlpha",
    reference_short="Mendenhall 2019",
    reference_plot_instrument_gaussian_fwhm=0.02,
    nominal_peak_energy=17479.389,
    energies=np.array((17479.389, 17374.577)),
    lorentzian_fwhm=np.array((6.389, 6.3876)),
    reference_amplitude=np.array((3331.119, 1684.988)),
    ka12_energy_diff=104.812,
    reference_amplitude_type=AmplitudeType.LORENTZIAN_INTEGRAL_INTENSITY,
)


addline(
    element="Mo",
    material="metal",
    linetype="KBeta",
    reference_short="Mendenhall 2019",
    reference_plot_instrument_gaussian_fwhm=0.02,
    nominal_peak_energy=19606.734,
    energies=np.array((19606.733, 19589.251, 19623.217)),
    lorentzian_fwhm=np.array((6.88, 6.88, 6.88)),
    reference_amplitude=np.array((958.08, 488.67, 29.14)),
    reference_amplitude_type=AmplitudeType.LORENTZIAN_INTEGRAL_INTENSITY,
)


addline(
    element="Pr",
    material="metal",
    linetype="KAlpha",
    reference_short="Zschornack",
    nominal_peak_energy=36026.71,
    energies=np.array([35550.59, 36026.71]),
    lorentzian_fwhm=np.array([19.08, 1887]),
    reference_amplitude=np.array([4.15, 7.57]),
    reference_amplitude_type=AmplitudeType.LORENTZIAN_INTEGRAL_INTENSITY,
    ka12_energy_diff=36026.71 - 35550.59,
    reference_plot_instrument_gaussian_fwhm=None
)


addline(
    element="Pr",
    material="metal",
    linetype="KBeta",
    reference_short="Zschornack",
    nominal_peak_energy=40748.67,
    energies=np.array([40653.27, 40748.67]),
    lorentzian_fwhm=np.array([21.79, 22.73]),
    reference_amplitude=np.array([0.742, 1.436]),
    reference_amplitude_type=AmplitudeType.LORENTZIAN_INTEGRAL_INTENSITY,
    reference_plot_instrument_gaussian_fwhm=None
)


addline(
    element="Nd",
    material="metal",
    linetype="KAlpha",
    reference_short="Zschornack",
    nominal_peak_energy=37360.74,
    energies=np.array([36847.50, 37360.74]),
    lorentzian_fwhm=np.array([20.25, 20.05]),
    reference_amplitude=np.array([4.51, 8.21]),
    reference_amplitude_type=AmplitudeType.LORENTZIAN_INTEGRAL_INTENSITY,
    ka12_energy_diff=37360.74 - 36847.50,
    reference_plot_instrument_gaussian_fwhm=None
)


addline(
    element="Nd",
    material="metal",
    linetype="KBeta",
    reference_short="Zschornack",
    nominal_peak_energy=42271.2,
    energies=np.array([42166.24, 42271.17]),
    lorentzian_fwhm=np.array([23.22, 24.16]),
    reference_amplitude=np.array([0.845, 1.636]),
    reference_amplitude_type=AmplitudeType.LORENTZIAN_INTEGRAL_INTENSITY,
    reference_plot_instrument_gaussian_fwhm=None
)


addline(
    element="Sm",
    material="metal",
    linetype="KAlpha",
    reference_short="Zschornack",
    nominal_peak_energy=40118.48,
    energies=np.array([39523.39, 40118.48]),
    lorentzian_fwhm=np.array([22.8, 22.6]),
    reference_amplitude=np.array([5.12, 9.27]),
    reference_amplitude_type=AmplitudeType.LORENTZIAN_INTEGRAL_INTENSITY,
    ka12_energy_diff=40118.48 - 39523.39,
    reference_plot_instrument_gaussian_fwhm=None
)


addline(
    element="Sm",
    material="metal",
    linetype="KBeta",
    reference_short="Zschornack",
    nominal_peak_energy=45413.0,
    energies=np.array([45288.6, 45413.0]),
    lorentzian_fwhm=np.array([26.2, 26.6]),
    reference_amplitude=np.array([0.927, 1.797]),
    reference_amplitude_type=AmplitudeType.LORENTZIAN_INTEGRAL_INTENSITY,
    reference_plot_instrument_gaussian_fwhm=None
)


addline(
    element="Eu",
    material="metal",
    linetype="KAlpha",
    reference_short="Zschornack",
    nominal_peak_energy=41542.63,
    energies=np.array([40902.33, 41542.63]),
    lorentzian_fwhm=np.array([24.1, 23.93]),
    reference_amplitude=np.array([5.54, 10.00]),
    reference_amplitude_type=AmplitudeType.LORENTZIAN_INTEGRAL_INTENSITY,
    ka12_energy_diff=41542.63 - 40902.33,
    reference_plot_instrument_gaussian_fwhm=None
)


addline(
    element="Eu",
    material="metal",
    linetype="KBeta",
    reference_short="Zschornack",
    nominal_peak_energy=47038.4,
    energies=np.array([46904.0, 47038.4]),
    lorentzian_fwhm=np.array([27.86, 28.25]),
    reference_amplitude=np.array([1.050, 2.031]),
    reference_amplitude_type=AmplitudeType.LORENTZIAN_INTEGRAL_INTENSITY,
    reference_plot_instrument_gaussian_fwhm=None
)


addline(
    element="Gd",
    material="metal",
    linetype="KAlpha",
    reference_short="Zschornack",
    nominal_peak_energy=42996.72,
    energies=np.array([42309.30, 42996.72]),
    lorentzian_fwhm=np.array([25.5, 25.4]),
    reference_amplitude=np.array([5.9, 10.6]),
    reference_amplitude_type=AmplitudeType.LORENTZIAN_INTEGRAL_INTENSITY,
    ka12_energy_diff=42996.72 - 42309.30,
    reference_plot_instrument_gaussian_fwhm=None
)


addline(
    element="Gd",
    material="metal",
    linetype="KBeta",
    reference_short="Zschornack",
    nominal_peak_energy=48696.9,
    energies=np.array([48555.8, 49696.9]),
    lorentzian_fwhm=np.array([29.5, 30.0]),
    reference_amplitude=np.array([1.10, 2.13]),
    reference_amplitude_type=AmplitudeType.LORENTZIAN_INTEGRAL_INTENSITY,
    reference_plot_instrument_gaussian_fwhm=None
)


addline(
    element="Tb",
    material="metal",
    linetype="KAlpha",
    reference_short="Zschornack",
    nominal_peak_energy=44482.7,
    energies=np.array([43744.6, 44482.7]),
    lorentzian_fwhm=np.array([27.0, 26.9]),
    reference_amplitude=np.array([6.3, 11.3]),
    reference_amplitude_type=AmplitudeType.LORENTZIAN_INTEGRAL_INTENSITY,
    ka12_energy_diff=44482.7 - 43744.6,
    reference_plot_instrument_gaussian_fwhm=None
)


addline(
    element="Tb",
    material="metal",
    linetype="KBeta",
    reference_short="Zschornack",
    nominal_peak_energy=50382.9,
    energies=np.array([50229.8, 50382.9]),
    lorentzian_fwhm=np.array([31.7, 31.8]),
    reference_amplitude=np.array([1.15, 2.33]),
    reference_amplitude_type=AmplitudeType.LORENTZIAN_INTEGRAL_INTENSITY,
    reference_plot_instrument_gaussian_fwhm=None
)


addline(
    element="Dy",
    material="metal",
    linetype="KAlpha",
    reference_short="Zschornack",
    nominal_peak_energy=45998.94,
    energies=np.array([45208.27, 45998.94]),
    lorentzian_fwhm=np.array([28.54, 28.41]),
    reference_amplitude=np.array([6.68, 11.94]),
    reference_amplitude_type=AmplitudeType.LORENTZIAN_INTEGRAL_INTENSITY,
    ka12_energy_diff=45998.94 - 45208.27,
    reference_plot_instrument_gaussian_fwhm=None
)


addline(
    element="Dy",
    material="metal",
    linetype="KBeta",
    reference_short="Zschornack",
    nominal_peak_energy=52119.7,
    energies=np.array([51958.1, 52119.7]),
    lorentzian_fwhm=np.array([33.5, 33.66]),
    reference_amplitude=np.array([1.23, 2.376]),
    reference_amplitude_type=AmplitudeType.LORENTZIAN_INTEGRAL_INTENSITY,
    reference_plot_instrument_gaussian_fwhm=None
)


addline(
    element="Ho",
    material="metal",
    linetype="KAlpha",
    reference_short="Zschornack",
    nominal_peak_energy=47546.70,
    energies=np.array([46699.70, 47546.70]),
    lorentzian_fwhm=np.array([30.16, 30.04]),
    reference_amplitude=np.array([7.13, 12.69]),
    reference_amplitude_type=AmplitudeType.LORENTZIAN_INTEGRAL_INTENSITY,
    ka12_energy_diff=47546.70 - 46699.70,
    reference_plot_instrument_gaussian_fwhm=None
)


addline(
    element="Ho",
    material="metal",
    linetype="KBeta",
    reference_short="Zschornack",
    nominal_peak_energy=53877.0,
    energies=np.array([53711.0, 53877.0]),
    lorentzian_fwhm=np.array([35.4, 35.6]),
    reference_amplitude=np.array([1.31, 2.54]),
    reference_amplitude_type=AmplitudeType.LORENTZIAN_INTEGRAL_INTENSITY,
    reference_plot_instrument_gaussian_fwhm=None
)


def plot_all_spectra(maxplots=10):
    """Makes plots showing the line shape and component parts for some lines.
    Intended to replicate plots in the literature giving spectral lineshapes."""
    keys = list(spectra.keys())[:maxplots]
    for name in keys:
        spectrum = spectra[name]()
        spectrum.plot_like_reference()
