"""
hci_lines.py

Uses pickle file containing NIST ASD levels data to generate some commonly used HCI lines in mass.
Meant to be a replacement for _highly_charged_ion_lines.py, which hard codes in line parameters.

The pickle file can be gzip-compressed, provided the compressed filename ends with ".gz".

February 2020
Paul Szypryt
"""

import importlib.resources as pkg_resources
import numpy as np
from numpy.typing import ArrayLike
import pickle
import gzip
import scipy.constants as sp_const
import os
from . import fluorescence_lines
from .fluorescence_lines import SpectralLine
from . import AmplitudeType
import xraydb

INVCM_TO_EV = sp_const.c * sp_const.physical_constants["Planck constant in eV s"][0] * 100.0
DEFAULT_PICKLE_NAME = "nist_asd_2023.pickle.gz"
DEFAULT_PICKLE_PATH = pkg_resources.files("mass2").joinpath("data", DEFAULT_PICKLE_NAME)


class NIST_ASD:
    """Class for working with a pickled atomic spectra database"""

    def __init__(self, pickleFilename: str | None = None):
        """Loads ASD pickle file (optionally gzipped)

        Parameters
        ----------
        pickleFilename : str | None, optional
            ASD pickle file name, as str, or if none then `mass2.calibration.hci_lines.DEFAULT_PICKLE_PATH` (default None)
        """

        if pickleFilename is None:
            pickleFilename = os.path.join(os.path.split(__file__)[0], str(DEFAULT_PICKLE_PATH))

        if pickleFilename.endswith(".gz"):
            with gzip.GzipFile(pickleFilename, "rb") as handle:
                self.NIST_ASD_Dict = pickle.load(handle)
        else:
            with open(pickleFilename, "rb") as handle:
                self.NIST_ASD_Dict = pickle.load(handle)

    def getAvailableElements(self) -> list[str]:
        """Returns a list of all available elements from the ASD pickle file"""

        return list(self.NIST_ASD_Dict.keys())

    def getAvailableSpectralCharges(self, element: str) -> list[int]:
        """For a given element, returns a list of all available charge states from the ASD pickle file

        Parameters
        ----------
        element : str
            atomic symbol of element, e.g. 'Ne'

        Returns
        -------
        list[int]
            Available charge states
        """

        return list(self.NIST_ASD_Dict[element].keys())

    def getAvailableLevels(
        self,
        element: str,
        spectralCharge: int,
        requiredConf: str | None = None,
        requiredTerm: str | None = None,
        requiredJVal: str | None = None,
        maxLevels: int | None = None,
        units: str = "eV",
        getUncertainty: bool = True,
    ) -> dict:
        """For a given element and spectral charge state, return a dict of all known levels from the ASD pickle file

        Parameters
        ----------
        element : str
            Elemental atomic symbol, e.g. 'Ne'
        spectralCharge : int
            spectral charge state, e.g. 1 for neutral atoms, 10 for H-like Ne
        requiredConf : str | None, optional
            if not None, limits results to those with `conf == requiredConf`, by default None
        requiredTerm : str | None, optional
            if not None, limits results to those with `term == requiredTerm`, by default None
        requiredJVal : str | None, optional
            if not None, limits results to those with `a == requiredJVal`, by default None
        maxLevels : int | None, optional
            the maximum number of levels (sorted by energy) to return, by default None
        units : str, optional
            'cm-1' or 'eV' for returned line position. If 'eV', converts from database 'cm-1' values, by default "eV"
        getUncertainty : bool, optional
            whether to return uncertain values, by default True

        Returns
        -------
        dict
            A dictionary of energy level strings to energy levels.
        """
        if units not in {"eV", "cm-1"}:
            raise ValueError("Unit type not supported, please use eV or cm-1")

        spectralCharge = int(spectralCharge)
        levelsDict: dict = {}
        numLevels = 0
        for iLevel in list(self.NIST_ASD_Dict[element][spectralCharge].keys()):
            try:
                # Check to see if we reached maximum number of levels to return
                if maxLevels is not None:
                    if numLevels == maxLevels:
                        return levelsDict
                # If required, check to see if level matches search conf, term, JVal
                includeTerm = False
                includeJVal = False
                conf, term, j_str = iLevel.split()
                JVal = j_str.split("=")[1]
                includeConf = (requiredConf is None) or conf == requiredConf
                includeTerm = (requiredTerm is None) or term == requiredTerm
                includeJVal = (requiredJVal is None) or JVal == requiredJVal

                # Include levels that match, in either cm-1 or eV
                if includeConf and includeTerm and includeJVal:
                    numLevels += 1
                    if units == "cm-1":
                        if getUncertainty:
                            levelsDict[iLevel] = self.NIST_ASD_Dict[element][spectralCharge][iLevel]
                        else:
                            levelsDict[iLevel] = self.NIST_ASD_Dict[element][spectralCharge][iLevel][0]
                    elif units == "eV":
                        if getUncertainty:
                            levelsDict[iLevel] = [
                                iValue * INVCM_TO_EV for iValue in self.NIST_ASD_Dict[element][spectralCharge][iLevel]
                            ]
                        else:
                            levelsDict[iLevel] = INVCM_TO_EV * self.NIST_ASD_Dict[element][spectralCharge][iLevel][0]
            except ValueError:
                f"Warning: cannot parse level: {iLevel}"
        return levelsDict

    def getSingleLevel(
        self, element: str, spectralCharge: int, conf: str, term: str, JVal: str, units: str = "eV", getUncertainty: bool = True
    ) -> float:
        """Return the level data for a fully defined element, charge state, conf, term, and JVal.

        Parameters
        ----------
        element : str
            atomic symbol of element, e.g. 'Ne'
        spectralCharge : int
            spectral charge state, e.g. 1 for neutral atoms, 10 for H-like Ne
        conf : str
            nuclear configuration, e.g. '2p'
        term : str
            nuclear term, e.g. '2P*'
        JVal : str
            total angular momentum J, e.g. '3/2'
        units : str, optional
            'cm-1' or 'eV' for returned line position. If 'eV', converts from database 'cm-1' values, by default "eV"
        getUncertainty : bool, optional
            includes uncertainties in list of levels, by default True

        Returns
        -------
        float
            _description_
        """

        levelString = f"{conf} {term} J={JVal}"
        if units == "cm-1":
            if getUncertainty:
                levelEnergy = self.NIST_ASD_Dict[element][spectralCharge][levelString]
            else:
                levelEnergy = self.NIST_ASD_Dict[element][spectralCharge][levelString][0]
        elif units == "eV":
            if getUncertainty:
                levelEnergy = [iValue * INVCM_TO_EV for iValue in self.NIST_ASD_Dict[element][spectralCharge][levelString]]
            else:
                levelEnergy = self.NIST_ASD_Dict[element][spectralCharge][levelString][0] * INVCM_TO_EV
        else:
            raise ValueError("Unit type not supported, please use eV or cm-1")
        return levelEnergy


# Some non-class functions useful for integration with mass
def add_hci_line(
    element: str,
    spectr_ch: int,
    line_identifier: str,
    energies: ArrayLike,
    widths: ArrayLike,
    ratios: ArrayLike,
    nominal_peak_energy: float | None = None,
) -> SpectralLine:
    energies = np.asarray(energies)
    widths = np.asarray(widths)
    ratios = np.asarray(ratios)
    if nominal_peak_energy is None:
        nominal_peak_energy = np.dot(energies, ratios) / np.sum(ratios)
    linetype = f"{int(spectr_ch)} {line_identifier}"

    spectrum_class = fluorescence_lines.addline(
        element=element,
        material="Highly Charged Ion",
        linetype=linetype,
        reference_short="NIST ASD",
        reference_plot_instrument_gaussian_fwhm=0.5,
        nominal_peak_energy=nominal_peak_energy,
        energies=energies,
        lorentzian_fwhm=widths,
        reference_amplitude=ratios,
        reference_amplitude_type=AmplitudeType.LORENTZIAN_PEAK_HEIGHT,
        ka12_energy_diff=None,
    )
    return spectrum_class


def add_H_like_lines_from_asd(asd: NIST_ASD, element: str, maxLevels: int | None = None) -> list[SpectralLine]:
    spectr_ch = xraydb.atomic_number(element)
    added_lines = []
    if maxLevels is not None:
        levelsDict = asd.getAvailableLevels(element, spectralCharge=spectr_ch, maxLevels=maxLevels + 1)
    else:
        levelsDict = asd.getAvailableLevels(element, spectralCharge=spectr_ch)
    for iLevel in list(levelsDict.keys()):
        lineEnergy = levelsDict[iLevel][0]
        if lineEnergy != 0.0:
            iLine = add_hci_line(
                element=element, spectr_ch=spectr_ch, line_identifier=iLevel, energies=[lineEnergy], widths=[0.1], ratios=[1.0]
            )
            added_lines.append(iLine)
    return added_lines


def add_He_like_lines_from_asd(asd: NIST_ASD, element: str, maxLevels: int | None = None) -> list[SpectralLine]:
    spectr_ch = xraydb.atomic_number(element) - 1
    added_lines = []
    if maxLevels is not None:
        levelsDict = asd.getAvailableLevels(element, spectralCharge=spectr_ch, maxLevels=maxLevels + 1)
    else:
        levelsDict = asd.getAvailableLevels(element, spectralCharge=spectr_ch)
    for iLevel in list(levelsDict.keys()):
        lineEnergy = levelsDict[iLevel][0]
        if lineEnergy != 0.0:
            iLine = add_hci_line(
                element=element, spectr_ch=spectr_ch, line_identifier=iLevel, energies=[lineEnergy], widths=[0.1], ratios=[1.0]
            )
            added_lines.append(iLine)
    return added_lines


# Script for adding some lines for elements commonly used at the EBIT
asd = NIST_ASD()
elementList = ["N", "O", "Ne", "Ar"]
# Add all known H- and He-like lines for these elements
for iElement in elementList:
    add_H_like_lines_from_asd(asd=asd, element=iElement, maxLevels=None)
    add_He_like_lines_from_asd(asd=asd, element=iElement, maxLevels=None)
