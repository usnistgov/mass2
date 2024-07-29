import numpy as np
import numba
import scipy as sp
import scipy.optimize
from dataclasses import dataclass, field
import mass
import moss

def drift_correct_mass(indicator, uncorrected):
    slope, dc_info = \
            mass.core.analysis_algorithms.drift_correct(indicator, uncorrected)
    offset = dc_info["median_pretrig_mean"]
    return DriftCorrection(slope=slope, offset=offset)

def drift_correct(indicator, uncorrected):
    opt_result, offset=moss.rough_cal.minimize_entropy_linear(indicator, uncorrected, 
                    bin_edges = np.arange(0, 60000, 1), fwhm_in_bin_number_units=5)
    print(f"{opt_result=}")
    return DriftCorrection(offset=offset.astype(np.float64), slope=opt_result.x.astype(np.float64))

@dataclass
class DriftCorrection:
    offset: float
    slope: float

    def __call__(self, indicator, uncorrected):
        return uncorrected*(1+(indicator-self.offset)*self.slope)

