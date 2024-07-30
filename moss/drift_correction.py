import numpy as np
import numba
import scipy as sp
import scipy.optimize
from dataclasses import dataclass, field
import mass
import moss
from moss import CalStep
import polars as pl
import typing
import pylab as plt

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

@dataclass(frozen=True)
class DriftCorrectStep(CalStep):
    dc: typing.Any

    def calc_from_df(self, df):
        indicator, uncorrected = self.inputs
        slope, offset = self.dc.slope, self.dc.offset
        df2 = df.select(
            (pl.col(uncorrected) * (1 + slope * (pl.col(indicator) - offset))).alias(self.output[0])
        ).with_columns(df)
        return df2

    def dbg_plot(self, df):
        indicator, uncorrected = self.inputs
        # breakpoint()
        df_small = (
            df.lazy()
            .filter(self.good_expr)
            .filter(self.use_expr)
            .select(self.inputs + self.output)
            .collect()
        )
        moss.misc.plot_a_vs_b_series(df_small[indicator], df_small[uncorrected])
        moss.misc.plot_a_vs_b_series(
            df_small[indicator],
            df_small[self.output[0]],
            plt.gca(),
        )
        plt.legend()
        plt.tight_layout()
        return plt.gca()
    
    @classmethod
    def learn(cls, ch, indicator, uncorrected, corrected, use_expr):
        if corrected is None:
            corrected = uncorrected + "_dc"
        indicator_s, uncorrected_s = ch.good_serieses([indicator, uncorrected], use_expr)
        dc = moss.drift_correct(
            indicator=indicator_s.to_numpy(),
            uncorrected=uncorrected_s.to_numpy(),
        )
        step = cls(
            inputs=[indicator, uncorrected],
            output=[corrected],
            good_expr=ch.good_expr,
            use_expr=use_expr,
            dc=dc,
        )
        return step


@dataclass
class DriftCorrection:
    offset: float
    slope: float

    def __call__(self, indicator, uncorrected):
        return uncorrected*(1+(indicator-self.offset)*self.slope)

