import numpy as np
from dataclasses import dataclass
import polars as pl
import typing
import pylab as plt

import mass2
from mass2.core.recipe import RecipeStep


def drift_correct_mass(indicator, uncorrected):
    slope, dc_info = mass2.core.analysis_algorithms.drift_correct(indicator, uncorrected)
    offset = dc_info["median_pretrig_mean"]
    return DriftCorrection(slope=slope, offset=offset)


def drift_correct_wip(indicator, uncorrected):
    opt_result, offset = mass2.core.rough_cal.minimize_entropy_linear(
        indicator,
        uncorrected,
        bin_edges=np.arange(0, 60000, 1),
        fwhm_in_bin_number_units=5,
    )
    return DriftCorrection(offset=offset.astype(np.float64), slope=opt_result.x.astype(np.float64))


drift_correct = drift_correct_mass


@dataclass(frozen=True)
class DriftCorrectStep(RecipeStep):
    dc: typing.Any

    def calc_from_df(self, df):
        indicator_col, uncorrected_col = self.inputs
        slope, offset = self.dc.slope, self.dc.offset
        df2 = df.select((pl.col(uncorrected_col) * (1 + slope * (pl.col(indicator_col) - offset))).alias(self.output[0])).with_columns(
            df
        )
        return df2

    def dbg_plot(self, df):
        indicator_col, uncorrected_col = self.inputs
        # breakpoint()
        df_small = df.lazy().filter(self.good_expr).filter(self.use_expr).select(self.inputs + self.output).collect()
        mass2.misc.plot_a_vs_b_series(df_small[indicator_col], df_small[uncorrected_col])
        mass2.misc.plot_a_vs_b_series(
            df_small[indicator_col],
            df_small[self.output[0]],
            plt.gca(),
        )
        plt.legend()
        plt.tight_layout()
        return plt.gca()

    @classmethod
    def learn(cls, ch, indicator_col, uncorrected_col, corrected_col, use_expr):
        if corrected_col is None:
            corrected_col = uncorrected_col + "_dc"
        indicator_s, uncorrected_s = ch.good_serieses([indicator_col, uncorrected_col], use_expr)
        dc = mass2.core.drift_correct(
            indicator=indicator_s.to_numpy(),
            uncorrected=uncorrected_s.to_numpy(),
        )
        step = cls(
            inputs=[indicator_col, uncorrected_col],
            output=[corrected_col],
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
        return uncorrected * (1 + (indicator - self.offset) * self.slope)
