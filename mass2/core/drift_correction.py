"""
Provide `DriftCorrectStep` and `DriftCorrection` for correcting gain drifts that correlate with pretrigger mean.
"""

from typing import Any, TYPE_CHECKING
from numpy.typing import ArrayLike, NDArray
import numpy as np
from dataclasses import dataclass
import polars as pl
import typing
import pylab as plt

import mass2
from .recipe import RecipeStep

if TYPE_CHECKING:
    from .channel import Channel


def drift_correct_mass(indicator: ArrayLike, uncorrected: ArrayLike) -> "DriftCorrection":
    """Determine drift correction parameters using mass2.core.analysis_algorithms.drift_correct."""
    slope, dc_info = mass2.core.analysis_algorithms.drift_correct(indicator, uncorrected)
    offset = dc_info["median_pretrig_mean"]
    return DriftCorrection(slope=slope, offset=offset)


def drift_correct_wip(indicator: ArrayLike, uncorrected: ArrayLike) -> "DriftCorrection":
    """Work in progress to Determine drift correction parameters directly (??)."""
    opt_result, offset = mass2.core.rough_cal.minimize_entropy_linear(
        np.asarray(indicator),
        np.asarray(uncorrected),
        bin_edges=np.arange(0, 60000, 1),
        fwhm_in_bin_number_units=5,
    )
    return DriftCorrection(offset=float(offset), slope=opt_result.x.astype(np.float64))


drift_correct = drift_correct_mass


@dataclass(frozen=True)
class DriftCorrectStep(RecipeStep):
    """A RecipeStep to apply a linear drift correction to pulse data in a DataFrame."""

    dc: typing.Any

    def calc_from_df(self, df: pl.DataFrame) -> pl.DataFrame:
        """Apply the drift correction to the input DataFrame and return a new DataFrame with results."""
        indicator_col, uncorrected_col = self.inputs
        slope, offset = self.dc.slope, self.dc.offset
        df2 = df.select((pl.col(uncorrected_col) * (1 + slope * (pl.col(indicator_col) - offset))).alias(self.output[0])).with_columns(
            df
        )
        return df2

    def dbg_plot(self, df_after: pl.DataFrame, **kwargs: Any) -> plt.Axes:
        """Plot the uncorrected and corrected values against the indicator for debugging purposes."""
        indicator_col, uncorrected_col = self.inputs
        # breakpoint()
        df_small = df_after.lazy().filter(self.good_expr).filter(self.use_expr).select(self.inputs + self.output).collect()
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
    def learn(
        cls, ch: "Channel", indicator_col: str, uncorrected_col: str, corrected_col: str | None, use_expr: pl.Expr
    ) -> "DriftCorrectStep":
        """Create a DriftCorrectStep by learning the correction from data in the given Channel."""
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
    """A linear correction used to attempt remove any correlation between pretrigger mean and pulse height;
    will work with other quantities instead."""

    offset: float
    slope: float

    def __call__(self, indicator: ArrayLike, uncorrected: ArrayLike) -> NDArray:
        """Apply the drift correction to the given uncorrected values using the given indicator values."""
        indicator = np.asarray(indicator)
        uncorrected = np.asarray(uncorrected)
        return uncorrected * (1 + (indicator - self.offset) * self.slope)
