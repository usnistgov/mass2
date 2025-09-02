"""
Provide `Filter5LagStep`, a step to apply a 5-lag optimal filter to pulse data in a DataFrame.
"""

import polars as pl
from dataclasses import dataclass
import dataclasses
from collections.abc import Callable
from typing import Any
import pylab as plt
from mass2.core.recipe import RecipeStep
from mass2.core.noise_algorithms import NoiseResult
from mass2.core.optimal_filtering import Filter, FilterMaker


@dataclass(frozen=True)
class Filter5LagStep(RecipeStep):
    """A step to apply a 5-lag optimal filter to pulse data in a DataFrame."""

    filter: Filter
    spectrum: NoiseResult | None
    filter_maker: "FilterMaker"
    transform_raw: Callable | None = None

    def calc_from_df(self, df: pl.DataFrame) -> pl.DataFrame:
        """Apply the 5-lag optimal filter to the input DataFrame and return a new DataFrame with results."""
        dfs = []
        for df_iter in df.iter_slices(10000):
            raw = df_iter[self.inputs[0]].to_numpy()
            if self.transform_raw is not None:
                raw = self.transform_raw(raw)
            peak_y, peak_x = self.filter.filter_records(raw)
            dfs.append(pl.DataFrame({"peak_x": peak_x, "peak_y": peak_y}))
        df2 = pl.concat(dfs).with_columns(df)
        df2 = df2.rename({"peak_x": self.output[0], "peak_y": self.output[1]})
        return df2

    def dbg_plot(self, df_after: pl.DataFrame, **kwargs: Any) -> plt.Axes:
        """Plot the filter shape for debugging purposes."""
        self.filter.plot()
        return plt.gca()

    def drop_debug(self) -> "Filter5LagStep":
        """Return a copy of this step with debugging information (the NoiseResult) removed."""
        return dataclasses.replace(self, spectrum=None)
