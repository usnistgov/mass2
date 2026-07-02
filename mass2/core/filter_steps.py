"""
Provide `OptimalFilterStep`, a step to apply an optimal filter to pulse data in a DataFrame.
"""

import polars as pl
from dataclasses import dataclass
import dataclasses
from collections.abc import Callable
from typing import Any
import pylab as plt
from .misc import PulseDataFramer
from .noise_algorithms import NoiseResult
from .optimal_filtering import Filter, FilterMaker
from .recipe import RecipeStep


@dataclass(frozen=True)
class OptimalFilterStep(RecipeStep):
    """A step to apply an optimal filter to pulse data in a DataFrame."""

    filter: Filter
    spectrum: NoiseResult | None
    filter_maker: "FilterMaker"
    transform_raw: Callable | None = None

    def calc_from_df(self, df: pl.DataFrame, pulseframer: PulseDataFramer | None = None) -> pl.DataFrame:
        """Apply the optimal filter to the input DataFrame and return a new DataFrame with results."""
        assert pulseframer is not None
        rawcol = self.inputs[0]
        dfs = []
        for raw_df in pulseframer.iterate_raw_pulses(chunksize=4096):
            raw = raw_df[rawcol].to_numpy()
            if self.transform_raw is not None:
                raw = self.transform_raw(raw)
            peak_y, peak_x = self.filter.filter_records(raw)
            dfs.append(pl.DataFrame({"peak_x": peak_x, "peak_y": peak_y}))
        df2 = pl.concat(dfs).with_columns(df)
        df2 = df2.rename({"peak_x": self.output[0], "peak_y": self.output[1]})
        return df2

    def dbg_plot(self, df_after: pl.DataFrame, **kwargs: Any) -> plt.Axes:
        """Plot the filter shape for debugging purposes."""
        plt.figure()
        axis = plt.subplot(111)
        self.filter.plot(axis)
        return axis

    def drop_debug(self) -> "OptimalFilterStep":
        """Return a copy of this step with debugging information (the NoiseResult) removed."""
        return dataclasses.replace(self, spectrum=None)
