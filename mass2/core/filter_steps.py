import polars as pl

from dataclasses import dataclass
from collections.abc import Callable
from mass2.core.cal_steps import CalStep
from mass2.core.noise_algorithms import NoiseResult
from mass2.core.optimal_filtering import Filter, FilterMaker


@dataclass(frozen=True)
class Filter5LagStep(CalStep):
    filter: Filter
    spectrum: NoiseResult
    filter_maker: "FilterMaker"
    transform_raw: Callable | None = None

    def calc_from_df(self, df: pl.DataFrame) -> pl.DataFrame:
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

    def dbg_plot(self, _):
        return self.filter.plot()
