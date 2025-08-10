import polars as pl

from dataclasses import dataclass
from mass2.core.cal_steps import CalStep
from mass2.core.noise_algorithms import NoisePSD
from mass2.core.optimal_filtering import Filter


@dataclass(frozen=True)
class Filter5LagStep(CalStep):
    filter: Filter
    spectrum: NoisePSD

    def calc_from_df(self, df):
        dfs = []
        for df_iter in df.iter_slices(10000):
            peak_y, peak_x = self.filter.filter_records(df_iter[self.inputs[0]].to_numpy())
            dfs.append(pl.DataFrame({"peak_x": peak_x, "peak_y": peak_y}))
        df2 = pl.concat(dfs).with_columns(df)
        df2 = df2.rename({"peak_x": self.output[0], "peak_y": self.output[1]})
        return df2

    def dbg_plot(self, _):
        return self.filter.plot()
