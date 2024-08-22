from dataclasses import dataclass, field
import polars as pl
import pylab as plt
import functools
import moss
import numpy as np

@dataclass(frozen=True)
class NoiseChannel:
    df: pl.DataFrame | pl.LazyFrame  # DO NOT MUTATE THIS!!!
    header_df: pl.DataFrame | pl.LazyFrame  # DO NOT MUTATE THIS!!
    frametime_s: float

    # @functools.cache
    def calc_max_excursion(
        self, trace_col_name="pulse", n_limit=10000, excursion_nsigma=5
    ):
        def excursion2d(noise_trace):
            return np.amax(noise_trace, axis=1) - np.amin(noise_trace, axis=1)
        noise_traces = self.df.limit(n_limit)[trace_col_name].to_numpy()
        excursion = excursion2d(noise_traces)
        max_excursion = moss.misc.outlier_resistant_nsigma_above_mid(
            excursion, nsigma=excursion_nsigma
        )
        df_noise2 = self.df.limit(n_limit).with_columns(excursion=excursion)
        return df_noise2, max_excursion

    # @functools.cache
    def spectrum(
        self,
        trace_col_name="pulse",
        n_limit=10000,
        excursion_nsigma=5,
        trunc_front=0,
        trunc_back=0,
    ):
        df_noise2, max_excursion = self.calc_max_excursion(
            trace_col_name, n_limit, excursion_nsigma
        )
        noise_traces_clean = (
            df_noise2.filter(pl.col("excursion") < max_excursion)["pulse"]
            .to_numpy()
        )
        if trunc_back == 0:
            noise_traces_clean2 = noise_traces_clean[:, trunc_front:]
        elif trunc_back > 0:
            noise_traces_clean2 = noise_traces_clean[:, trunc_front:-trunc_back]
        else:
            raise ValueError(f"trunc_back must be >= 0")
        spectrum = moss.noise_psd(noise_traces_clean2, dt=self.frametime_s)
        return spectrum

    def __hash__(self):
        # needed to make functools.cache work
        # if self or self.anything is mutated, assumptions will be broken
        # and we may get nonsense results
        return hash(id(self))

    def __eq__(self, other):
        return id(self) == id(other)

    @classmethod
    def from_ljh(cls, path):
        ljh = moss.LJHFile(path)
        df, header_df = ljh.to_polars()
        noise_channel = cls(df, header_df, header_df["Timebase"][0])
        return noise_channel
