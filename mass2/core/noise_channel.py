from dataclasses import dataclass
import polars as pl
import numpy as np
import mass2


@dataclass(frozen=True)
class NoiseChannel:
    df: pl.DataFrame | pl.LazyFrame  # DO NOT MUTATE THIS!!!
    header_df: pl.DataFrame | pl.LazyFrame  # DO NOT MUTATE THIS!!
    frametime_s: float

    # @functools.cache
    def calc_max_excursion(self, trace_col_name="pulse", n_limit=10000, excursion_nsigma=5):
        def excursion2d(noise_trace):
            return np.amax(noise_trace, axis=1) - np.amin(noise_trace, axis=1)

        noise_traces = self.df.limit(n_limit)[trace_col_name].to_numpy()
        excursion = excursion2d(noise_traces)
        max_excursion = mass2.misc.outlier_resistant_nsigma_above_mid(excursion, nsigma=excursion_nsigma)
        df_noise2 = self.df.limit(n_limit).with_columns(excursion=excursion)
        return df_noise2, max_excursion

    def get_records_2d(
        self,
        trace_col_name="pulse",
        n_limit=10000,
        excursion_nsigma=5,
        trunc_front=0,
        trunc_back=0,
    ):
        """
        Return a 2D NumPy array of cleaned noise traces from the specified column.

        This method identifies noise traces with excursions below a threshold and
        optionally truncates the beginning and/or end of each trace.

        Parameters:
        ----------
        trace_col_name : str, optional
            Name of the column containing trace data. Default is "pulse".
        n_limit : int, optional
            Maximum number of traces to analyze. Default is 10000.
        excursion_nsigma : float, optional
            Threshold for maximum excursion in units of noise sigma. Default is 5.
        trunc_front : int, optional
            Number of samples to truncate from the front of each trace. Default is 0.
        trunc_back : int, optional
            Number of samples to truncate from the back of each trace. Must be >= 0. Default is 0.

        Returns:
        -------
        np.ndarray
            A 2D array of cleaned and optionally truncated noise traces.

            Shape: (n_pulses, len(pulse))
        """
        df_noise2, max_excursion = self.calc_max_excursion(trace_col_name, n_limit, excursion_nsigma)
        noise_traces_clean = df_noise2.filter(pl.col("excursion") <= max_excursion)["pulse"].to_numpy()
        if trunc_back == 0:
            noise_traces_clean2 = noise_traces_clean[:, trunc_front:]
        elif trunc_back > 0:
            noise_traces_clean2 = noise_traces_clean[:, trunc_front:-trunc_back]
        else:
            raise ValueError("trunc_back must be >= 0")
        assert noise_traces_clean2.shape[0] > 0
        return noise_traces_clean2

    # @functools.cache
    def spectrum(
        self,
        trace_col_name="pulse",
        n_limit=10000,
        excursion_nsigma=5,
        trunc_front=0,
        trunc_back=0,
    ):
        records = self.get_records_2d(trace_col_name, n_limit, excursion_nsigma, trunc_front, trunc_back)
        spectrum = mass2.core.noise_algorithms.noise_psd_mass(records, dt=self.frametime_s)
        return spectrum

    def __hash__(self):
        # needed to make functools.cache work
        # if self or self.anything is mutated, assumptions will be broken
        # and we may get nonsense results
        return hash(id(self))

    def __eq__(self, other):
        return id(self) == id(other)

    @property
    def is_continuous(self):
        if "continuous" in self.header_df:
            return self.header_df["continuous"][0]
        return False

    @classmethod
    def from_ljh(cls, path):
        ljh = mass2.LJHFile.open(path)
        df, header_df = ljh.to_polars()
        noise_channel = cls(df, header_df, header_df["Timebase"][0])
        return noise_channel
