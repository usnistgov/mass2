"""
Hold a class to represent a channel with noise data only, and to analyze its noise characteristics.
"""

from typing import Any
from numpy.typing import NDArray
from pathlib import Path
from dataclasses import dataclass
import polars as pl
import numpy as np
import mass2
from .noise_algorithms import NoiseResult
from .pulsefiles import PulseMaker


@dataclass(frozen=True)
class NoiseChannel:
    """A class to represent a channel with noise data only, and to analyze its noise characteristics."""

    df: pl.DataFrame  # DO NOT MUTATE THIS!!!
    header_df: pl.DataFrame  # DO NOT MUTATE THIS!!
    frametime_s: float
    pulsereader: PulseMaker | None = None

    @property
    def npulses(self) -> int:
        return len(self.df)

    def load_raw(self, r: range | slice | None = None) -> NDArray:
        assert self.pulsereader is not None, "Cannot run `NoiseChannel.load_raw()` without a pulsereader function"
        if r is None:
            r = range(0, self.npulses)
        return self.pulsereader.pulses(r)

    # @functools.cache
    def calc_max_excursion(self, n_limit: int = 10000, excursion_nsigma: float = 5) -> tuple[pl.DataFrame, float]:
        """Compute the maximum excursion from the median for each noise record, and store in dataframe."""

        def excursion2d(noise_trace: NDArray) -> NDArray:
            """Return the excursion (max - min) for each trace in a 2D array of traces."""
            return np.amax(noise_trace, axis=1) - np.amin(noise_trace, axis=1)

        noise_traces = self.load_raw(slice(n_limit))
        excursion = excursion2d(noise_traces)
        max_excursion = mass2.misc.outlier_resistant_nsigma_above_mid(excursion, nsigma=excursion_nsigma)
        df_noise2 = self.df.limit(n_limit).with_columns(excursion=excursion)
        return df_noise2, max_excursion

    def get_records_2d(
        self,
        n_limit: int = 10000,
        excursion_nsigma: float = 5,
        trunc_front: int = 0,
        trunc_back: int = 0,
    ) -> NDArray:
        """
        Return a 2D NumPy array of cleaned noise traces from the specified column.

        This method identifies noise traces with excursions below a threshold and
        optionally truncates the beginning and/or end of each trace.

        Parameters:
        ----------
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
        df_noise2, max_excursion = self.calc_max_excursion(n_limit, excursion_nsigma)
        noise_trace_is_clean = (
            df_noise2.with_row_index("pulse_idx").filter(pl.col("excursion") <= max_excursion)["pulse_idx"].to_numpy()
        )
        noise_traces_clean = self.load_raw()[noise_trace_is_clean]
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
        n_limit: int = 10000,
        excursion_nsigma: float = 5,
        trunc_front: int = 0,
        trunc_back: int = 0,
        skip_autocorr_if_length_over: int = 100_000,
    ) -> NoiseResult:
        """Compute and return the noise result from the noise traces."""
        records = self.get_records_2d(n_limit, excursion_nsigma, trunc_front, trunc_back)
        continuous = self.is_continuous and trunc_front == 0 and trunc_back == 0
        spectrum = mass2.core.noise_algorithms.calc_noise_result(
            records, continuous=continuous, dt=self.frametime_s, skip_autocorr_if_length_over=skip_autocorr_if_length_over
        )
        return spectrum

    def __hash__(self) -> int:
        """A hash function based on the object's id."""
        # needed to make functools.cache work
        # if self or self.anything is mutated, assumptions will be broken
        # and we may get nonsense results
        return hash(id(self))

    def __eq__(self, other: Any) -> bool:
        """Equality based on object identity."""
        return id(self) == id(other)

    @property
    def is_continuous(self) -> bool:
        "Whether this channel is continuous data (True) or triggered records with arbitrary gaps (False)."
        if "continuous" in self.header_df:
            return self.header_df["continuous"][0]
        return False

    @classmethod
    def from_ljh(cls, path: str | Path) -> "NoiseChannel":
        """Create a NoiseChannel by loading data from the given LJH file path."""
        ljh = mass2.LJHFile.open(path)
        df, header_df = ljh.to_polars()
        noise_channel = cls(df, header_df, header_df["Timebase"][0], pulsereader=ljh.pulsereader)
        return noise_channel
