"""
Hold a class to represent a channel with noise data only, and to analyze its noise characteristics.
"""

import dataclasses
import warnings
from typing import Any
from numpy.typing import NDArray
from pathlib import Path
from dataclasses import dataclass
import polars as pl
import numpy as np
import mass2
from .noise_algorithms import NoiseResult


@dataclass(frozen=True)
class NoiseChannel:
    """A class to represent a channel with noise data only, and to analyze its noise characteristics."""

    df: pl.DataFrame  # DO NOT MUTATE THIS!!!
    header_df: pl.DataFrame  # DO NOT MUTATE THIS!!
    frametime_s: float

    # @functools.cache
    def calc_max_excursion(
        self, trace_col_name: str = "pulse", n_limit: int = 10000, excursion_nsigma: float = 5
    ) -> tuple[pl.DataFrame, float]:
        """Compute the maximum excursion from the median for each noise record, and store in dataframe."""

        def excursion2d(noise_trace: NDArray) -> float:
            """Return the excursion (max - min) for each trace in a 2D array of traces."""
            return np.amax(noise_trace, axis=1) - np.amin(noise_trace, axis=1)

        noise_traces = self.df.limit(n_limit)[trace_col_name].to_numpy()
        excursion = excursion2d(noise_traces)
        max_excursion = mass2.misc.outlier_resistant_nsigma_above_mid(excursion, nsigma=excursion_nsigma)
        df_noise2 = self.df.limit(n_limit).with_columns(excursion=excursion)
        return df_noise2, max_excursion

    def get_records_2d(
        self,
        trace_col_name: str = "pulse",
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
        trace_col_name: str = "pulse",
        n_limit: int = 10000,
        excursion_nsigma: float = 5,
        trunc_front: int = 0,
        trunc_back: int = 0,
        skip_autocorr_if_length_over: int = 100_000,
    ) -> NoiseResult:
        """Compute and return the noise result from the noise traces."""
        records = self.get_records_2d(trace_col_name, n_limit, excursion_nsigma, trunc_front, trunc_back)
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
    def _load_from_ipc_cache(
        cls,
        data_ipc_path: Path,
        header_ipc_path: Path,
        path: "str | Path",
        load_pulses: bool,
    ) -> "NoiseChannel":
        if load_pulses:
            df = pl.read_ipc(data_ipc_path, memory_map=True)
        else:
            df = pl.scan_ipc(data_ipc_path).select(pl.exclude("pulse")).collect()
        header_df = pl.read_ipc(header_ipc_path)
        if "source_file" not in df.columns:
            df = df.with_columns(
                pl.lit(str(path)).alias("source_file").cast(pl.Categorical),
                pl.int_range(0, len(df), dtype=pl.Int64).alias("source_id"),
            )
        elif "source_id" not in df.columns:
            df = df.with_columns(pl.int_range(0, len(df), dtype=pl.Int64).alias("source_id"))
        return cls(df, header_df, header_df["Timebase"][0])

    @classmethod
    def from_ljh(
        cls,
        path: str | Path,
        keep_posix_usec: bool = False,
        use_cache: bool = True,
        generate_cache: bool = False,
        load_pulses: bool = True,
    ) -> "NoiseChannel":
        """Create a NoiseChannel by loading data from the given LJH file path."""
        path_obj = Path(path)
        data_ipc_path = path_obj.with_suffix(".ipc")
        header_ipc_path = path_obj.with_suffix(".header.ipc")
        cache_exists = data_ipc_path.exists() and header_ipc_path.exists()
        cache_is_valid = False
        if cache_exists:
            if path_obj.stat().st_mtime < data_ipc_path.stat().st_mtime:
                cache_is_valid = True
            else:
                print(f"Cache for {path_obj.name} is out of date. Regenerating...")
        if use_cache and cache_is_valid:
            try:
                return cls._load_from_ipc_cache(data_ipc_path, header_ipc_path, path, load_pulses)
            except Exception as e:
                print(f"Warning: Corrupted cache detected for {path_obj.name} ({e}). Falling back to raw LJH.")
        ljh = mass2.LJHFile.open(path)
        df, header_df = ljh.to_polars(keep_posix_usec)
        if "source_file" not in df.columns:
            df = df.with_columns(
                pl.lit(str(path)).alias("source_file").cast(pl.Categorical),
                pl.int_range(0, len(df), dtype=pl.Int64).alias("source_id"),
            )
        elif "source_id" not in df.columns:
            df = df.with_columns(pl.int_range(0, len(df), dtype=pl.Int64).alias("source_id"))
        if generate_cache:
            print(f"Generating IPC cache for {path_obj.name}...")
            tmp_data_path = Path(str(data_ipc_path) + ".tmp")
            tmp_header_path = Path(str(header_ipc_path) + ".tmp")
            df.write_ipc(tmp_data_path, compression="uncompressed")
            header_df.write_ipc(tmp_header_path, compression="uncompressed")
            tmp_data_path.replace(data_ipc_path)
            tmp_header_path.replace(header_ipc_path)
            if use_cache:
                del df, header_df
                if load_pulses:
                    df = pl.read_ipc(data_ipc_path, memory_map=True)
                else:
                    df = pl.scan_ipc(data_ipc_path).select(pl.exclude("pulse")).collect()
                header_df = pl.read_ipc(header_ipc_path)
        if not load_pulses and "pulse" in df.columns:
            df = df.drop("pulse", strict=False)
        return cls(df, header_df, header_df["Timebase"][0])

    def load_pulse(self, use_cache: bool = True, generate_cache: bool = False) -> "NoiseChannel":
        """Rehydrate the noise pulse column lazily."""
        if "pulse" in self.df.columns:
            return self
        src = None
        if "source_file" in self.df.columns and len(self.df) > 0:
            src = self.df["source_file"][0]
        elif "Filename" in self.header_df.columns and len(self.header_df) > 0:
            src = self.header_df["Filename"][0]
        elif "filename" in self.header_df.columns and len(self.header_df) > 0:
            src = self.header_df["filename"][0]
        if src is not None and isinstance(src, str):
            if src.endswith(".ljh") or src.endswith(".noi"):
                temp_chan = self.__class__.from_ljh(src, use_cache=use_cache, generate_cache=generate_cache, load_pulses=True)
                if "source_id" in self.df.columns:
                    source_ids = self.df["source_id"].to_numpy()
                    pulse_series = temp_chan.df["pulse"].gather(source_ids)
                    return dataclasses.replace(self, df=self.df.with_columns(pulse_series))
                if len(self.df) == len(temp_chan.df):
                    return dataclasses.replace(self, df=self.df.with_columns(temp_chan.df["pulse"]))
                warnings.warn(
                    "NoiseChannel has different length than source file and no source_id; "
                    "cannot splice pulse column. Returning fresh channel (computed columns lost).",
                    RuntimeWarning,
                    stacklevel=2,
                )
                return dataclasses.replace(self, df=temp_chan.df)
        return self

    def drop_pulse(self) -> "NoiseChannel":
        """Drop the heavy pulse array from RAM."""
        if "pulse" in self.df.columns:
            return dataclasses.replace(self, df=self.df.drop("pulse", strict=False))
        return self
