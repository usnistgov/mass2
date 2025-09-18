"""
Data structures and methods for handling a group of microcalorimeter channels.
"""

from dataclasses import dataclass, field
import dataclasses
from collections.abc import Callable, Iterable
from numpy.typing import ArrayLike
from typing import Any
import polars as pl
import pylab as plt
import matplotlib
import numpy as np
import functools
import joblib
import traceback
import lmfit
import dill
import io
import os
import pathlib
from pathlib import Path
from zipfile import ZipFile

import mass2
from .channel import Channel, ChannelHeader, BadChannel
from ..calibration.fluorescence_lines import SpectralLine
from ..calibration.line_models import GenericLineModel, LineModelResult
from .recipe import Recipe
from . import ljhutil


@dataclass(frozen=True)  # noqa: PLR0904
class Channels:
    """A collection of microcalorimeter channels, with methods to operate in parallel on all channels."""

    channels: dict[int, Channel]
    description: str
    bad_channels: dict[int, BadChannel] = field(default_factory=dict)

    @property
    def ch0(self) -> Channel:
        """Return a representative Channel object for convenient exploration (the one with the lowest channel number)."""
        assert len(self.channels) > 0, "channels must be non-empty"
        return next(iter(self.channels.values()))

    def with_more_channels(self, more: "Channels") -> "Channels":
        """Return a Channels object with additional Channels in it.
        New channels with the same number will overrule existing ones.

        Parameters
        ----------
        more : Channels
            Another Channels object, to be added

        Returns
        -------
        Channels
            The replacement
        """
        channels = self.channels.copy()
        channels.update(more.channels)
        bad = self.bad_channels.copy()
        bad.update(more.bad_channels)
        descr = self.description + more.description + "\nWarning! created by with_more_channels()"
        return dataclasses.replace(self, channels=channels, bad_channels=bad, description=descr)

    @functools.cache
    def dfg(self, exclude: str = "pulse") -> pl.DataFrame:
        """Return a DataFrame containing good pulses from each channel. Excludes the given columns (default "pulse")."""
        # return a dataframe containing good pulses from each channel,
        # exluding "pulse" by default
        # and including column "ch_num"
        # the more common call should be to wrap this in a convenient plotter
        dfs = []
        for ch_num, channel in self.channels.items():
            df = channel.df.select(pl.exclude(exclude)).filter(channel.good_expr)
            # key_series = pl.Series("key", dtype=pl.Int64).extend_constant(key, len(df))
            assert ch_num == channel.header.ch_num
            ch_series = pl.Series("ch_num", dtype=pl.Int64).extend_constant(channel.header.ch_num, len(df))
            dfs.append(df.with_columns(ch_series))
        return pl.concat(dfs)

    def linefit(  # noqa: PLR0917
        self,
        line: float | str | SpectralLine | GenericLineModel,
        col: str,
        use_expr: pl.Expr = pl.lit(True),
        has_linear_background: bool = False,
        has_tails: bool = False,
        dlo: float = 50,
        dhi: float = 50,
        binsize: float = 0.5,
        params_update: lmfit.Parameters = lmfit.Parameters(),
    ) -> LineModelResult:
        """Perform a fit to one spectral line in the coadded histogram of the given column."""
        model = mass2.calibration.algorithms.get_model(line, has_linear_background=has_linear_background, has_tails=has_tails)
        pe = model.spect.peak_energy
        _bin_edges = np.arange(pe - dlo, pe + dhi, binsize)
        df_small = self.dfg().lazy().filter(use_expr).select(col).collect()
        bin_centers, counts = mass2.misc.hist_of_series(df_small[col], _bin_edges)
        params = model.guess(counts, bin_centers=bin_centers, dph_de=1)
        params["dph_de"].set(1.0, vary=False)
        print(f"before update {params=}")
        params = params.update(params_update)
        print(f"after update {params=}")
        result = model.fit(counts, params, bin_centers=bin_centers, minimum_bins_per_fwhm=3)
        result.set_label_hints(
            binsize=bin_centers[1] - bin_centers[0],
            ds_shortname=f"{len(self.channels)} channels, {self.description}",
            unit_str="eV",
            attr_str=col,
            states_hint=f"{use_expr=}",
            cut_hint="",
        )
        return result

    def plot_hist(self, col: str, bin_edges: ArrayLike, use_expr: pl.Expr = pl.lit(True), axis: plt.Axes | None = None) -> None:
        """Plot a histogram for the given column across all channels."""
        df_small = self.dfg().lazy().filter(use_expr).select(col).collect()
        ax = mass2.misc.plot_hist_of_series(df_small[col], bin_edges, axis)
        ax.set_title(f"{len(self.channels)} channels, {self.description}")

    def plot_hists(
        self,
        col: str,
        bin_edges: ArrayLike,
        group_by_col: bool,
        axis: plt.Axes | None = None,
        use_expr: pl.Expr | None = None,
        skip_none: bool = True,
    ) -> None:
        """
        Plots histograms for the given column, grouped by the specified column.

        Parameters:
        - col (str): The column name to plot.
        - bin_edges (array-like): The edges of the bins for the histogram.
        - group_by_col (str): The column name to group by. This is required.
        - axis (matplotlib.Axes, optional): The axis to plot on. If None, a new figure is created.
        """
        if axis is None:
            _, ax = plt.subplots()  # Create a new figure if no axis is provided
        else:
            ax = axis

        if use_expr is None:
            df_small = (self.dfg().lazy().select(col, group_by_col)).collect().sort(group_by_col, descending=False)
        else:
            df_small = (self.dfg().lazy().filter(use_expr).select(col, group_by_col)).collect().sort(group_by_col, descending=False)

        # Plot a histogram for each group
        for (group_name,), group_data in df_small.group_by(group_by_col, maintain_order=True):
            if group_name is None and skip_none:
                continue
            # Get the data for the column to plot
            values = group_data[col]
            # Plot the histogram for the current group
            if group_name == "EBIT":
                ax.hist(values, bins=bin_edges, alpha=0.9, color="k", label=str(group_name))
            else:
                ax.hist(values, bins=bin_edges, alpha=0.5, label=str(group_name))
            # bin_centers, counts = mass2.misc.hist_of_series(values, bin_edges)
            # plt.plot(bin_centers, counts, label=group_name)

        # Customize the plot
        ax.set_xlabel(str(col))
        ax.set_ylabel("Frequency")
        ax.set_title(f"Coadded Histogram of {col} grouped by {group_by_col}")

        # Add a legend to label the groups
        ax.legend(title=group_by_col)

        plt.tight_layout()

    def _limited_chan_list(self, limit: int | None = 20, channels: list[int] | None = None) -> list[int]:
        """A helper to get a list of channel numbers, limited to the given number if needed, and including only
        channel numbers from `channels` if not None."""
        limited_chan = list(self.channels.keys())
        if channels is not None:
            limited_chan = list(set(limited_chan).intersection(set(channels)))
            limited_chan.sort()
        if limit and len(limited_chan) > limit:
            limited_chan = limited_chan[:limit]
        return limited_chan

    def plot_filters(
        self,
        limit: int | None = 20,
        channels: list[int] | None = None,
        colormap: matplotlib.colors.Colormap = plt.cm.viridis,
        axis: plt.Axes | None = None,
    ) -> plt.Axes:
        """Plot the optimal filters for the channels in this Channels object.

        Parameters
        ----------
        limit : int | None, optional
            Plot at most this many filters if not None, by default 20
        channels : list[int] | None, optional
            Plot only channels with numbers in this list if not None, by default None
        colormap : matplotlib.colors.Colormap, optional
            The color scale to use, by default plt.cm.viridis
        axis : plt.Axes | None, optional
            A `plt.Axes` to plot on, or if None a new one, by default None

        Returns
        -------
        plt.Axes
            The `plt.Axes` containing the plot.
        """
        if axis is None:
            fig = plt.figure()
            axis = fig.subplots()

        plot_these_chan = self._limited_chan_list(limit, channels)
        n_expected = len(plot_these_chan)
        for i, ch_num in enumerate(plot_these_chan):
            ch = self.channels[ch_num]
            # The next line _assumes_ a 5-lag filter. Fix as needed.
            x = np.arange(ch.header.n_samples - 4) - ch.header.n_presamples + 2
            y = ch.last_filter
            if y is not None:
                plt.plot(x, y, color=colormap(i / n_expected), label=f"Chan {ch_num}")
        plt.legend()
        plt.xlabel("Samples after trigger")
        plt.title("Optimal filters")

    def plot_avg_pulses(
        self,
        limit: int | None = 20,
        channels: list[int] | None = None,
        colormap: matplotlib.colors.Colormap = plt.cm.viridis,
        axis: plt.Axes | None = None,
    ) -> plt.Axes:
        """Plot the average pulses (the signal model) for the channels in this Channels object.

        Parameters
        ----------
        limit : int | None, optional
            Plot at most this many filters if not None, by default 20
        channels : list[int] | None, optional
            Plot only channels with numbers in this list if not None, by default None
        colormap : matplotlib.colors.Colormap, optional
            The color scale to use, by default plt.cm.viridis
        axis : plt.Axes | None, optional
            A `plt.Axes` to plot on, or if None a new one, by default None

        Returns
        -------
        plt.Axes
            The `plt.Axes` containing the plot.
        """
        if axis is None:
            fig = plt.figure()
            axis = fig.subplots()

        plot_these_chan = self._limited_chan_list(limit, channels)
        n_expected = len(plot_these_chan)
        for i, ch_num in enumerate(plot_these_chan):
            ch = self.channels[ch_num]
            x = np.arange(ch.header.n_samples) - ch.header.n_presamples
            y = ch.last_avg_pulse
            if y is not None:
                plt.plot(x, y, color=colormap(i / n_expected), label=f"Chan {ch_num}")
        plt.legend()
        plt.xlabel("Samples after trigger")
        plt.title("Average pulses")

    def plot_noise_spectrum(
        self,
        limit: int | None = 20,
        channels: list[int] | None = None,
        colormap: matplotlib.colors.Colormap = plt.cm.viridis,
        axis: plt.Axes | None = None,
    ) -> plt.Axes:
        """Plot the noise power spectrum for the channels in this Channels object.

        Parameters
        ----------
        limit : int | None, optional
            Plot at most this many filters if not None, by default 20
        channels : list[int] | None, optional
            Plot only channels with numbers in this list if not None, by default None
        colormap : matplotlib.colors.Colormap, optional
            The color scale to use, by default plt.cm.viridis
        axis : plt.Axes | None, optional
            A `plt.Axes` to plot on, or if None a new one, by default None

        Returns
        -------
        plt.Axes
            The `plt.Axes` containing the plot.
        """
        if axis is None:
            fig = plt.figure()
            axis = fig.subplots()

        plot_these_chan = self._limited_chan_list(limit, channels)
        n_expected = len(plot_these_chan)
        for i, ch_num in enumerate(plot_these_chan):
            ch = self.channels[ch_num]
            freqpsd = ch.last_noise_psd
            if freqpsd is not None:
                freq, psd = freqpsd
                plt.plot(freq, psd, color=colormap(i / n_expected), label=f"Chan {ch_num}")
        plt.legend()
        plt.loglog()
        plt.xlabel("Frequency (Hz)")
        plt.title("Noise power spectral density")

    def plot_noise_autocorr(
        self,
        limit: int | None = 20,
        channels: list[int] | None = None,
        colormap: matplotlib.colors.Colormap = plt.cm.viridis,
        axis: plt.Axes | None = None,
    ) -> plt.Axes:
        """Plot the noise power autocorrelation for the channels in this Channels object.

        Parameters
        ----------
        limit : int | None, optional
            Plot at most this many filters if not None, by default 20
        channels : list[int] | None, optional
            Plot only channels with numbers in this list if not None, by default None
        colormap : matplotlib.colors.Colormap, optional
            The color scale to use, by default plt.cm.viridis
        axis : plt.Axes | None, optional
            A `plt.Axes` to plot on, or if None a new one, by default None

        Returns
        -------
        plt.Axes
            The `plt.Axes` containing the plot.
        """
        if axis is None:
            fig = plt.figure()
            axis = fig.subplots()

        plot_these_chan = self._limited_chan_list(limit, channels)
        n_expected = len(plot_these_chan)
        for i, ch_num in enumerate(plot_these_chan):
            ch = self.channels[ch_num]
            ac = ch.last_noise_autocorrelation
            if ac is not None:
                color = colormap(i / n_expected)
                plt.plot(ac, color=color, label=f"Chan {ch_num}")
                plt.plot(0, ac[0], "o", color=color)
        plt.legend()
        plt.xlabel("Lags")
        plt.title("Noise autocorrelation")

    def map(self, f: Callable, allow_throw: bool = False) -> "Channels":
        """Map function `f` over all channels, returning a new Channels object containing the new Channel objects."""
        new_channels = {}
        new_bad_channels = {}
        for key, channel in self.channels.items():
            try:
                new_channels[key] = f(channel)
            except KeyboardInterrupt as kint:
                raise kint
            except Exception as ex:
                error_type: type = type(ex)
                error_message: str = str(ex)
                backtrace: str = traceback.format_exc()
                if allow_throw:
                    raise
                print(f"{key=} {channel=} failed the step {f}")
                print(f"{error_type=}")
                print(f"{error_message=}")
                new_bad_channels[key] = channel.as_bad(error_type, error_message, backtrace)
        new_bad_channels = mass2.misc.merge_dicts_ordered_by_keys(self.bad_channels, new_bad_channels)

        return Channels(new_channels, self.description, bad_channels=new_bad_channels)

    def set_bad(self, ch_num: int, msg: str, require_ch_num_exists: bool = True) -> "Channels":
        """Return a copy of this Channels object with the given channel number marked as bad."""
        new_channels = {}
        new_bad_channels = {}
        if require_ch_num_exists:
            assert ch_num in self.channels.keys(), f"{ch_num} can't be set bad because it does not exist"
        for key, channel in self.channels.items():
            if key == ch_num:
                new_bad_channels[key] = channel.as_bad(None, msg, None)
            else:
                new_channels[key] = channel
        return Channels(new_channels, self.description, bad_channels=new_bad_channels)

    def linefit_joblib(self, line: str, col: str, prefer: str = "threads", n_jobs: int = 4) -> LineModelResult:
        """No one but Galen understands this function."""

        def work(key: int) -> LineModelResult:
            """A unit of parallel work: fit line to one channel."""
            channel = self.channels[key]
            return channel.linefit(line, col)

        parallel = joblib.Parallel(n_jobs=n_jobs, prefer=prefer)  # its not clear if threads are better.... what blocks the gil?
        results = parallel(joblib.delayed(work)(key) for key in self.channels.keys())
        return results

    def __hash__(self) -> int:
        """Hash based on the object's id (identity)."""
        # needed to make functools.cache work
        # if self or self.anything is mutated, assumptions will be broken
        # and we may get nonsense results
        return hash(id(self))

    def __eq__(self, other: Any) -> bool:
        """Equality test based on object identity."""
        return id(self) == id(other)

    @classmethod
    def from_ljh_path_pairs(cls, pulse_noise_pairs: Iterable[tuple[str, str]], description: str) -> "Channels":
        """
        Create a :class:`Channels` instance from pairs of LJH files.

        Args:
            pulse_noise_pairs (List[Tuple[str, str]]):
                A list of `(pulse_path, noise_path)` tuples, where each entry contains
                the file path to a pulse LJH file and its corresponding noise LJH file.
            description (str):
                A human-readable description for the resulting Channels object.

        Returns:
            Channels:
                A Channels object with one :class:`Channel` per `(pulse_path, noise_path)` pair.

        Raises:
            AssertionError:
                If two input files correspond to the same channel number.

        Notes:
            Each channel is created via :meth:`Channel.from_ljh`.
            The channel number is taken from the LJH file header and used as the key
            in the returned Channels mapping.

        Examples:
            >>> pairs = [
            ...     ("datadir/run0000_ch0000.ljh", "datadir/run0001_ch0000.ljh"),
            ...     ("datadir/run0000_ch0001.ljh", "datadir/run0001_ch0001.ljh"),
            ... ]
            >>> channels = Channels.from_ljh_path_pairs(pairs, description="Test run")
            >>> list(channels.keys())
            [0, 1]
        """
        channels: dict[int, Channel] = {}
        for pulse_path, noise_path in pulse_noise_pairs:
            channel = Channel.from_ljh(pulse_path, noise_path)
            assert channel.header.ch_num not in channels.keys()
            channels[channel.header.ch_num] = channel
        return cls(channels, description)

    @classmethod
    def from_off_paths(cls, off_paths: Iterable[str | Path], description: str) -> "Channels":
        """Create an instance from a sequence of OFF-file paths"""
        channels = {}
        for path in off_paths:
            ch = Channel.from_off(mass2.core.OffFile(str(path)))
            channels[ch.header.ch_num] = ch
        return cls(channels, description)

    @classmethod
    def from_ljh_folder(
        cls,
        pulse_folder: str | Path,
        noise_folder: str | Path | None = None,
        limit: int | None = None,
        exclude_ch_nums: list[int] | None = None,
    ) -> "Channels":
        """Create an instance from a directory of LJH files."""
        assert os.path.isdir(pulse_folder), f"{pulse_folder=} {noise_folder=}"
        pulse_folder = str(pulse_folder)
        if exclude_ch_nums is None:
            exclude_ch_nums = []
        if noise_folder is None:
            paths = ljhutil.find_ljh_files(pulse_folder, exclude_ch_nums=exclude_ch_nums)
            if limit is not None:
                paths = paths[:limit]
            pairs = [(path, "") for path in paths]
        else:
            assert os.path.isdir(noise_folder), f"{pulse_folder=} {noise_folder=}"
            noise_folder = str(noise_folder)
            pairs = ljhutil.match_files_by_channel(pulse_folder, noise_folder, limit=limit, exclude_ch_nums=exclude_ch_nums)
        description = f"from_ljh_folder {pulse_folder=} {noise_folder=}"
        print(f"{description}")
        print(f"   from_ljh_folder has {len(pairs)} pairs")
        data = cls.from_ljh_path_pairs(pairs, description)
        print(f"   and the Channels obj has {len(data.channels)} pairs")
        return data

    def get_an_ljh_path(self) -> Path:
        """Return the path to a representative one of the LJH files used to create this Channels object."""
        return pathlib.Path(self.ch0.header.df["Filename"][0])

    def get_path_in_output_folder(self, filename: str | Path) -> Path:
        """Return a path in an output folder named like the run number, sibling to the LJH folder."""
        ljh_path = self.get_an_ljh_path()
        base_name, _ = ljh_path.name.split("_chan")
        date, run_num = base_name.split("_run")  # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = ljh_path.parent.parent / f"{run_num}mass2_output"
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir / filename

    def get_experiment_state_df(self, experiment_state_path: str | Path | None = None) -> pl.DataFrame:
        """Return a DataFrame containing experiment state information,
        loading from the given path or (if None) inferring it from an LJH file."""
        if experiment_state_path is None:
            ljh_path = self.get_an_ljh_path()
            experiment_state_path = ljhutil.experiment_state_path_from_ljh_path(ljh_path)
        df = pl.read_csv(experiment_state_path, new_columns=["unixnano", "state_label"])
        # _col0, _col1 = df.columns
        df_es = df.select(pl.from_epoch("unixnano", time_unit="ns").dt.cast_time_unit("us").alias("timestamp"))
        # strip whitespace from state_label column
        sl_series = df.select(pl.col("state_label").str.strip_chars()).to_series()
        df_es = df_es.with_columns(state_label=pl.Series(values=sl_series, dtype=pl.Categorical))
        return df_es

    def with_experiment_state_by_path(self, experiment_state_path: str | None = None) -> "Channels":
        """Return a copy of this Channels object with experiment state information added,
        loaded from the given path."""
        df_es = self.get_experiment_state_df(experiment_state_path)
        return self.with_experiment_state_df(df_es)

    def with_external_trigger_by_path(self, path: str | None = None) -> "Channels":
        """Return a copy of this Channels object with external trigger information added, loaded
        from the given path or EVENTUALLY (if None) inferring it from an LJH file (not yet implemented)."""
        if path is None:
            raise NotImplementedError("cannot infer external trigger path yet")
        with open(path, "rb") as _f:
            _header_line = _f.readline()  # read the one header line before opening the binary data
            external_trigger_subframe_count = np.fromfile(_f, "int64")
        df_ext = pl.DataFrame({
            "subframecount": external_trigger_subframe_count,
        })
        return self.with_external_trigger_df(df_ext)

    def with_external_trigger_df(self, df_ext: pl.DataFrame) -> "Channels":
        """Return a copy of this Channels object with external trigger information added to each Channel,
        found from the given DataFrame."""

        def with_etrig_df(channel: Channel) -> Channel:
            """Return a copy of one Channel object with external trigger information added to it"""
            return channel.with_external_trigger_df(df_ext)

        return self.map(with_etrig_df)

    def with_experiment_state_df(self, df_es: pl.DataFrame) -> "Channels":
        """Return a copy of this Channels object with experiment state information added to each Channel,
        found from the given DataFrame."""
        # this is not as performant as making use_exprs for states
        # and using .set_sorted on the timestamp column
        ch2s = {}
        for ch_num, ch in self.channels.items():
            ch2s[ch_num] = ch.with_experiment_state_df(df_es)
        return Channels(ch2s, self.description)

    def with_steps_dict(self, steps_dict: dict[int, Recipe]) -> "Channels":
        """Return a copy of this Channels object with the given Recipe objects added to each Channel."""

        def load_recipes(channel: Channel) -> Channel:
            """Return a copy of one Channel object with Recipe steps added to it"""
            try:
                steps = steps_dict[channel.header.ch_num]
            except KeyError:
                raise Exception("steps dict did not contain steps for this ch_num")
            return channel.with_steps(steps)

        return self.map(load_recipes)

    def save_recipes(
        self, filename: str, required_fields: str | Iterable[str] | None = None, drop_debug: bool = True
    ) -> dict[int, Recipe]:
        """Pickle a dictionary (one entry per channel) of Recipe objects.

        If you want to save a "recipe", a minimal series of steps required to reproduce the required field(s),
        then set `required_fields` to be a list/tuple/set of DataFrame column names (or a single column name)
        whose production from raw data should be possible.

        Parameters
        ----------
        filename : str
            Filename to store recipe in, typically of the form "*.pkl"
        required_fields : str | Iterable[str] | None
            The field (str) or fields (Iterable[str]) that the recipe should be able to generate from a raw LJH file.
            Drop all steps that do not lead (directly or indireactly) to producing this field or these fields.
            If None, then preserve all steps (default None).
        drop_debug: bool
            Whether to remove debugging-related data from each `RecipeStep`, if the subclass supports this (via the
            `RecipeStep.drop_debug() method).

        Returns
        -------
        dict
            Dictionary with keys=channel numbers, values=the (possibly trimmed and debug-dropped) Recipe objects.
        """
        steps = {}
        for channum, ch in self.channels.items():
            steps[channum] = ch.steps.trim_dead_ends(required_fields=required_fields, drop_debug=drop_debug)
        mass2.misc.pickle_object(steps, filename)
        return steps

    def load_recipes(self, filename: str) -> "Channels":
        """Return a copy of this Channels object with Recipe objects loaded from the given pickle file
        and applied to each Channel."""
        steps = mass2.misc.unpickle_object(filename)
        return self.with_steps_dict(steps)

    def parent_folder_path(self) -> pathlib.Path:
        """Return the parent folder of the LJH files used to create this Channels object. Specifically, the
        `self.ch0` channel's directory is used (normally the answer would be the same for all channels)."""
        parent_folder_path = pathlib.Path(self.ch0.header.df["Filename"][0]).parent.parent
        print(f"{parent_folder_path=}")
        return parent_folder_path

    def concat_data(self, other_data: "Channels") -> "Channels":
        """Return a new Channels object with data from this and the other Channels object concatenated together.
        Only channels that exist in both objects are included in the result."""
        # sorting here to show intention, but I think set is sorted by insertion order as
        # an implementation detail so this may not do anything
        ch_nums = sorted(list(set(self.channels.keys()).intersection(other_data.channels.keys())))
        new_channels = {}
        for ch_num in ch_nums:
            ch = self.channels[ch_num]
            other_ch = other_data.channels[ch_num]
            combined_df = mass2.core.misc.concat_dfs_with_concat_state(ch.df, other_ch.df)
            new_ch = ch.with_replacement_df(combined_df)
            new_channels[ch_num] = new_ch
        return mass2.Channels(new_channels, self.description + other_data.description)

    @classmethod
    def from_df(
        cls,
        df_in: pl.DataFrame,
        frametime_s: float,
        n_presamples: int,
        n_samples: int,
        description: str = "from Channels.channels_from_df",
    ) -> "Channels":
        """Create a Channels object from a single DataFrame that holds data from multiple channels."""
        # requres a column named "ch_num" containing the channel number
        keys_df: dict[tuple, pl.DataFrame] = df_in.partition_by(by=["ch_num"], as_dict=True)
        dfs: dict[int, pl.DataFrame] = {keys[0]: df for (keys, df) in keys_df.items()}
        channels: dict[int, Channel] = {}
        for ch_num, df in dfs.items():
            channels[ch_num] = Channel(
                df,
                header=ChannelHeader(
                    description="from df",
                    data_source=None,
                    ch_num=ch_num,
                    frametime_s=frametime_s,
                    n_presamples=n_presamples,
                    n_samples=n_samples,
                    df=df,
                ),
                npulses=len(df),
            )
        return Channels(channels, description)

    def save_analysis(self, zip_path: Path | str, overwrite: bool = False) -> None:
        """Save an analysis-in-progress completely to a zip file, only tested for ljh backed channels so far

        Parameters
        ----------
        path : Path | str
            Directory to save work in. If it doesn't exist, its parent should.
        overwrite : bool, optional
            If `path` exists, whether to overwrite it, by default False
        """
        zip_path = pathlib.Path(zip_path)
        if zip_path.suffix != ".zip":
            zip_path = zip_path.with_suffix(".zip")

        if os.path.exists(zip_path) and not overwrite:
            raise ValueError(f"File exists; use `save_analysis(...overwrite=True)` to overwrite the existing {zip_path=}")

        def store_dataframe_to_parquet_and_return_pickleable_channel(ch: Channel, zf: ZipFile, parquet_path: str) -> Channel:
            """Store the `ch.df` to a parquet file of the given name in the ZipFile (open for writing).
            Prepare `ch` for pickling by removing its dataframe and dataframe history, and stripping debug info from `steps`

            Parameters
            ----------
            ch : Channel
                The channel to modify by storing dataframe to parquet and removing it.
            zf : ZipFile
                A ZipFile object currently open for writing.
            parquet_file : str
                The name to use for storing the parquet file within `zf`

            Returns
            -------
            Channel
                A copy of `ch` amenable to pickling with the dataframe and dataframe history removed and with trimmed steps.
            """
            # Don't store the memmapped LJH file info (if present) in the Parquet file
            df = ch.df.select(pl.exclude(["pulse", "timestamp", "subframecount"]))
            buffer = io.BytesIO()
            df.write_parquet(buffer)
            zf.writestr(parquet_path, buffer.getvalue())
            steps = ch.steps.trim_debug_info()
            return dataclasses.replace(ch, df=pl.DataFrame(), df_history=[], noise=None, steps=steps)

        with ZipFile(str(zip_path), "w") as zf:
            channels = {}
            bad_channels = {}
            for ch_num, ch in self.channels.items():
                parquet_path = f"data_chan{ch_num:04d}.parquet"
                channels[ch_num] = store_dataframe_to_parquet_and_return_pickleable_channel(ch, zf, parquet_path)
            for ch_num, badch in self.bad_channels.items():
                parquet_path = f"data_bad_chan{ch_num:04d}.parquet"
                ch = store_dataframe_to_parquet_and_return_pickleable_channel(badch.ch, zf, parquet_path)
                bad_channels[ch_num] = dataclasses.replace(badch, ch=ch)
            data = dataclasses.replace(self, channels=channels, bad_channels=bad_channels)
            pickle_file = "data_all.pkl"
            zf.writestr(pickle_file, dill.dumps(data))

    @staticmethod
    def load_analysis(path: Path | str) -> "Channels":
        """Load an analysis-in-progress from a zipfile

        Parameters
        ----------
        path : Path | str
            Zipfile that work was saved in.
        """
        path = pathlib.Path(path)
        path.exists() and path.is_file()

        def restore_dataframe(ch: Channel, df: pl.DataFrame) -> Channel:
            """Take a channel and replace its dataframe with the given one, loaded from a parquet file

            Parameters
            ----------
            ch : Channel
                A channel, loaded from a pickle file, with an empty dataframe
            df : DataFrame
                A replacement dataframe for the existing one (typically, the existing one is empty)

            Returns
            -------
            Channel
                The Channel `ch` but with `ch.df` updated, including any raw data backed by an LJH file
            """
            # If this channel was based on an LJH file, restore columns from the LJH file to the dataframe.
            print(f"Joe sees {ch.header.data_source=}")
            if ch.header.data_source is not None:
                ljh_path = ch.header.data_source
                if ljh_path.endswith(".ljh") or ljh_path.endswith(".noi"):
                    raw_ch = Channel.from_ljh(ljh_path)
                    df = pl.concat([df, raw_ch.df], how="horizontal")
            df_history = [df] * len(ch.steps)
            return dataclasses.replace(ch, df=df, df_history=df_history)

        with ZipFile(path, "r") as zf:
            pickle_file = "data_all.pkl"
            pickle_bytes = zf.read(pickle_file)
            data: Channels = dill.loads(pickle_bytes)

            restored_channels = {}
            for ch_num, ch in data.channels.items():
                parquet_file = f"data_chan{ch_num:04d}.parquet"
                df = pl.read_parquet(zf.read(parquet_file))
                restored_channels[ch_num] = restore_dataframe(ch, df)

            restored_bad_channels = {}
            for ch_num, badch in data.bad_channels.items():
                parquet_file = f"data_bad_chan{ch_num:04d}.parquet"
                df = pl.read_parquet(zf.read(parquet_file))
                ch = restore_dataframe(badch.ch, df)
                restored_bad_channels[ch_num] = dataclasses.replace(badch, ch=ch)

            return dataclasses.replace(data, channels=restored_channels, bad_channels=restored_bad_channels)
