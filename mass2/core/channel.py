"""
Data structures and methods for handling a single microcalorimeter channel's pulse data and metadata.
"""

from dataclasses import dataclass, field
import dataclasses
from typing import Any
from numpy.typing import ArrayLike, NDArray
from collections.abc import Callable, Iterable
import os
import lmfit
import polars as pl
import pylab as plt
from matplotlib.backend_bases import MouseEvent
import marimo as mo
import functools
import numpy as np
import time
from pathlib import Path

from .noise_channel import NoiseChannel
from .recipe import Recipe, RecipeStep, SummarizeStep
from .drift_correction import DriftCorrectStep
from .optimal_filtering import FilterMaker
from .filter_steps import OptimalFilterStep
from .multifit import MultiFit, MultiFitQuadraticGainStep, MultiFitMassCalibrationStep
from .misc import alwaysTrue
from .offfiles import OffFile
from . import misc
from ..calibration.line_models import GenericLineModel, LineModelResult
from ..calibration.fluorescence_lines import SpectralLine
import mass2


@dataclass(frozen=True)
class ChannelHeader:
    """Metadata about a Channel, of the sort read from file header."""

    description: str  # filename or date/run number, etc
    data_source: str | None  # complete file path, if read from a file
    ch_num: int
    frametime_s: float
    n_presamples: int
    n_samples: int
    df: pl.DataFrame = field(repr=False)

    @classmethod
    def from_ljh_header_df(cls, df: pl.DataFrame) -> "ChannelHeader":
        """Construct from the LJH header dataframe as returned by LJHFile.to_polars()"""
        filepath = df["Filename"][0]
        return cls(
            description=os.path.split(filepath)[-1],
            data_source=filepath,
            ch_num=df["Channel"][0],
            frametime_s=df["Timebase"][0],
            n_presamples=df["Presamples"][0],
            n_samples=df["Total Samples"][0],
            df=df,
        )


@dataclass(frozen=True)  # noqa: PLR0904
class Channel:
    """A single microcalorimeter channel's pulse data and associated metadata."""

    df: pl.DataFrame = field(repr=False)
    header: ChannelHeader = field(repr=True)
    npulses: int
    subframediv: int | None = None
    noise: NoiseChannel | None = field(default=None, repr=False)
    good_expr: pl.Expr = field(default_factory=alwaysTrue)
    df_history: list[pl.DataFrame] = field(default_factory=list, repr=False)
    steps: Recipe = field(default_factory=Recipe.new_empty, repr=False)
    steps_elapsed_s: list[float] = field(default_factory=list)
    transform_raw: Callable | None = None

    @property
    def shortname(self) -> str:
        """A short name for this channel, suitable for plot titles."""
        return self.header.description

    def mo_stepplots(self) -> mo.ui.dropdown:
        """Marimo UI element to choose and display step plots, with a dropdown to choose channel number."""
        desc_ind = {step.description: i for i, step in enumerate(self.steps)}
        first_non_summarize_step = self.steps[0]
        for step in self.steps:
            if isinstance(step, SummarizeStep):
                continue
            first_non_summarize_step = step
            break
        mo_ui = mo.ui.dropdown(
            desc_ind,
            value=first_non_summarize_step.description,
            label=f"choose step for ch {self.header.ch_num}",
        )

        def show() -> mo.Html:
            """Show the selected step plot."""
            return self._mo_stepplots_explicit(mo_ui)

        def step_ind() -> Any:
            """Get the selected step index from the dropdown item, if any."""
            return mo_ui.value

        mo_ui.show = show
        mo_ui.step_ind = step_ind
        return mo_ui

    def _mo_stepplots_explicit(self, mo_ui: mo.ui.dropdown) -> mo.Html:
        """Marimo UI element to choose and display step plots."""
        step_ind = mo_ui.value
        self.step_plot(step_ind)
        fig = plt.gcf()
        return mo.vstack([mo_ui, misc.show(fig)])

    def get_step(self, index: int) -> tuple[RecipeStep, int]:
        """Get the step at the given index, supporting negative indices."""
        # normalize the index to a positive index
        if index < 0:
            index = len(self.steps) + index
        step = self.steps[index]
        return step, index

    def step_plot(self, step_ind: int, **kwargs: Any) -> plt.Axes:
        """Make a debug plot for the given step index, supporting negative indices."""
        step, step_ind = self.get_step(step_ind)
        if step_ind + 1 == len(self.df_history):
            df_after = self.df
        else:
            df_after = self.df_history[step_ind + 1]
        return step.dbg_plot(df_after, **kwargs)

    def hist(
        self,
        col: str,
        bin_edges: ArrayLike,
        use_good_expr: bool = True,
        use_expr: pl.Expr = pl.lit(True),
    ) -> tuple[NDArray, NDArray]:
        """Compute a histogram of the given column, optionally filtering by good_expr and use_expr."""
        if use_good_expr and self.good_expr is not True:
            # True doesn't implement .and_, haven't found a exper literal equivalent that does
            # so we special case True
            filter_expr = self.good_expr.and_(use_expr)
        else:
            filter_expr = use_expr

        # Group by the specified column and filter using good_expr
        df_small = (self.df.lazy().filter(filter_expr).select(col)).collect()

        values = df_small[col]
        bin_centers, counts = misc.hist_of_series(values, bin_edges)
        return bin_centers, counts

    def plot_hist(
        self,
        col: str,
        bin_edges: ArrayLike,
        axis: plt.Axes | None = None,
        use_good_expr: bool = True,
        use_expr: pl.Expr = pl.lit(True),
    ) -> tuple[NDArray, NDArray]:
        """Compute and plot a histogram of the given column, optionally filtering by good_expr and use_expr."""
        if axis is None:
            _, ax = plt.subplots()  # Create a new figure if no axis is provided
        else:
            ax = axis

        bin_centers, counts = self.hist(col, bin_edges=bin_edges, use_good_expr=use_good_expr, use_expr=use_expr)
        _, step_size = misc.midpoints_and_step_size(bin_edges)
        plt.step(bin_centers, counts, where="mid")

        # Customize the plot
        ax.set_xlabel(str(col))
        ax.set_ylabel(f"Counts per {step_size:.02f} unit bin")
        ax.set_title(f"Histogram of {col} for {self.shortname}")

        plt.tight_layout()
        return bin_centers, counts

    def plot_hists(
        self,
        col: str,
        bin_edges: ArrayLike,
        group_by_col: str,
        axis: plt.Axes | None = None,
        use_good_expr: bool = True,
        use_expr: pl.Expr = pl.lit(True),
        skip_none: bool = True,
    ) -> tuple[NDArray, dict[str, NDArray]]:
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

        if use_good_expr and self.good_expr is not True:
            # True doesn't implement .and_, haven't found a exper literal equivalent that does
            # so we special case True
            filter_expr = self.good_expr.and_(use_expr)
        else:
            filter_expr = use_expr

        # Group by the specified column and filter using good_expr
        df_small = (self.df.lazy().filter(filter_expr).select(col, group_by_col)).collect().sort(group_by_col, descending=False)

        # Plot a histogram for each group
        counts_dict: dict[str, NDArray] = {}
        for (group_name,), group_data in df_small.group_by(group_by_col, maintain_order=True):
            if group_name is None and skip_none:
                continue
            # Get the data for the column to plot
            values = group_data[col]
            _, step_size = misc.midpoints_and_step_size(bin_edges)
            bin_centers, counts = misc.hist_of_series(values, bin_edges)
            group_name_str = str(group_name)
            counts_dict[group_name_str] = counts
            plt.step(bin_centers, counts, where="mid", label=group_name_str)
            # Plot the histogram for the current group
            # if group_name == "EBIT":
            #     ax.hist(values, bins=bin_edges, alpha=0.9, color="k", label=group_name_str)
            # else:
            #     ax.hist(values, bins=bin_edges, alpha=0.5, label=group_name_str)
            # bin_centers, counts = misc.hist_of_series(values, bin_edges)
            # plt.plot(bin_centers, counts, label=group_name)
        # Customize the plot
        ax.set_xlabel(str(col))
        if len(counts_dict)>0:
            ax.set_ylabel(f"Counts per {step_size:.02f} unit bin")
        ax.set_title(f"Histogram of {col} grouped by {group_by_col}")

        # Add a legend to label the groups
        ax.legend(title=group_by_col)

        plt.tight_layout()
        return bin_centers, counts_dict

    def plot_scatter(
        self,
        x_col: str,
        y_col: str,
        color_col: str | None = None,
        use_expr: pl.Expr = pl.lit(True),
        use_good_expr: bool = True,
        skip_none: bool = True,
        ax: plt.Axes | None = None,
        annotate: bool = False,
        decimate_by_n: int = 1
    ) -> None:
        """Generate a scatter plot of `y_col` vs `x_col`, optionally colored by `color_col`.

        Parameters
        ----------
        x_col : str
            Name of the column to put on the x axis
        y_col : str
            Name of the column to put on the y axis
        color_col : str | None, optional
            Name of the column to color points by (generally a category like "state_label"), by default None
        use_expr : pl.Expr, optional
            An expression to select plottable points, by default pl.lit(True)
        use_good_expr : bool, optional
            Whether to apply the object's `good_expr` before plotting, by default True
        skip_none : bool, optional
            Whether to skip color categories with no name, by default True
        ax : plt.Axes | None, optional
            Axes to plot on, by default None
        annotate : bool, optional
            Whether to annotate points that are hovered over or clicked on by the mouse, by default True
        """
        if ax is None:
            fig = plt.figure()
            ax = plt.gca()
        plt.sca(ax)  # set current axis so I can use plt api
        fig = plt.gcf()
        filter_expr = use_expr
        if use_good_expr:
            filter_expr = self.good_expr.and_(use_expr)
        index_name = "pulse_idx"
        # Caused errors in Polars 1.35 if this was "index". See issue #85.

        columns_to_keep = [x_col, y_col, index_name]
        if color_col is not None:
            columns_to_keep.append(color_col)
        df_small = self.df.lazy().with_row_index(name=index_name).filter(filter_expr).select(*columns_to_keep).gather_every(decimate_by_n).collect()
        lines_pnums: list[tuple[plt.Line2D, pl.Series]] = []

        for (name,), data in df_small.group_by(color_col, maintain_order=True):
            if name is None and skip_none and color_col is not None:
                continue
            (line,) = plt.plot(
                data.select(x_col).to_series(),
                data.select(y_col).to_series(),
                ".",
                label=name,
            )
            lines_pnums.append((line, data.select(index_name).to_series()))

        if annotate:
            annotation = ax.annotate(
                "",
                xy=(0, 0),
                xytext=(-20, 20),
                textcoords="offset points",
                bbox=dict(boxstyle="round", fc="w"),
                arrowprops=dict(arrowstyle="->"),
            )
            annotation.set_visible(False)

            def update_note(points: list) -> None:
                """Generate a matplotlib hovering note about the data point index

                Parameters
                ----------
                points : list
                    List of the plotted data points that are hovered over
                """
                # TODO: this only works if the first line object has the pulse we want.
                line, pnum = lines_pnums[0]
                x, y = line.get_data()
                annotation.xy = (x[points[0]], y[points[0]])
                text2 = " ".join([str(pnum[int(n)]) for n in points])
                if len(points) > 1:
                    text = f"Pulses [{text2}]"
                else:
                    text = f"Pulse {text2}"
                annotation.set_text(text)
                annotation.get_bbox_patch().set_alpha(0.75)

            def hover(event: MouseEvent) -> None:
                """Callback to be used when mouse hovers near a plotted point

                Parameters
                ----------
                event : MouseEvent
                    The mouse-related event; contains location information
                """
                vis = annotation.get_visible()
                if event.inaxes != ax:
                    return
                cont, ind = line.contains(event)
                if cont:
                    update_note(ind["ind"])
                    annotation.set_visible(True)
                    fig.canvas.draw_idle()
                elif vis:
                    annotation.set_visible(False)
                    fig.canvas.draw_idle()

            def click(event: MouseEvent) -> None:
                """Callback to be used when mouse clicks near a plotted point

                Parameters
                ----------
                event : MouseEvent
                    The mouse-related event; contains location information
                """
                if event.inaxes != ax:
                    return
                cont, ind = line.contains(event)
                if cont:
                    pnum = lines_pnums[0][1]
                    rownum = pnum[int(ind["ind"][0])]
                    print(f"This is pulse# {rownum}")
                    print(self.df.drop("pulse").row(rownum, named=True))

            fig.canvas.mpl_connect("motion_notify_event", hover)
            fig.canvas.mpl_connect("button_press_event", click)

        plt.xlabel(str(x_col))
        plt.ylabel(str(y_col))
        title_str = f"""{self.header.description}
        use_expr={str(use_expr)}
        good_expr={str(self.good_expr)}"""
        plt.title(title_str)
        if color_col is not None:
            plt.legend(title=color_col)
        plt.tight_layout()

    def good_series(self, col: str, use_expr: pl.Expr = pl.lit(True)) -> pl.Series:
        """Return a Polars Series of the given column, filtered by good_expr and use_expr."""
        return mass2.misc.good_series(self.df, col, self.good_expr, use_expr)

    @property
    def last_avg_pulse(self) -> NDArray | None:
        """Return the average pulse stored in the last recipe step that's an optimal filter step

        Returns
        -------
        NDArray | None
            The last filtering step's signal model, or None if no such step
        """
        for step in reversed(self.steps):
            if isinstance(step, OptimalFilterStep):
                return step.filter_maker.signal_model
        return None

    @property
    def last_filter(self) -> NDArray | None:
        """Return the average pulse stored in the last recipe step that's an optimal filter step

        Returns
        -------
        NDArray | None
            The last filtering step's signal model, or None if no such step
        """
        for step in reversed(self.steps):
            if isinstance(step, OptimalFilterStep):
                return step.filter.values
        return None

    @property
    def last_noise_psd(self) -> tuple[NDArray, NDArray] | None:
        """Return the noise PSD stored in the last recipe step that's an optimal filter step

        Returns
        -------
        tuple[NDArray, NDArray] | None
            The last filtering step's (frequencies, noise spectrum), or None if no such step
        """
        for step in reversed(self.steps):
            if isinstance(step, OptimalFilterStep) and step.spectrum is not None:
                return step.spectrum.frequencies, step.spectrum.psd
        return None

    @property
    def last_noise_autocorrelation(self) -> NDArray | None:
        """Return the noise autocorrelation stored in the last recipe step that's an optimal filter step

        Returns
        -------
        NDArray | None
            The last filtering step's noise autocorrelation, or None if no such step
        """
        for step in reversed(self.steps):
            if isinstance(step, OptimalFilterStep) and step.spectrum is not None:
                return step.spectrum.autocorr_vec
        return None

    def rough_cal_combinatoric(
        self,
        line_names: list[str],
        uncalibrated_col: str,
        calibrated_col: str,
        ph_smoothing_fwhm: float,
        n_extra: int = 3,
        use_expr: pl.Expr = pl.lit(True),
    ) -> "Channel":
        """Learn a rough calibration by trying all combinatorically possible peak assignments."""
        step = mass2.core.RoughCalibrationStep.learn_combinatoric(
            self,
            line_names,
            uncalibrated_col=uncalibrated_col,
            calibrated_col=calibrated_col,
            ph_smoothing_fwhm=ph_smoothing_fwhm,
            n_extra=n_extra,
            use_expr=use_expr,
        )
        return self.with_step(step)

    def rough_cal_combinatoric_height_info(
        self,
        line_names: list[str],
        line_heights_allowed: list[list[int]],
        uncalibrated_col: str,
        calibrated_col: str,
        ph_smoothing_fwhm: float,
        n_extra: int = 3,
        use_expr: pl.Expr = pl.lit(True),
    ) -> "Channel":
        """Learn a rough calibration by trying all combinatorically possible peak assignments,
        using known relative peak heights to limit the possibilities."""
        step = mass2.core.RoughCalibrationStep.learn_combinatoric_height_info(
            self,
            line_names,
            line_heights_allowed,
            uncalibrated_col=uncalibrated_col,
            calibrated_col=calibrated_col,
            ph_smoothing_fwhm=ph_smoothing_fwhm,
            n_extra=n_extra,
            use_expr=use_expr,
        )
        return self.with_step(step)

    def rough_cal(  # noqa: PLR0917
        self,
        line_names: list[str | float],
        uncalibrated_col: str = "filtValue",
        calibrated_col: str | None = None,
        use_expr: pl.Expr = pl.lit(True),
        max_fractional_energy_error_3rd_assignment: float = 0.1,
        min_gain_fraction_at_ph_30k: float = 0.25,
        fwhm_pulse_height_units: float = 75,
        n_extra_peaks: int = 10,
        acceptable_rms_residual_e: float = 10,
    ) -> "Channel":
        """Learn a rough calibration by trying to assign the 3 brightest peaks,
        then fitting a line to those and looking for other peaks that fit that line.
        """
        step = mass2.core.RoughCalibrationStep.learn_3peak(
            self,
            line_names,
            uncalibrated_col,
            calibrated_col,
            use_expr,
            max_fractional_energy_error_3rd_assignment,
            min_gain_fraction_at_ph_30k,
            fwhm_pulse_height_units,
            n_extra_peaks,
            acceptable_rms_residual_e,
        )
        return self.with_step(step)

    def with_step(self, step: RecipeStep) -> "Channel":
        """Return a new Channel with the given step applied to generate new columns in the dataframe."""
        t_start = time.time()
        df2 = step.calc_from_df(self.df)
        elapsed_s = time.time() - t_start
        ch2 = dataclasses.replace(
            self,
            df=df2,
            good_expr=step.good_expr,
            df_history=self.df_history + [self.df],
            steps=self.steps.with_step(step),
            steps_elapsed_s=self.steps_elapsed_s + [elapsed_s],
        )
        return ch2

    def with_steps(self, steps: Recipe) -> "Channel":
        """Return a new Channel with the given steps applied to generate new columns in the dataframe."""
        ch2 = self
        for step in steps:
            ch2 = ch2.with_step(step)
        return ch2

    def with_good_expr(self, good_expr: pl.Expr, replace: bool = False) -> "Channel":
        """Return a new Channel with the given good_expr, combined with the existing good_expr by "and",
        of by replacing it entirely if `replace` is True."""
        # the default value of self.good_expr is pl.lit(True)
        # and_(True) will just add visual noise when looking at good_expr and not affect behavior
        if not replace and good_expr is not True and not good_expr.meta.eq(pl.lit(True)):
            good_expr = good_expr.and_(self.good_expr)
        return dataclasses.replace(self, good_expr=good_expr)

    def with_column_map_step(self, input_col: str, output_col: str, f: Callable) -> "Channel":
        """f should take a numpy array and return a numpy array with the same number of elements"""
        step = mass2.core.recipe.ColumnAsNumpyMapStep([input_col], [output_col], good_expr=self.good_expr, use_expr=pl.lit(True), f=f)
        return self.with_step(step)

    def with_good_expr_pretrig_rms_and_postpeak_deriv(
        self, n_sigma_pretrig_rms: float = 20, n_sigma_postpeak_deriv: float = 20, replace: bool = False
    ) -> "Channel":
        """Set good_expr to exclude pulses with pretrigger RMS or postpeak derivative above outlier-resistant thresholds."""
        max_postpeak_deriv = misc.outlier_resistant_nsigma_above_mid(
            self.df["postpeak_deriv"].to_numpy(), nsigma=n_sigma_postpeak_deriv
        )
        max_pretrig_rms = misc.outlier_resistant_nsigma_above_mid(self.df["pretrig_rms"].to_numpy(), nsigma=n_sigma_pretrig_rms)
        good_expr = (pl.col("postpeak_deriv") < max_postpeak_deriv).and_(pl.col("pretrig_rms") < max_pretrig_rms)
        return self.with_good_expr(good_expr, replace)

    def with_range_around_median(self, col: str, range_up: float, range_down: float) -> "Channel":
        """Set good_expr to exclude pulses with `col` outside the given range around its median."""
        med = np.median(self.df[col].to_numpy())
        return self.with_good_expr(pl.col(col).is_between(med - range_down, med + range_up))

    def with_good_expr_below_nsigma_outlier_resistant(
        self, col_nsigma_pairs: Iterable[tuple[str, float]], replace: bool = False, use_prev_good_expr: bool = True
    ) -> "Channel":
        """Set good_expr to exclude pulses with any of the given columns above outlier-resistant thresholds.
        Always sets lower limit at 0, so don't use for values that can be negative
        """
        if use_prev_good_expr:
            df = self.df.lazy().select(pl.exclude("pulse")).filter(self.good_expr).collect()
        else:
            df = self.df
        for i, (col, nsigma) in enumerate(col_nsigma_pairs):
            max_for_col = misc.outlier_resistant_nsigma_above_mid(df[col].to_numpy(), nsigma=nsigma)
            this_iter_good_expr = pl.col(col).is_between(0, max_for_col)
            if i == 0:
                good_expr = this_iter_good_expr
            else:
                good_expr = good_expr.and_(this_iter_good_expr)
        return self.with_good_expr(good_expr, replace)

    def with_good_expr_nsigma_range_outlier_resistant(
        self, col_nsigma_pairs: Iterable[tuple[str, float]], replace: bool = False, use_prev_good_expr: bool = True
    ) -> "Channel":
        """Set good_expr to exclude pulses with any of the given columns above outlier-resistant thresholds.
        Always sets lower limit at 0, so don't use for values that can be negative
        """
        if use_prev_good_expr:
            df = self.df.lazy().select(pl.exclude("pulse")).filter(self.good_expr).collect()
        else:
            df = self.df
        for i, (col, nsigma) in enumerate(col_nsigma_pairs):
            min_for_col, max_for_col = misc.outlier_resistant_nsigma_range_from_mid(df[col].to_numpy(), nsigma=nsigma)
            this_iter_good_expr = pl.col(col).is_between(min_for_col, max_for_col)
            if i == 0:
                good_expr = this_iter_good_expr
            else:
                good_expr = good_expr.and_(this_iter_good_expr)
        return self.with_good_expr(good_expr, replace)

    @functools.cache
    def typical_peak_ind(self, col: str = "pulse") -> int:
        """Return the typical peak index of the given column, using the median peak index for the first 100 pulses."""
        raw = self.df.limit(100)[col].to_numpy()
        if self.transform_raw is not None:
            raw = self.transform_raw(raw)
        return int(np.median(raw.argmax(axis=1)))

    def summarize_pulses(self, col: str = "pulse", pretrigger_ignore_samples: int = 0, peak_index: int | None = None) -> "Channel":
        """Summarize the pulses, adding columns for pulse height, pretrigger mean, etc."""
        if peak_index is None:
            peak_index = self.typical_peak_ind(col)
        out_names = mass2.core.pulse_algorithms.result_dtype.names
        # mypy (incorrectly) thinks `out_names` might be None, and `list(None)` is forbidden. Assertion makes it happy again.
        assert out_names is not None
        outputs = list(out_names)
        step = SummarizeStep(
            inputs=[col],
            output=outputs,
            good_expr=self.good_expr,
            use_expr=pl.lit(True),
            frametime_s=self.header.frametime_s,
            peak_index=peak_index,
            pulse_col=col,
            pretrigger_ignore_samples=pretrigger_ignore_samples,
            n_presamples=self.header.n_presamples,
            transform_raw=self.transform_raw,
        )
        return self.with_step(step)

    def correct_pretrig_mean_jumps(
        self, uncorrected: str = "pretrig_mean", corrected: str = "ptm_jf", period: int = 4096
    ) -> "Channel":
        """Correct pretrigger mean jumps in the raw pulse data, writing to a new column."""
        step = mass2.core.recipe.PretrigMeanJumpFixStep(
            inputs=[uncorrected],
            output=[corrected],
            good_expr=self.good_expr,
            use_expr=pl.lit(True),
            period=period,
        )
        return self.with_step(step)

    def with_select_step(self, col_expr_dict: dict[str, pl.Expr]) -> "Channel":
        """
        This step is meant for interactive exploration; it's basically like the df.select() method, but it's saved as a step.
        """
        extract = mass2.misc.extract_column_names_from_polars_expr
        inputs: set[str] = set()
        for expr in col_expr_dict.values():
            inputs.update(extract(expr))
        step = mass2.core.recipe.SelectStep(
            inputs=list(inputs),
            output=list(col_expr_dict.keys()),
            good_expr=self.good_expr,
            use_expr=pl.lit(True),
            col_expr_dict=col_expr_dict,
        )
        return self.with_step(step)

    def with_categorize_step(self, category_condition_dict: dict[str, pl.Expr], output_col: str = "category") -> "Channel":
        """Add a recipe step that categorizes pulses based on the given conditions."""
        # ensure the first condition is True, to be used as a fallback
        first_expr = next(iter(category_condition_dict.values()))
        if not first_expr.meta.eq(pl.lit(True)):
            category_condition_dict = {"fallback": pl.lit(True), **category_condition_dict}
        extract = mass2.misc.extract_column_names_from_polars_expr
        inputs: set[str] = set()
        for expr in category_condition_dict.values():
            inputs.update(extract(expr))
        step = mass2.core.recipe.CategorizeStep(
            inputs=list(inputs),
            output=[output_col],
            good_expr=self.good_expr,
            use_expr=pl.lit(True),
            category_condition_dict=category_condition_dict,
        )
        return self.with_step(step)

    def compute_average_pulse(self, pulse_col: str = "pulse", use_expr: pl.Expr = pl.lit(True), limit: int = 2000) -> NDArray:
        """Compute an average pulse given a use expression.

        Parameters
        ----------
        pulse_col : str, optional
            Name of the column in self.df containing raw pulses, by default "pulse"
        use_expr : pl.Expr, optional
            Selection (in addition to self.good_expr) to use, by default pl.lit(True)
        limit : int, optional
            Use no more than this many pulses, by default 2000

        Returns
        -------
        NDArray
            _description_
        """
        avg_pulse = (
            self.df
            .lazy()
            .filter(self.good_expr)
            .filter(use_expr)
            .select(pulse_col)
            .limit(limit)
            .collect()
            .to_series()
            .to_numpy()
            .mean(axis=0)
        )
        avg_pulse -= avg_pulse[: self.header.n_presamples].mean()
        return avg_pulse

    def filter5lag(
        self,
        pulse_col: str = "pulse",
        peak_y_col: str = "5lagy",
        peak_x_col: str = "5lagx",
        f_3db: float = 25e3,
        use_expr: pl.Expr = pl.lit(True),
        time_constant_s_of_exp_to_be_orthogonal_to: float | None = None,
    ) -> "Channel":
        """Compute a 5-lag optimal filter and apply it.

        Parameters
        ----------
        pulse_col : str, optional
            Which column contains raw data, by default "pulse"
        peak_y_col : str, optional
            Column to contain the optimal filter results, by default "5lagy"
        peak_x_col : str, optional
            Column to contain the 5-lag filter's estimate of arrival-time/phase, by default "5lagx"
        f_3db : float, optional
            A low-pass filter 3 dB point to apply to the computed filter, by default 25e3
        use_expr : pl.Expr, optional
            An expression to select pulses for averaging, by default pl.lit(True)
        time_constant_s_of_exp_to_be_orthogonal_to : float | None, optional
            Optionally an exponential decay time to make the filter insensitive to, by default None

        Returns
        -------
        Channel
            This channel with a Filter5LagStep added to the recipe.
        """
        assert self.noise
        noiseresult = self.noise.spectrum(trunc_back=2, trunc_front=2)
        avg_pulse = self.compute_average_pulse(pulse_col=pulse_col, use_expr=use_expr)
        filter_maker = FilterMaker(
            signal_model=avg_pulse,
            n_pretrigger=self.header.n_presamples,
            noise_psd=noiseresult.psd,
            noise_autocorr=noiseresult.autocorr_vec,
            sample_time_sec=self.header.frametime_s,
        )
        if time_constant_s_of_exp_to_be_orthogonal_to is None:
            filter5lag = filter_maker.compute_5lag(f_3db=f_3db)
        else:
            filter5lag = filter_maker.compute_5lag_noexp(f_3db=f_3db, exp_time_seconds=time_constant_s_of_exp_to_be_orthogonal_to)
        step = OptimalFilterStep(
            inputs=["pulse"],
            output=[peak_x_col, peak_y_col],
            good_expr=self.good_expr,
            use_expr=use_expr,
            filter=filter5lag,
            spectrum=noiseresult,
            filter_maker=filter_maker,
            transform_raw=self.transform_raw,
        )
        return self.with_step(step)

    def compute_ats_model(self, pulse_col: str, use_expr: pl.Expr = pl.lit(True), limit: int = 2000) -> tuple[NDArray, NDArray]:
        """Compute the average pulse and arrival-time model for an ATS filter.
        We use the first `limit` pulses that pass `good_expr` and `use_expr`.

        Parameters
        ----------
        pulse_col : str
            _description_
        use_expr : pl.Expr, optional
            _description_, by default pl.lit(True)
        limit : int, optional
            _description_, by default 2000

        Returns
        -------
        tuple[NDArray, NDArray]
            _description_
        """
        df = (
            self.df
            .lazy()
            .filter(self.good_expr)
            .filter(use_expr)
            .limit(limit)
            .select(pulse_col, "pulse_rms", "promptness", "pretrig_mean")
            .collect()
        )

        # Adjust promptness: subtract a linear trend with pulse_rms
        prms = df["pulse_rms"].to_numpy()
        promptness = df["promptness"].to_numpy()
        poly = np.poly1d(np.polyfit(prms, promptness, 1))
        df = df.with_columns(promptshifted=(promptness - poly(prms)))

        # Rescale promptness quadratically to span approximately [-0.5, +0.5], dropping any pulses with abs(t) > 0.45.
        x, y, z = np.percentile(df["promptshifted"], [10, 50, 90])
        A = np.array([[x * x, x, 1], [y * y, y, 1], [z * z, z, 1]])
        param = np.linalg.solve(A, [-0.4, 0, +0.4])
        ATime = np.poly1d(param)(df["promptshifted"])
        df = df.with_columns(ATime=ATime).filter(np.abs(ATime) < 0.45).drop("promptshifted")

        # Compute mean pulse and dt model as the offset and slope of a linear fit to each pulse sample vs ATime
        pulse = df["pulse"].to_numpy()
        avg_pulse = np.zeros(self.header.n_samples, dtype=float)
        dt_model = np.zeros(self.header.n_samples, dtype=float)
        for i in range(self.header.n_presamples, self.header.n_samples):
            slope, offset = np.polyfit(df["ATime"], (pulse[:, i] - df["pretrig_mean"]), 1)
            dt_model[i] = -slope
            avg_pulse[i] = offset
        return avg_pulse, dt_model

    def filterATS(
        self,
        pulse_col: str = "pulse",
        peak_y_col: str = "ats_y",
        peak_x_col: str = "ats_x",
        f_3db: float = 25e3,
        use_expr: pl.Expr = pl.lit(True),
    ) -> "Channel":
        """Compute an arrival-time-safe (ATS) optimal filter and apply it.

        Parameters
        ----------
        pulse_col : str, optional
            Which column contains raw data, by default "pulse"
        peak_y_col : str, optional
            Column to contain the optimal filter results, by default "ats_y"
        peak_x_col : str, optional
            Column to contain the ATS filter's estimate of arrival-time/phase, by default "ats_x"
        f_3db : float, optional
            A low-pass filter 3 dB point to apply to the computed filter, by default 25e3
        use_expr : pl.Expr, optional
            An expression to select pulses for averaging, by default pl.lit(True)

        Returns
        -------
        Channel
            This channel with a Filter5LagStep added to the recipe.
        """
        assert self.noise
        mprms = self.good_series("pulse_rms", use_expr).median()
        use = use_expr.and_(np.abs(pl.col("pulse_rms") / mprms - 1.0) < 0.3)
        limit = 4000
        avg_pulse, dt_model = self.compute_ats_model(pulse_col, use, limit)
        noiseresult = self.noise.spectrum()
        filter_maker = FilterMaker(
            signal_model=avg_pulse,
            dt_model=dt_model,
            n_pretrigger=self.header.n_presamples,
            noise_psd=noiseresult.psd,
            noise_autocorr=noiseresult.autocorr_vec,
            sample_time_sec=self.header.frametime_s,
        )
        filter_ats = filter_maker.compute_ats(f_3db=f_3db)
        step = OptimalFilterStep(
            inputs=["pulse"],
            output=[peak_x_col, peak_y_col],
            good_expr=self.good_expr,
            use_expr=use_expr,
            filter=filter_ats,
            spectrum=noiseresult,
            filter_maker=filter_maker,
            transform_raw=self.transform_raw,
        )
        return self.with_step(step)

    def good_df(self, cols: list[str] | pl.Expr = pl.all(), use_expr: pl.Expr = pl.lit(True)) -> pl.DataFrame:
        """Return a Polars DataFrame of the given columns, filtered by good_expr and use_expr."""
        good_df = self.df.lazy().filter(self.good_expr)
        if use_expr is not True:
            good_df = good_df.filter(use_expr)
        return good_df.select(cols).collect()

    def bad_df(self, cols: list[str] | pl.Expr = pl.all(), use_expr: pl.Expr = pl.lit(True)) -> pl.DataFrame:
        """Return a Polars DataFrame of the given columns, filtered by the inverse of good_expr, and use_expr."""
        bad_df = self.df.lazy().filter(self.good_expr.not_())
        if use_expr is not True:
            bad_df = bad_df.filter(use_expr)
        return bad_df.select(cols).collect()

    def good_serieses(self, cols: list[str], use_expr: pl.Expr = pl.lit(True)) -> list[pl.Series]:
        """Return a list of Polars Series of the given columns, filtered by good_expr and use_expr."""
        df2 = self.good_df(cols, use_expr)
        return [df2[col] for col in cols]

    def driftcorrect(
        self,
        indicator_col: str = "pretrig_mean",
        uncorrected_col: str = "5lagy",
        corrected_col: str | None = None,
        use_expr: pl.Expr = pl.lit(True),
    ) -> "Channel":
        """Correct for gain drift correlated with the given indicator column."""
        # by defining a seperate learn method that takes ch as an argument,
        # we can move all the code for the step outside of Channel
        step = DriftCorrectStep.learn(
            ch=self,
            indicator_col=indicator_col,
            uncorrected_col=uncorrected_col,
            corrected_col=corrected_col,
            use_expr=use_expr,
        )
        return self.with_step(step)

    def linefit(  # noqa: PLR0917
        self,
        line: GenericLineModel | SpectralLine | str | float,
        col: str,
        use_expr: pl.Expr = pl.lit(True),
        has_linear_background: bool = False,
        has_tails: bool = False,
        dlo: float = 50,
        dhi: float = 50,
        binsize: float = 0.5,
        params_update: lmfit.Parameters = lmfit.Parameters(),
    ) -> LineModelResult:
        """Fit a spectral line to the  binned data from the given column, optionally filtering by use_expr."""
        model = mass2.calibration.algorithms.get_model(line, has_linear_background=has_linear_background, has_tails=has_tails)
        pe = model.spect.peak_energy
        _bin_edges = np.arange(pe - dlo, pe + dhi, binsize)
        df_small = self.df.lazy().filter(self.good_expr).filter(use_expr).select(col).collect()
        bin_centers, counts = misc.hist_of_series(df_small[col], _bin_edges)
        params = model.guess(counts, bin_centers=bin_centers, dph_de=1)
        params["dph_de"].set(1.0, vary=False)
        print(f"before update {params=}")
        params = params.update(params_update)
        print(f"after update {params=}")
        result = model.fit(counts, params, bin_centers=bin_centers, minimum_bins_per_fwhm=3)
        result.set_label_hints(
            binsize=bin_centers[1] - bin_centers[0],
            ds_shortname=self.header.description,
            unit_str="eV",
            attr_str=col,
            states_hint=f"{use_expr=}",
            cut_hint="",
        )
        return result

    def step_summary(self) -> list[tuple[str, float]]:
        """Return a list of (step type name, elapsed time in seconds) for each step in the recipe."""
        return [(type(a).__name__, b) for (a, b) in zip(self.steps, self.steps_elapsed_s)]

    def __hash__(self) -> int:
        """Return a hash based on the object's id."""
        # needed to make functools.cache work
        # if self or self.anything is mutated, assumptions will be broken
        # and we may get nonsense results
        return hash(id(self))

    def __eq__(self, other: object) -> bool:
        """Return True if the other object is the same object (by id)."""
        # needed to make functools.cache work
        # if self or self.anything is mutated, assumptions will be broken
        # and we may get nonsense results
        # only checks if the ids match, does not try to be equal if all contents are equal
        return id(self) == id(other)

    @classmethod
    def from_ljh(
        cls,
        path: str | Path,
        noise_path: str | Path | None = None,
        keep_posix_usec: bool = False,
        transform_raw: Callable | None = None,
    ) -> "Channel":
        """Load a Channel from an LJH file, optionally with a NoiseChannel from a corresponding noise LJH file."""
        if not noise_path:
            noise_channel = None
        else:
            noise_channel = NoiseChannel.from_ljh(noise_path)
        ljh = mass2.LJHFile.open(path)
        df, header_df = ljh.to_polars(keep_posix_usec)
        header = ChannelHeader.from_ljh_header_df(header_df)
        channel = cls(
            df, header=header, npulses=ljh.npulses, subframediv=ljh.subframediv, noise=noise_channel, transform_raw=transform_raw
        )
        return channel

    @classmethod
    def from_off(cls, off: OffFile) -> "Channel":
        """Load a Channel from an OFF file."""
        assert off._mmap is not None
        df = pl.from_numpy(np.asarray(off._mmap))
        df = (
            df
            .select(pl.from_epoch("unixnano", time_unit="ns").dt.cast_time_unit("us").alias("timestamp"))
            .with_columns(df)
            .select(pl.exclude("unixnano"))
        )
        df_header = pl.DataFrame(off.header)
        df_header = df_header.with_columns(pl.Series("Filename", [off.filename]))
        header = ChannelHeader(
            f"{os.path.split(off.filename)[1]}",
            off.filename,
            off.header["ChannelNumberMatchingName"],
            off.framePeriodSeconds,
            off._mmap["recordPreSamples"][0],
            off._mmap["recordSamples"][0],
            df_header,
        )
        channel = cls(df, header, off.nRecords, subframediv=off.subframediv)
        return channel

    def with_experiment_state_df(self, df_es: pl.DataFrame, force_timestamp_monotonic: bool = False) -> "Channel":
        """Add experiment states from an existing dataframe"""
        if not self.df["timestamp"].is_sorted():
            df = self.df.select(pl.col("timestamp").cum_max().alias("timestamp")).with_columns(self.df.select(pl.exclude("timestamp")))
            # print("WARNING: in with_experiment_state_df, timestamp is not monotonic, forcing it to be")
            # print("This is likely a BUG in DASTARD.")
        else:
            df = self.df
        df2 = df.join_asof(df_es, on="timestamp", strategy="backward")
        return self.with_replacement_df(df2)

    def with_external_trigger_df(self, df_ext: pl.DataFrame) -> "Channel":
        """Add external trigger times from an existing dataframe"""
        df2 = (
            self.df
            .with_columns(subframecount=pl.col("framecount") * self.subframediv)
            .join_asof(df_ext, on="subframecount", strategy="backward", coalesce=False, suffix="_prev_ext_trig")
            .join_asof(df_ext, on="subframecount", strategy="forward", coalesce=False, suffix="_next_ext_trig")
        )
        return self.with_replacement_df(df2)

    def with_replacement_df(self, df2: pl.DataFrame) -> "Channel":
        """Replace the dataframe with a new one, keeping all other attributes the same."""
        return dataclasses.replace(
            self,
            df=df2,
        )

    def with_columns(self, df2: pl.DataFrame) -> "Channel":
        """Append columns from df2 to the existing dataframe, keeping all other attributes the same."""
        df3 = self.df.with_columns(df2)
        return self.with_replacement_df(df3)

    def multifit_quadratic_gain_cal(
        self,
        multifit: MultiFit,
        previous_cal_step_index: int,
        calibrated_col: str,
        use_expr: pl.Expr = pl.lit(True),
    ) -> "Channel":
        """Fit multiple spectral lines, to create a quadratic gain calibration."""
        step = MultiFitQuadraticGainStep.learn(
            self,
            multifit_spec=multifit,
            previous_cal_step_index=previous_cal_step_index,
            calibrated_col=calibrated_col,
            use_expr=use_expr,
        )
        return self.with_step(step)

    def multifit_mass_cal(
        self,
        multifit: MultiFit,
        previous_cal_step_index: int,
        calibrated_col: str,
        use_expr: pl.Expr = pl.lit(True),
    ) -> "Channel":
        """Fit multiple spectral lines, to create a Mass1-style gain calibration."""
        step = MultiFitMassCalibrationStep.learn(
            self,
            multifit_spec=multifit,
            previous_cal_step_index=previous_cal_step_index,
            calibrated_col=calibrated_col,
            use_expr=use_expr,
        )
        return self.with_step(step)

    def concat_df(self, df: pl.DataFrame) -> "Channel":
        """Concat the given dataframe to the existing dataframe, keeping all other attributes the same.
        If the new frame `df` has a history and/or steps, those will be lost"""
        ch2 = Channel(
            mass2.core.misc.concat_dfs_with_concat_state(self.df, df),
            self.header,
            self.npulses,
            subframediv=self.subframediv,
            noise=self.noise,
            good_expr=self.good_expr,
        )
        # we won't copy over df_history and steps. I don't think you should use this when those are filled in?
        return ch2

    def concat_ch(self, ch: "Channel") -> "Channel":
        """Concat the given channel's dataframe to the existing dataframe, keeping all other attributes the same.
        If the new channel `ch` has a history and/or steps, those will be lost"""
        ch2 = self.concat_df(ch.df)
        return ch2

    def phase_correct_mass_specific_lines(
        self,
        indicator_col: str,
        uncorrected_col: str,
        line_names: Iterable[str | float],
        previous_cal_step_index: int,
        corrected_col: str | None = None,
        use_expr: pl.Expr = pl.lit(True),
    ) -> "Channel":
        """Apply phase correction to the given uncorrected column, where specific lines are used to judge the correction."""
        if corrected_col is None:
            corrected_col = uncorrected_col + "_pc"
        step = mass2.core.phase_correct_steps.phase_correct_mass_specific_lines(
            self,
            indicator_col,
            uncorrected_col,
            corrected_col,
            previous_cal_step_index,
            line_names,
            use_expr,
        )
        return self.with_step(step)

    def as_bad(self, error_type: type | None, error_msg: str, backtrace: str | None) -> "BadChannel":
        """Return a BadChannel object, which wraps this Channel and includes error information."""
        return BadChannel(self, error_type, error_msg, backtrace)

    def save_recipes(self, filename: str) -> dict[int, Recipe]:
        """Save the recipe steps to a pickle file, keyed by channel number."""
        steps = {self.header.ch_num: self.steps}
        misc.pickle_object(steps, filename)
        return steps

    def plot_summaries(self, use_expr_in: pl.Expr | None = None, downsample: int | None = None, log: bool = False) -> None:
        """Plot a summary of the data set, including time series and histograms of key pulse properties.

        Parameters
        ----------
        use_expr_in: pl.Expr | None, optional
            A polars expression to determine valid pulses, by default None. If None, use `self.good_expr`
        downsample: int | None, optional
            Plot only every one of `downsample` pulses in the scatter plots, by default None.
            If None, choose the smallest value so that no more than 10000 points appear
        log: bool, optional
            Whether to make the histograms have a logarithmic y-scale, by default False.
        """
        plt.figure()
        tpi_microsec = (self.typical_peak_ind() - self.header.n_presamples) * (1e6 * self.header.frametime_s)
        plottables = (
            ("pulse_rms", "Pulse RMS", "#dd00ff", None),
            ("pulse_average", "Pulse Avg", "purple", None),
            ("peak_value", "Peak value", "blue", None),
            ("pretrig_rms", "Pretrig RMS", "green", [0, 4000]),
            ("pretrig_mean", "Pretrig Mean", "#00ff26", None),
            ("postpeak_deriv", "Max PostPk deriv", "gold", [0, 200]),
            ("rise_time_µs", "Rise time (µs)", "orange", [-0.3 * tpi_microsec, 2 * tpi_microsec]),
            ("peak_time_µs", "Peak time (µs)", "red", [-0.3 * tpi_microsec, 2 * tpi_microsec]),
        )

        use_expr = self.good_expr if use_expr_in is None else use_expr_in

        if downsample is None:
            downsample = self.npulses // 10000
        downsample = max(downsample, 1)

        df = self.df.lazy().gather_every(downsample)
        df = df.with_columns(
            ((pl.col("peak_index") - self.header.n_presamples) * (1e6 * self.header.frametime_s)).alias("peak_time_µs")
        )
        df = df.with_columns((pl.col("rise_time") * 1e6).alias("rise_time_µs"))
        existing_columns = df.collect_schema().names()
        preserve = [p[0] for p in plottables if p[0] in existing_columns]
        preserve.append("timestamp")
        df2 = df.filter(use_expr).select(preserve).collect()

        # Plot timeseries relative to 0 = the last 00 UT during or before the run.
        timestamp = df2["timestamp"].to_numpy()
        last_midnight = timestamp[-1].astype("datetime64[D]")
        hour_rel = (timestamp - last_midnight).astype(float) / 3600e6

        for i, (column_name, label, color, limits) in enumerate(plottables):
            if column_name not in df2:
                continue
            y = df2[column_name].to_numpy()

            # Time series scatter plots (left-hand panels)
            plt.subplot(len(plottables), 2, 1 + i * 2)
            plt.ylabel(label)
            plt.plot(hour_rel, y, ".", ms=1, color=color)
            if i == len(plottables) - 1:
                plt.xlabel("Time since last UT midnight (hours)")

            # Histogram (right-hand panels)
            plt.subplot(len(plottables), 2, 2 + i * 2)
            contents, _, _ = plt.hist(y, 200, range=limits, log=log, histtype="stepfilled", fc=color, alpha=0.5)
            if log:
                plt.ylim(ymin=contents.min())
        print(f"Plotting {len(y)} out of {self.npulses} data points")

    def fit_pulse(self, index: int = 0, col: str = "pulse", verbose: bool = True) -> LineModelResult:
        """Fit a single pulse to a 2-exponential-with-tail model, returning the fit result."""
        pulse = self.df[col][index].to_numpy()
        result = mass2.core.pulse_algorithms.fit_pulse_2exp_with_tail(pulse, npre=self.header.n_presamples, dt=self.header.frametime_s)
        if verbose:
            print(f"ch={self}")
            print(f"pulse index={index}")
            print(result.fit_report())
        return result


@dataclass(frozen=True)
class BadChannel:
    """A wrapper around Channel that includes error information."""

    ch: Channel
    error_type: type | None
    error_msg: str
    backtrace: str | None
