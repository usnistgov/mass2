import os
import lmfit
from dataclasses import dataclass, field
import polars as pl
import pylab as plt
import marimo as mo
import functools
from collections.abc import Callable
import numpy as np
import time

from .noise_channel import NoiseChannel
from .cal_steps import CalSteps, SummarizeStep
from .drift_correction import DriftCorrectStep
from .optimal_filtering import FilterMaker
from .filter_steps import Filter5LagStep
from .multifit import MultiFit, MultiFitQuadraticGainCalStep, MultiFitMassCalibrationStep
from . import misc
import mass2


@dataclass(frozen=True)
class ChannelHeader:
    description: str  # filename or date/run number, etc
    ch_num: int
    frametime_s: float
    n_presamples: int
    n_samples: int
    df: pl.DataFrame = field(repr=False)

    @classmethod
    def from_ljh_header_df(cls, df):
        return cls(
            description=os.path.split(df["Filename"][0])[-1],
            ch_num=df["Channel"][0],
            frametime_s=df["Timebase"][0],
            n_presamples=df["Presamples"][0],
            n_samples=df["Total Samples"][0],
            df=df,
        )


@dataclass(frozen=True)  # noqa: PLR0904
class Channel:
    df: pl.DataFrame = field(repr=False)
    header: ChannelHeader = field(repr=True)
    npulses: int
    noise: NoiseChannel | None = field(default=None, repr=False)
    good_expr: bool | pl.Expr = True
    df_history: list[pl.DataFrame] = field(default_factory=list, repr=False)
    steps: CalSteps = field(default_factory=CalSteps.new_empty)
    steps_elapsed_s: list[float] = field(default_factory=list)
    transform_raw: Callable | None = None

    def mo_stepplots(self):
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

        def show():
            return self._mo_stepplots_explicit(mo_ui)

        def step_ind():
            return mo_ui.value

        mo_ui.show = show
        mo_ui.step_ind = step_ind
        return mo_ui

    def _mo_stepplots_explicit(self, mo_ui):
        step_ind = mo_ui.step_ind()
        self.step_plot(step_ind)
        fig = plt.gcf()
        return mo.vstack([mo_ui, misc.show(fig)])

    def get_step(self, index):
        if index < 0:
            # normalize the index to a positive index
            index = len(self.steps) + index
        step = self.steps[index]
        return step, index

    def step_plot(self, step_ind, **kwargs):
        step, step_ind = self.get_step(step_ind)
        if step_ind + 1 == len(self.df_history):
            df_after = self.df
        else:
            df_after = self.df_history[step_ind + 1]
        return step.dbg_plot(df_after, **kwargs)

    def plot_hist(self, col, bin_edges, axis=None):
        axis = misc.plot_hist_of_series(self.good_series(col), bin_edges, axis)
        axis.set_title(f"ch {self.header.ch_num} plot_hist")
        return axis

    def plot_hists(
        self,
        col,
        bin_edges,
        group_by_col,
        axis=None,
        use_good_expr=True,
        use_expr=True,
        skip_none=True,
    ):
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
            # bin_centers, counts = misc.hist_of_series(values, bin_edges)
            # plt.plot(bin_centers, counts, label=group_name)

        # Customize the plot
        ax.set_xlabel(str(col))
        ax.set_ylabel("Frequency")
        ax.set_title(f"Histogram of {col} grouped by {group_by_col}")

        # Add a legend to label the groups
        ax.legend(title=group_by_col)

        plt.tight_layout()

    def plot_scatter(
        self,
        x_col,
        y_col,
        color_col=None,
        use_expr=True,
        use_good_expr=True,
        skip_none=True,
        ax=None,
    ):
        if ax is None:
            plt.figure()
            ax = plt.gca()
        plt.sca(ax)  # set current axis so I can use plt api
        if use_good_expr and self.good_expr is not True:
            # True doesn't implement .and_, haven't found a exper literal equivalent that does
            # so we special case True
            filter_expr = self.good_expr.and_(use_expr)
        else:
            filter_expr = use_expr
        df_small = self.df.lazy().filter(filter_expr).select(x_col, y_col, color_col).collect()
        for (name,), data in df_small.group_by(color_col, maintain_order=True):
            if name is None and skip_none and color_col is not None:
                continue
            plt.plot(
                data.select(x_col).to_series(),
                data.select(y_col).to_series(),
                ".",
                label=name,
            )
        plt.xlabel(str(x_col))
        plt.ylabel(str(y_col))
        title_str = f"""{self.header.description}
        use_expr={str(use_expr)}
        good_expr={str(self.good_expr)}"""
        plt.title(title_str)
        if color_col is not None:
            plt.legend(title=color_col)
        plt.tight_layout()

    def good_series(self, col, use_expr=True):
        return mass2.misc.good_series(self.df, col, self.good_expr, use_expr)

    def rough_cal_combinatoric(
        self,
        line_names,
        uncalibrated_col,
        calibrated_col,
        ph_smoothing_fwhm,
        n_extra=3,
        use_expr=True,
    ) -> "Channel":
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
        line_names,
        line_heights_allowed,
        uncalibrated_col,
        calibrated_col,
        ph_smoothing_fwhm,
        n_extra=3,
        use_expr=True,
    ) -> "Channel":
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
        use_expr: bool | pl.Expr = True,
        max_fractional_energy_error_3rd_assignment: float = 0.1,
        min_gain_fraction_at_ph_30k: float = 0.25,
        fwhm_pulse_height_units: float = 75,
        n_extra_peaks: int = 10,
        acceptable_rms_residual_e: float = 10,
    ) -> "Channel":
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

    def with_step(self, step) -> "Channel":
        t_start = time.time()
        df2 = step.calc_from_df(self.df)
        elapsed_s = time.time() - t_start
        ch2 = Channel(
            df=df2,
            header=self.header,
            npulses=self.npulses,
            noise=self.noise,
            good_expr=step.good_expr,
            df_history=self.df_history + [self.df],
            steps=self.steps.with_step(step),
            steps_elapsed_s=self.steps_elapsed_s + [elapsed_s],
        )
        return ch2

    def with_steps(self, steps) -> "Channel":
        ch2 = self
        for step in steps:
            ch2 = ch2.with_step(step)
        return ch2

    def with_good_expr(self, good_expr, replace=False) -> "Channel":
        # the default value of self.good_expr is True
        # and_(True) will just add visual noise when looking at good_expr and not affect behavior
        if not replace and good_expr is not True:
            good_expr = good_expr.and_(self.good_expr)
        return Channel(
            df=self.df,
            header=self.header,
            npulses=self.npulses,
            noise=self.noise,
            good_expr=good_expr,
            df_history=self.df_history,
            steps=self.steps,
            steps_elapsed_s=self.steps_elapsed_s,
        )

    def with_column_map_step(self, input_col: str, output_col: str, f: Callable) -> "Channel":
        """f should take a numpy array and return a numpy array with the same number of elements"""
        step = mass2.core.cal_steps.ColumnAsNumpyMapStep([input_col], [output_col], good_expr=self.good_expr, use_expr=True, f=f)
        return self.with_step(step)

    def with_good_expr_pretrig_rms_and_postpeak_deriv(
        self, n_sigma_pretrig_rms=20, n_sigma_postpeak_deriv=20, replace=False
    ) -> "Channel":
        max_postpeak_deriv = misc.outlier_resistant_nsigma_above_mid(
            self.df["postpeak_deriv"].to_numpy(), nsigma=n_sigma_postpeak_deriv
        )
        max_pretrig_rms = misc.outlier_resistant_nsigma_above_mid(self.df["pretrig_rms"].to_numpy(), nsigma=n_sigma_pretrig_rms)
        good_expr = (pl.col("postpeak_deriv") < max_postpeak_deriv).and_(pl.col("pretrig_rms") < max_pretrig_rms)
        return self.with_good_expr(good_expr, replace)

    def with_range_around_median(self, col, range_up, range_down):
        med = np.median(self.df[col].to_numpy())
        return self.with_good_expr(pl.col(col).is_between(med - range_down, med + range_up))

    def with_good_expr_below_nsigma_outlier_resistant(self, col_nsigma_pairs, replace=False, use_prev_good_expr=True) -> "Channel":
        """
        always sets lower limit at 0, don't use for values that can be negative
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

    def with_good_expr_nsigma_range_outlier_resistant(self, col_nsigma_pairs, replace=False, use_prev_good_expr=True) -> "Channel":
        """
        always sets lower limit at 0, don't use for values that can be negative
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
    def typical_peak_ind(self, col="pulse"):
        raw = self.df.limit(100)[col].to_numpy()
        if self.transform_raw is not None:
            raw = self.transform_raw(raw)
        return int(np.median(raw.argmax(axis=1)))

    def summarize_pulses(self, col="pulse", pretrigger_ignore_samples=0, peak_index=None) -> "Channel":
        if peak_index is None:
            peak_index = self.typical_peak_ind(col)
        step = SummarizeStep(
            inputs=[col],
            output=["many"],
            good_expr=self.good_expr,
            use_expr=True,
            frametime_s=self.header.frametime_s,
            peak_index=peak_index,
            pulse_col=col,
            pretrigger_ignore_samples=pretrigger_ignore_samples,
            n_presamples=self.header.n_presamples,
            transform_raw=self.transform_raw,
        )
        return self.with_step(step)

    def correct_pretrig_mean_jumps(self, uncorrected="pretrig_mean", corrected="ptm_jf", period=4096):
        step = mass2.core.cal_steps.PretrigMeanJumpFixStep(
            inputs=[uncorrected],
            output=[corrected],
            good_expr=self.good_expr,
            use_expr=True,
            period=period,
        )
        return self.with_step(step)

    def with_select_step(self, col_expr_dict: dict[str, pl.Expr]) -> "Channel":
        """
        This step is meant for interactive exploration, it's basically like the df.select() method, but it's saved as a step.
        """
        extract = mass2.misc.extract_column_names_from_polars_expr
        inputs = [extract(expr) for expr in col_expr_dict.values()]  # list of lists
        inputs = list(set([col for expr in inputs for col in expr]))  # flatten the list of lists and get unique values
        step = mass2.core.cal_steps.SelectStep(
            inputs=inputs,
            output=list(col_expr_dict.keys()),
            good_expr=self.good_expr,
            use_expr=True,
            col_expr_dict=col_expr_dict,
        )
        return self.with_step(step)

    def with_categorize_step(self, category_condition_dict: dict[str, pl.Expr], output_col="category") -> "Channel":
        # ensure the first condition is True, to be used as a fallback
        if next(iter(category_condition_dict.values())) is not True:
            category_condition_dict = {"fallback": True, **category_condition_dict}
        extract = mass2.misc.extract_column_names_from_polars_expr
        inputs = [extract(expr) for expr in category_condition_dict.values()]
        inputs = list(set([col for expr in inputs for col in expr]))
        step = mass2.core.cal_steps.CategorizeStep(
            inputs=inputs,
            output=[output_col],
            good_expr=self.good_expr,
            use_expr=True,
            category_condition_dict=category_condition_dict,
        )
        return self.with_step(step)

    def filter5lag(
        self,
        pulse_col="pulse",
        peak_y_col="5lagy",
        peak_x_col="5lagx",
        f_3db=25e3,
        use_expr=True,
    ) -> "Channel":
        avg_pulse = (
            self.df.lazy()
            .filter(self.good_expr)
            .filter(use_expr)
            .select(pulse_col)
            .limit(2000)
            .collect()
            .to_series()
            .to_numpy()
            .mean(axis=0)
        )
        assert self.noise
        spectrum5lag = self.noise.spectrum(trunc_front=2, trunc_back=2)
        filter5lag = FilterMaker.create_5lag(
            avg_signal=avg_pulse,
            n_pretrigger=self.header.n_presamples,
            noise_psd=spectrum5lag.psd,
            noise_autocorr_vec=spectrum5lag.autocorr_vec,
            dt=self.header.frametime_s,
            f_3db=f_3db,
        )
        step = Filter5LagStep(
            inputs=["pulse"],
            output=[peak_x_col, peak_y_col],
            good_expr=self.good_expr,
            use_expr=use_expr,
            filter=filter5lag,
            spectrum=spectrum5lag,
            transform_raw=self.transform_raw,
        )
        return self.with_step(step)

    def good_df(self, cols=pl.all(), use_expr=True):
        return self.df.lazy().filter(self.good_expr).filter(use_expr).select(cols).collect()

    def bad_df(self, cols=pl.all(), use_expr=True):
        return self.df.lazy().filter(self.good_expr.not_()).filter(use_expr).select(cols).collect()

    def good_serieses(self, cols, use_expr):
        df2 = self.good_df(cols, use_expr)
        return [df2[col] for col in cols]

    def driftcorrect(
        self,
        indicator_col="pretrig_mean",
        uncorrected_col="5lagy",
        corrected_col=None,
        use_expr=True,
    ) -> "Channel":
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
        line,
        col,
        use_expr=True,
        has_linear_background=False,
        has_tails=False,
        dlo=50,
        dhi=50,
        binsize=0.5,
        params_update=lmfit.Parameters(),
    ):
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

    def step_summary(self):
        return [(type(a).__name__, b) for (a, b) in zip(self.steps, self.steps_elapsed_s)]

    def __hash__(self) -> int:
        # needed to make functools.cache work
        # if self or self.anything is mutated, assumptions will be broken
        # and we may get nonsense results
        return hash(id(self))

    def __eq__(self, other):
        # needed to make functools.cache work
        # if self or self.anything is mutated, assumptions will be broken
        # and we may get nonsense results
        # only checks if the ids match, does not try to be equal if all contents are equal
        return id(self) == id(other)

    @classmethod
    def from_ljh(cls, path, noise_path=None, keep_posix_usec=False, transform_raw: Callable | None = None) -> "Channel":
        if not noise_path:
            noise_channel = None
        else:
            noise_channel = NoiseChannel.from_ljh(noise_path)
        ljh = mass2.LJHFile.open(path)
        df, header_df = ljh.to_polars(keep_posix_usec)
        header = ChannelHeader.from_ljh_header_df(header_df)
        channel = Channel(df, header=header, npulses=ljh.npulses, noise=noise_channel, transform_raw=transform_raw)
        return channel

    @classmethod
    def from_off(cls, off) -> "Channel":
        df = pl.from_numpy(off._mmap)
        df = (
            df.select(pl.from_epoch("unixnano", time_unit="ns").dt.cast_time_unit("us").alias("timestamp"))
            .with_columns(df)
            .select(pl.exclude("unixnano"))
        )
        df_header = pl.DataFrame(off.header)
        df_header = df_header.with_columns(pl.Series("Filename", [off.filename]))
        header = ChannelHeader(
            f"{os.path.split(off.filename)[1]}",
            off.header["ChannelNumberMatchingName"],
            off.framePeriodSeconds,
            off._mmap["recordPreSamples"][0],
            off._mmap["recordSamples"][0],
            df_header,
        )
        channel = cls(df, header, off.nPulses)
        return channel

    def with_experiment_state_df(self, df_es, force_timestamp_monotonic=False) -> "Channel":
        if not self.df["timestamp"].is_sorted():
            df = self.df.select(pl.col("timestamp").cum_max().alias("timestamp")).with_columns(self.df.select(pl.exclude("timestamp")))
            # print("WARNING: in with_experiment_state_df, timestamp is not monotonic, forcing it to be")
            # print("This is likely a BUG in DASTARD.")
        else:
            df = self.df
        df2 = df.join_asof(df_es, on="timestamp", strategy="backward")
        return self.with_replacement_df(df2)

    def with_external_trigger_df(self, df_ext: pl.DataFrame):
        # df2 = self.df.join_asof(df_ext, )
        raise NotImplementedError("not implemented")

    def with_replacement_df(self, df2) -> "Channel":
        return Channel(
            df=df2,
            header=self.header,
            npulses=self.npulses,
            noise=self.noise,
            good_expr=self.good_expr,
            df_history=self.df_history,
            steps=self.steps,
            transform_raw=self.transform_raw,
        )

    def with_columns(self, df2) -> "Channel":
        df3 = self.df.with_columns(df2)
        return self.with_replacement_df(df3)

    def multifit_quadratic_gain_cal(
        self,
        multifit: MultiFit,
        previous_cal_step_index,
        calibrated_col,
        use_expr=True,
    ) -> "Channel":
        step = MultiFitQuadraticGainCalStep.learn(
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
        previous_cal_step_index,
        calibrated_col,
        use_expr=True,
    ) -> "Channel":
        step = MultiFitMassCalibrationStep.learn(
            self,
            multifit_spec=multifit,
            previous_cal_step_index=previous_cal_step_index,
            calibrated_col=calibrated_col,
            use_expr=use_expr,
        )
        return self.with_step(step)

    def concat_df(self, df: pl.DataFrame) -> "Channel":
        ch2 = Channel(mass2.core.misc.concat_dfs_with_concat_state(self.df, df), self.header, self.npulses, self.noise, self.good_expr)
        # we won't copy over df_history and steps. I don't think you should use this when those are filled in?
        return ch2

    def concat_ch(self, ch: "Channel") -> "Channel":
        ch2 = self.concat_df(ch.df)
        return ch2

    def phase_correct_mass_specific_lines(
        self,
        indicator_col,
        uncorrected_col,
        line_names,
        previous_cal_step_index,
        corrected_col=None,
        use_expr=True,
    ) -> "Channel":
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

    def as_bad(self, error_type, error_msg, backtrace):
        return BadChannel(self, error_type, error_msg, backtrace)

    def save_steps(self, filename):
        steps = {self.header.ch_num: self.steps[:]}
        misc.pickle_object(steps, filename)
        return steps

    def plot_summaries(self, use_expr_in: pl.Expr | None = None, downsample: int | None = None, log: bool = False) -> None:
        """Plot a summary of the data set, including time series and histograms of key pulse properties.

        Parameters
        ----------
        use_expr_in : pl.Expr | None, optional
            A polars expression to determine valid pulses, by default None. If None, use `self.good_expr`
        downsample : int | None, optional
            Plot only every one of `downsample` pulses in the scatter plots, by default None.
            If None, choose the smallest value so that no more than 10000 points appear
        log : bool, optional
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
            if limits is None:
                in_limit = np.ones(len(y), dtype=bool)
            else:
                in_limit = np.logical_and(y[:] > limits[0], y[:] < limits[1])
            contents, _, _ = plt.hist(y[in_limit], 200, log=log, histtype="stepfilled", fc=color, alpha=0.5)
            if log:
                plt.ylim(ymin=contents.min())
        print(f"Plotting {len(y)} out of {self.npulses} data points")


@dataclass(frozen=True)
class BadChannel:
    ch: Channel
    error_type: type
    error_msg: str
    backtrace: str
