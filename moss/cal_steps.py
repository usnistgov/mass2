import os
from dataclasses import dataclass, field
import polars as pl
import pylab as plt
import functools
import typing
import numpy as np
import moss


@dataclass(frozen=True)
class CalStep:
    inputs: list[str]
    output: list[str]
    good_expr: pl.Expr
    use_expr: pl.Expr


@dataclass(frozen=True)
class DriftCorrectStep(CalStep):
    dc: typing.Any

    def calc_from_df(self, df):
        indicator, uncorrected = self.inputs
        slope, offset = self.dc.slope, self.dc.offset
        df2 = df.select(
            (pl.col(uncorrected) * (1 + slope * (pl.col(indicator) - offset))).alias(self.output[0])
        ).with_columns(df)
        return df2

    def dbg_plot(self, df):
        indicator, uncorrected = self.inputs
        # breakpoint()
        df_small = (
            df.lazy()
            .filter(self.good_expr)
            .filter(self.use_expr)
            .select(self.inputs + self.output)
            .collect()
        )
        moss.misc.plot_a_vs_b_series(df_small[indicator], df_small[uncorrected])
        moss.misc.plot_a_vs_b_series(
            df_small[indicator],
            df_small[self.output[0]],
            plt.gca(),
        )
        plt.legend()
        plt.tight_layout()
        return plt.gca()


@dataclass(frozen=True)
class RoughCalibrationGainStep(CalStep):
    line_names: list[str]
    line_energies: np.ndarray
    predicted_energies: np.ndarray
    gain_pfit: np.polynomial.Polynomial

    def dbg_plot(self, df, bin_edges=np.arange(0, 10000, 1), axis=None, plotkwarg={}):
        series = moss.good_series(df, col=self.output[0], good_expr=self.good_expr, use_expr=True)
        axis = moss.misc.plot_hist_of_series(series, bin_edges)
        axis.plot(self.line_energies, np.zeros(len(self.line_energies)), "o")
        for line_name, energy in zip(self.line_names, self.line_energies):
            axis.annotate(line_name, (energy, 0), rotation=90)
        np.set_printoptions(precision=2)
        energy_residuals = self.predicted_energies-self.line_energies
        axis.set_title(f"RoughCalibrationStep dbg_plot\n{energy_residuals=}")
        return axis
    
@dataclass(frozen=True)
class RoughCalibrationStep(CalStep):
    pfresult: moss.rough_cal.SmoothedLocalMaximaResult
    assignment_result: moss.rough_cal.BestAssignmentPfitGainResult
    ph2energy: callable

    def calc_from_df(self, df):
        # only works with in memory data, but just takes it as numpy data and calls function
        # is much faster than map_elements approach, but wouldn't work with out of core data without some extra book keeping
        inputs_np = [df[input].to_numpy() for input in self.inputs]
        out = self.ph2energy(inputs_np[0])
        df2 = pl.DataFrame({self.output[0]: out}).with_columns(df)
        return df2

    def dbg_plot_old(self, df, bin_edges=np.arange(0, 10000, 1), axis=None, plotkwarg={}):
        series = moss.good_series(df, col=self.output[0], good_expr=self.good_expr, use_expr=self.use_expr)
        axis = moss.misc.plot_hist_of_series(series, bin_edges)
        axis.plot(self.line_energies, np.zeros(len(self.line_energies)), "o")
        for line_name, energy in zip(self.line_names, self.line_energies):
            axis.annotate(line_name, (energy, 0), rotation=90)
        np.set_printoptions(precision=2)
        energy_residuals = self.predicted_energies-self.line_energies
        axis.set_title(f"RoughCalibrationStep dbg_plot\n{energy_residuals=}")
        return axis
    
    def dbg_plot(self, df, axs=None):
        if axs is None:
            fig, axs = plt.subplots(2, 1, figsize=(8,6))
        self.assignment_result.plot(ax=axs[0])
        self.pfresult.plot(self.assignment_result, ax=axs[1])
    
    def energy2ph(self, energy):
        return self.assignment_result.energy2ph(energy)


@dataclass(frozen=True)
class SummarizeStep(CalStep):
    frametime_s: float
    peak_index: int
    pulse_col: str
    pretrigger_ignore: int
    n_presamples: int

    def calc_from_df(self, df):
        df2 = pl.concat(
            pl.from_numpy(
                moss.pulse_algorithms.summarize_data_numba(
                    df_iter[self.pulse_col].to_numpy(),
                    self.frametime_s,
                    peak_samplenumber=self.peak_index,
                    pretrigger_ignore=self.pretrigger_ignore,
                    nPresamples=self.n_presamples,
                )
            )
            for df_iter in df.iter_slices()
        ).with_columns(df)
        return df2
    
    def dbg_plot(self, df_after, **kwargs):
        pass


@dataclass(frozen=True)
class Filter5LagStep(CalStep):
    filter: moss.Filter
    spectrum: moss.NoisePSD

    def calc_from_df(self, df):
        dfs = []
        for df_iter in df.iter_slices(10000):
            peak_x, peak_y = moss.filters.filter_data_5lag(
                self.filter.filter, df_iter[self.inputs[0]].to_numpy()
            )
            dfs.append(pl.DataFrame({"peak_x": peak_x, "peak_y": peak_y}))
        df2 = pl.concat(dfs).with_columns(df)
        df2 = df2.rename({"peak_x": self.output[0], "peak_y": self.output[1]})
        return df2

    def dbg_plot(self, df):
        return self.filter.plot()


@dataclass(frozen=True)
class CalSteps:
    # leaves many optimizations on the table, but is very simple
    # 1. we could calculate filt_value_5lag and filt_phase_5lag at the same time
    # 2. we could calculate intermediate quantities optionally and not materialize all of them
    steps: list[CalStep]

    def calc_from_df(self, df):
        "return a dataframe with all the newly calculated info"
        for step in self.steps:
            df = step.calc_from_df(df).with_columns(df)
        return df

    def calc_from_df_np(self, df):
        "return a dataframe with all the newly calculated info"
        for step in self.steps:
            df = step.calc_from_df_np(df).with_columns(df)
        return df

    @classmethod
    def new_empty(cls):
        return cls([])

    def __getitem__(self, key):
        return self.steps[key]
    
    def __len__(self):
        return len(self.steps)

    # def copy(self):
    #     # copy by creating a new list containing all the entires in the old list
    #     # a list entry, aka a CalStep, should be immutable
    #     return CalSteps(self.steps[:])

    def with_step(self, step: CalStep):
        # return a new CalSteps with the step added, no mutation!
        return CalSteps(self.steps + [step])
    
@dataclass(frozen=True)
class MultiFitSplineStep(CalStep):
    ph2energy: callable
    multifit: moss.MultiFit

    def calc_from_df(self, df):
        # only works with in memory data, but just takes it as numpy data and calls function
        # is much faster than map_elements approach, but wouldn't work with out of core data without some extra book keeping
        inputs_np = [df[input].to_numpy() for input in self.inputs]
        out = self.ph2energy(inputs_np[0])
        df2 = pl.DataFrame({self.output[0]: out}).with_columns(df)
        return df2

    def dbg_plot(self, df):
        self.multifit.plot_results()

    def energy2ph(self, e):
        return self.ph2energy.solve(e)[0]