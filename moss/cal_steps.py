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