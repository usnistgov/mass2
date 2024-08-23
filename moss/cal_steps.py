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

    @property
    def description(self):
        return f"{type(self).__name__} inputs={self.inputs} outputs={self.output}"


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
    
