from dataclasses import dataclass
import polars as pl
import moss
import numpy as np
import pylab as plt



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
class PretrigMeanJumpFixStep(CalStep):
    period: float

    def calc_from_df(self, df: pl.DataFrame):
        ptm1 = df[self.inputs[0]].to_numpy()
        ptm2 = np.unwrap(ptm1 % self.period, period=self.period)
        df2 = pl.DataFrame({self.output[0]:ptm2}).with_columns(df)
        return df2
    
    def dbg_plot(self, df_after, **kwargs):
        plt.figure()
        plt.plot(df_after["timestamp"], df_after[self.inputs[0]], ".", label=self.inputs[0])
        plt.plot(df_after["timestamp"], df_after[self.output[0]], ".", label=self.output[0])
        plt.legend()
        plt.xlabel("timestamp")
        plt.ylabel("pretrig mean")
        plt.tight_layout()
        return plt.gca()





@dataclass(frozen=True)
class SummarizeStep(CalStep):
    frametime_s: float
    peak_index: int
    pulse_col: str
    pretrigger_ignore_samples: int
    n_presamples: int

    def calc_from_df(self, df):
        df2 = pl.concat(
            pl.from_numpy(
                moss.pulse_algorithms.summarize_data_numba(
                    df_iter[self.pulse_col].to_numpy(),
                    self.frametime_s,
                    peak_samplenumber=self.peak_index,
                    pretrigger_ignore_samples=self.pretrigger_ignore_samples,
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
