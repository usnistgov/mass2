from dataclasses import dataclass
from collections.abc import Iterable
from collections.abc import Callable
import polars as pl
import numpy as np
import pylab as plt
from . import pulse_algorithms


@dataclass(frozen=True)
class CalStep:
    inputs: list[str]
    output: list[str]
    good_expr: pl.Expr
    use_expr: pl.Expr

    @property
    def description(self):
        return f"{type(self).__name__} inputs={self.inputs} outputs={self.output}"

    def calc_from_df(self, df: pl.DataFrame) -> pl.DataFrame:
        # TODO: should this be an abstract method?
        return df.filter(self.good_expr)

    def dbg_plot(self, df_after: pl.DataFrame, **kwargs) -> plt.Axes:
        # this is a no-op, subclasses can override this to plot something
        plt.figure()
        plt.text(0.0, 0.5, f"No plot defined for: {self.description}")
        return plt.gca()


@dataclass(frozen=True)
class PretrigMeanJumpFixStep(CalStep):
    period: float

    def calc_from_df(self, df: pl.DataFrame) -> pl.DataFrame:
        ptm1 = df[self.inputs[0]].to_numpy()
        ptm2 = np.unwrap(ptm1 % self.period, period=self.period)
        df2 = pl.DataFrame({self.output[0]: ptm2}).with_columns(df)
        return df2

    def dbg_plot(self, df_after: pl.DataFrame, **kwargs) -> plt.Axes:
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
    transform_raw: Callable | None = None

    def calc_from_df(self, df: pl.DataFrame) -> pl.DataFrame:
        summaries = []
        for df_iter in df.select(self.inputs).iter_slices():
            raw = df_iter[self.pulse_col].to_numpy()
            if self.transform_raw is not None:
                raw = self.transform_raw(raw)

            s = pl.from_numpy(
                pulse_algorithms.summarize_data_numba(
                    raw,
                    self.frametime_s,
                    peak_samplenumber=self.peak_index,
                    pretrigger_ignore_samples=self.pretrigger_ignore_samples,
                    nPresamples=self.n_presamples,
                )
            )
            summaries.append(s)

        df2 = pl.concat(summaries).with_columns(df)
        return df2


@dataclass(frozen=True)
class ColumnAsNumpyMapStep(CalStep):
    """
    This step is meant for interactive exploration, it takes a column and applies a function to it,
    and makes a new column with the result. It makes it easy to test functions on a column without
    having to write a whole new step class,
    while maintaining the benefit of being able to use the step in a CalSteps chain, like replaying steps
    on another channel.

    example usage:
    >>> def my_function(x):
    ...     return x * 2
    >>> step = ColumnAsNumpyMapStep(inputs=["my_column"], output=["my_new_column"], f=my_function)
    >>> ch2 = ch.with_step(step)
    """

    f: Callable[[np.ndarray], np.ndarray]

    def __post_init__(self):
        assert len(self.inputs) == 1, "ColumnMapStep expects exactly one input"
        assert len(self.output) == 1, "ColumnMapStep expects exactly one output"
        if not callable(self.f):
            raise ValueError(f"f must be a callable, got {self.f}")

    def calc_from_df(self, df: pl.DataFrame) -> pl.DataFrame:
        output_col = self.output[0]
        output_segments = []
        for df_iter in df.select(self.inputs).iter_slices():
            series1 = df_iter[self.inputs[0]]
            # Have to apply the function differently when series elements are arrays vs scalars
            if series1.dtype.base_type() is pl.Array:
                output_numpy = np.array([self.f(v.to_numpy()) for v in series1])
            else:
                output_numpy = self.f(series1.to_numpy())
            this_output_segment = pl.Series(output_col, output_numpy)
            output_segments.append(this_output_segment)

        combined = pl.concat(output_segments)
        # Put into a DataFrame with one column
        df2 = pl.DataFrame({output_col: combined}).with_columns(df)
        return df2


@dataclass(frozen=True)
class CategorizeStep(CalStep):
    category_condition_dict: dict[str, pl.Expr]

    def __post_init__(self):
        err_msg = "The first condition must be True, to be used as a fallback"
        first_condition = next(iter(self.category_condition_dict.values()))
        assert first_condition is True or first_condition.meta.eq(pl.lit(True)), err_msg

    def calc_from_df(self, df: pl.DataFrame) -> pl.DataFrame:
        output_col = self.output[0]

        def categorize_df(df, category_condition_dict, output_col):
            """returns a series showing which category each pulse is in
            pulses will be assigned to the last category for which the condition evaluates to True"""
            dtype = pl.Enum(category_condition_dict.keys())
            physical = np.zeros(len(df), dtype=int)
            for category_int, (category_str, condition_expr) in enumerate(category_condition_dict.items()):
                if condition_expr is True or condition_expr.meta.eq(pl.lit(True)):
                    in_category = np.ones(len(df), dtype=bool)
                else:
                    in_category = df.select(condition_expr).fill_null(False).to_numpy().flatten()
                assert in_category.dtype == bool
                physical[in_category] = category_int
            series = pl.Series(name=output_col, values=physical).cast(dtype)
            df = pl.DataFrame({output_col: series})
            return df

        df2 = categorize_df(df, self.category_condition_dict, output_col).with_columns(df)
        return df2


@dataclass(frozen=True)
class SelectStep(CalStep):
    """
    This step is meant for interactive exploration, it's basically like the df.select() method, but it's saved as a step.

    """

    col_expr_dict: dict[str, pl.Expr]

    def calc_from_df(self, df: pl.DataFrame) -> pl.DataFrame:
        df2 = df.select(**self.col_expr_dict).with_columns(df)
        return df2


@dataclass(frozen=True)
class CalSteps:
    # leaves many optimizations on the table, but is very simple
    # 1. we could calculate filt_value_5lag and filt_phase_5lag at the same time
    # 2. we could calculate intermediate quantities optionally and not materialize all of them
    steps: list[CalStep]

    def calc_from_df(self, df: pl.DataFrame) -> pl.DataFrame:
        "return a dataframe with all the newly calculated info"
        for step in self.steps:
            df = step.calc_from_df(df).with_columns(df)
        return df

    @classmethod
    def new_empty(cls) -> "CalSteps":
        return cls([])

    def __getitem__(self, key: int) -> CalStep:
        return self.steps[key]

    def __len__(self) -> int:
        return len(self.steps)

    # def copy(self):
    #     # copy by creating a new list containing all the entires in the old list
    #     # a list entry, aka a CalStep, should be immutable
    #     return CalSteps(self.steps[:])

    def with_step(self, step: CalStep) -> "CalSteps":
        # return a new CalSteps with the step added, no mutation!
        return CalSteps(self.steps + [step])

    def trim_dead_ends(self, required_fields: Iterable[str]) -> "CalSteps":
        """Create a new CalSteps object with all dead-end steps removed.

        Dead-end steps are defined as any step that can be omitted without affecting the ability to
        compute any of the fields given in `required_fields`. The result of this method is to return
        a CalSteps where any step is remove if it does not contribute to computing any of the `required_fields`
        (i.e., if it is a dead end).

        Examples of a dead end are typically steps used to prepare a tentative, intermediate calibration function.

        Parameters
        ----------
        required_fields : list[str] | tuple[str] | set[str]
            Steps will be preserved if any of their outputs are among `required_fields`, or if their outputs are
            found recursively among the inputs to any such steps.

        Returns
        -------
        CalSteps
            A copy of `self`, except that any steps not required to compute any of `required_fields` is omitted.

        Raises
        ------
        ValueError
            if the argument is a single string (it should be a tuple, list, or set of strings)
        """
        if isinstance(required_fields, str):
            raise ValueError("CalSteps.trim_dead_ends(rf) wants rf to be Iterable[str], but was passed a string")

        all_fields_out: set[str] = set(required_fields)
        nsteps = len(self)
        required = np.zeros(nsteps, dtype=bool)

        # The easiest approach is to traverse the steps from last to first to build our list of required
        # fields, because necessarily no later step can produce the inputs needed by an earlier step.
        for istep in range(nsteps - 1, -1, -1):
            step = self[istep]
            for field in step.output:
                if field in all_fields_out:
                    required[istep] = True
                    all_fields_out.update(step.inputs)
                    print(f"We require step #{istep}")
                    print(all_fields_out)
                    break

        if np.all(required):
            return self
        steps = []
        for i in range(nsteps):
            if required[i]:
                steps.append(self[i])
        return CalSteps(steps)
