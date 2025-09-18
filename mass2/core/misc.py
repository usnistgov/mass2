"""
Miscellaneous utility functions used in mass2 for plotting, pickling, statistics, and DataFrame manipulation.
"""

from numpy.typing import ArrayLike, NDArray
from typing import Any
from pathlib import Path
import numpy as np
import pylab as plt
import polars as pl
import subprocess
import sys
import pathlib
import dill
import marimo as mo


def show(fig: plt.Figure | None = None) -> mo.Html:
    """Create a Marimo interactive view of the given Matplotlib figure (or the current figure if None)."""
    if fig is None:
        fig = plt.gcf()
    return mo.mpl.interactive(fig)


def pickle_object(obj: Any, filename: str | Path) -> None:
    """Pickle the given object to the given filename using dill.
    Mass2 Recipe objects are compatible with `dill` but _not_ with the standard `pickle` module."""
    with open(filename, "wb") as file:
        dill.dump(obj, file)


def unpickle_object(filename: str | Path) -> Any:
    """Unpickle an object from the given filename using dill."""
    with open(filename, "rb") as file:
        obj = dill.load(file)
        return obj


def smallest_positive_real(arr: ArrayLike) -> float:
    """Return the smallest positive real number in the given array-like object."""

    def is_positive_real(x: Any) -> bool:
        "Is `x` a positive real number?"
        return x > 0 and np.isreal(x)

    positive_real_numbers = np.array(list(filter(is_positive_real, np.asarray(arr))))
    return np.min(positive_real_numbers)


def good_series(df: pl.DataFrame, col: str, good_expr: pl.Expr, use_expr: bool | pl.Expr) -> pl.Series:
    """Return a Series from the given DataFrame column, filtered by the given good_expr and use_expr."""
    # This uses lazy before filtering. We hope this will allow polars to only access the data needed to filter
    # and the data needed to output what we want.
    good_df = df.lazy().filter(good_expr)
    if use_expr is not True:
        good_df = good_df.filter(use_expr)
    return good_df.select(pl.col(col)).collect().to_series()


def median_absolute_deviation(x: ArrayLike) -> float:
    """Return the median absolute deviation of the input, unnormalized."""
    x = np.asarray(x)
    return float(np.median(np.abs(x - np.median(x))))


def sigma_mad(x: ArrayLike) -> float:
    """Return the nomrlized median absolute deviation of the input, rescaled to give the standard deviation
    if distribution is Gaussian. This method is more robust to outliers than calculating the standard deviation directly."""
    return median_absolute_deviation(x) * 1.4826


def outlier_resistant_nsigma_above_mid(x: ArrayLike, nsigma: float = 5) -> float:
    """RReturn the value that is `nsigma` median absolute deviations (MADs) above the median of the input."""
    x = np.asarray(x)
    mid = np.median(x)
    return mid + nsigma * sigma_mad(x)


def outlier_resistant_nsigma_range_from_mid(x: ArrayLike, nsigma: float = 5) -> tuple[float, float]:
    """Return the values that are `nsigma` median absolute deviations (MADs) below and above the median of the input"""
    x = np.asarray(x)
    mid = np.median(x)
    smad = sigma_mad(x)
    return mid - nsigma * smad, mid + nsigma * smad


def midpoints_and_step_size(x: ArrayLike) -> tuple[NDArray, float]:
    """return midpoints, step_size for bin edges x"""
    x = np.asarray(x)
    d = np.diff(x)
    step_size = float(d[0])
    assert np.allclose(d, step_size, atol=1e-9), f"{d=}"
    return x[:-1] + step_size, step_size


def hist_of_series(series: pl.Series, bin_edges: ArrayLike) -> tuple[NDArray, NDArray]:
    """Return the bin centers and counts of a histogram of the given Series using the given bin edges."""
    bin_edges = np.asarray(bin_edges)
    bin_centers, _ = midpoints_and_step_size(bin_edges)
    counts = series.rename("count").hist(list(bin_edges), include_category=False, include_breakpoint=False)
    return bin_centers, counts.to_numpy().T[0]


def plot_hist_of_series(series: pl.Series, bin_edges: ArrayLike, axis: plt.Axes | None = None, **plotkwarg: dict) -> plt.Axes:
    """Plot a histogram of the given Series using the given bin edges on the given axis (or a new one if None)."""
    if axis is None:
        plt.figure()
        axis = plt.gca()
    bin_edges = np.asarray(bin_edges)
    bin_centers, step_size = midpoints_and_step_size(bin_edges)
    hist = series.rename("count").hist(list(bin_edges), include_category=False, include_breakpoint=False)
    axis.plot(bin_centers, hist, label=series.name, **plotkwarg)
    axis.set_xlabel(series.name)
    axis.set_ylabel(f"counts per {step_size:.2f} unit bin")
    return axis


def plot_a_vs_b_series(a: pl.Series, b: pl.Series, axis: plt.Axes | None = None, **plotkwarg: dict) -> None:
    """Plot the two given Series as a scatterplot on the given axis (or a new one if None)."""
    if axis is None:
        plt.figure()
        axis = plt.gca()
    axis.plot(a, b, ".", label=b.name, **plotkwarg)
    axis.set_xlabel(a.name)
    axis.set_ylabel(b.name)


def launch_examples() -> None:
    """Launch marimo edit in the examples folder."""
    examples_folder = pathlib.Path(__file__).parent.parent.parent / "examples"
    # use relative path to avoid this bug: https://github.com/marimo-team/marimo/issues/1895
    examples_folder_relative = str(examples_folder.relative_to(pathlib.Path.cwd()))
    # Prepare the command
    command = ["marimo", "edit", examples_folder_relative] + sys.argv[1:]

    # Execute the command
    print(f"launching marimo edit in {examples_folder_relative}")
    try:
        # Execute the command and directly forward stdout and stderr
        process = subprocess.Popen(command, stdout=sys.stdout, stderr=sys.stderr)
        process.communicate()

    except KeyboardInterrupt:
        # Handle cleanup on Ctrl-C
        try:
            process.terminate()
        except OSError:
            pass
        process.wait()
        sys.exit(1)

    # Check if the command was successful
    if process.returncode != 0:
        sys.exit(process.returncode)


def root_mean_squared(x: ArrayLike, axis: int | tuple[int] | None = None) -> float:
    """Return the root mean square of the input along the given axis or axes.
    Does _not_ subtract the mean first."""
    return np.sqrt(np.mean(np.asarray(x) ** 2, axis))


def merge_dicts_ordered_by_keys(dict1: dict[int, Any], dict2: dict[int, Any]) -> dict[int, Any]:
    """Merge two dictionaries and return a new dictionary with items ordered by key."""
    # Combine both dictionaries' items (key, value) into a list of tuples
    combined_items = list(dict1.items()) + list(dict2.items())

    # Sort the combined list of tuples by key
    combined_items.sort(key=lambda item: item[0])

    # Convert the sorted list of tuples back into a dictionary
    merged_dict: dict[int, Any] = {key: value for key, value in combined_items}

    return merged_dict


def concat_dfs_with_concat_state(df1: pl.DataFrame, df2: pl.DataFrame, concat_state_col: str = "concat_state") -> pl.DataFrame:
    """Concatenate two DataFrames vertically, adding a column `concat_state` (or named according to `concat_state_col`)
    to indicate which DataFrame each row came from."""
    if concat_state_col in df1.columns:
        # Continue incrementing from the last known concat_state
        max_state = df1[concat_state_col][-1]
        df2 = df2.with_columns(pl.lit(max_state + 1).alias(concat_state_col))
    else:
        # Fresh concat: label first as 0, second as 1
        df1 = df1.with_columns(pl.lit(0).alias(concat_state_col))
        df2 = df2.with_columns(pl.lit(1).alias(concat_state_col))

    df_out = pl.concat([df1, df2], how="vertical")
    return df_out


def extract_column_names_from_polars_expr(expr: pl.Expr) -> list[str]:
    """Recursively extract all column names from a Polars expression."""
    names = set()
    if hasattr(expr, "meta"):
        meta = expr.meta
        if hasattr(meta, "root_names"):
            # For polars >=0.19.0
            names.update(meta.root_names())
        elif hasattr(meta, "output_name"):
            # For older polars
            names.add(meta.output_name())
        if hasattr(meta, "inputs"):
            for subexpr in meta.inputs():
                names.update(extract_column_names_from_polars_expr(subexpr))
    return list(names)


def alwaysTrue() -> pl.Expr:
    """alwaysTrue: a factory function to generate a new copy of polars literal True for class construction

    Returns
    -------
    pl.Expr
        Literal True
    """
    return pl.lit(True)
