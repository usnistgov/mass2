import numpy as np
import pylab as plt
import polars as pl
import subprocess
import sys
import pathlib
import dill
import marimo as mo
from typing import Any


def show(fig=None):
    if fig is None:
        fig = plt.gcf()
    return mo.mpl.interactive(fig)


def pickle_object(obj, filename):
    with open(filename, "wb") as file:
        dill.dump(obj, file)


def unpickle_object(filename):
    with open(filename, "rb") as file:
        obj = dill.load(file)
        return obj


def smallest_positive_real(arr):
    def is_positive_real(x):
        return x > 0 and np.isreal(x)

    positive_real_numbers = np.array(list(filter(is_positive_real, arr)))
    return np.min(positive_real_numbers)


def good_series(df, col, good_expr, use_expr):
    # this uses lazy before filting to hopefully allow polars to only access the data needed to filter
    # and the data needed to output what we want
    return df.lazy().filter(good_expr).filter(use_expr).select(pl.col(col)).collect().to_series()


def median_absolute_deviation(x):
    return np.median(np.abs(x - np.median(x)))


def sigma_mad(x):
    return median_absolute_deviation(x) * 1.4826


def outlier_resistant_nsigma_above_mid(x, nsigma=5):
    mid = np.median(x)
    mad = np.median(np.abs(x - mid))
    sigma_mad = mad * 1.4826
    return mid + nsigma * sigma_mad


def outlier_resistant_nsigma_range_from_mid(x, nsigma=5):
    mid = np.median(x)
    mad = np.median(np.abs(x - mid))
    sigma_mad = mad * 1.4826
    return mid - nsigma * sigma_mad, mid + nsigma * sigma_mad


def midpoints_and_step_size(x):
    d = np.diff(x)
    step_size = d[0]
    assert np.allclose(d, step_size, atol=1e-9), f"{d=}"
    return x[:-1] + step_size, step_size


def hist_of_series(series, bin_edges):
    bin_centers, step_size = midpoints_and_step_size(bin_edges)
    counts = series.rename("count").hist(bin_edges, include_category=False, include_breakpoint=False)
    return bin_centers, counts.to_numpy().T[0]


def plot_hist_of_series(series, bin_edges, axis=None, **plotkwarg):
    if axis is None:
        plt.figure()
        axis = plt.gca()
    bin_centers, step_size = midpoints_and_step_size(bin_edges)
    hist = series.rename("count").hist(bin_edges, include_category=False, include_breakpoint=False)
    axis.plot(bin_centers, hist, label=series.name, **plotkwarg)
    axis.set_xlabel(series.name)
    axis.set_ylabel(f"counts per {step_size:.2f} unit bin")
    return axis


def plot_a_vs_b_series(a, b, axis=None, **plotkwarg):
    if axis is None:
        plt.figure()
        axis = plt.gca()
    axis.plot(a, b, ".", label=b.name, **plotkwarg)
    axis.set_xlabel(a.name)
    axis.set_ylabel(b.name)


def launch_examples():
    examples_folder = pathlib.Path(__file__).parent.parent.parent / "examples"
    # use relative path to avoid this bug: https://github.com/marimo-team/marimo/issues/1895
    examples_folder_relative = examples_folder.relative_to(pathlib.Path.cwd())
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


def root_mean_squared(x, axis=None):
    return np.sqrt(np.mean(x**2, axis))


def merge_dicts_ordered_by_keys(dict1: dict[int, Any], dict2: dict[int, Any]) -> dict[int, Any]:
    # Combine both dictionaries' items (key, value) into a list of tuples
    combined_items = list(dict1.items()) + list(dict2.items())

    # Sort the combined list of tuples by key
    combined_items.sort(key=lambda item: item[0])

    # Convert the sorted list of tuples back into a dictionary
    merged_dict: dict[int, Any] = {key: value for key, value in combined_items}

    return merged_dict


def concat_dfs_with_concat_state(df1: pl.DataFrame, df2: pl.DataFrame, concat_state_col: str = "concat_state"):
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
