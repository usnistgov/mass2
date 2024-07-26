import numpy as np
import pylab as plt
import polars as pl
import dill


def pickle_object(obj, filename):
    with open(filename, 'wb') as file:
        dill.dump(obj, file)

def unpickle_object(filename):
    with open(filename, 'rb') as file:
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
    return (
        df.lazy()
        .filter(good_expr)
        .filter(use_expr)
        .select(pl.col(col))
        .collect()
        .to_series()
    )

def median_absolute_deviation(x):
    return np.median(np.abs(x - np.median(x)))


def sigma_mad(x):
    return median_absolute_deviation(x) * 1.4826


def outlier_resistant_nsigma_above_mid(x, nsigma=5):
    mid = np.median(x)
    mad = np.median(np.abs(x - mid))
    sigma_mad = mad * 1.4826
    return mid + nsigma * sigma_mad

def midpoints_and_step_size(x):
    d = np.diff(x)
    step_size = d[0]
    assert np.allclose(d, step_size, atol=1e-7)
    return x[:-1] + step_size, step_size

def hist_of_series(series, bin_edges):
    bin_centers, step_size = midpoints_and_step_size(bin_edges)
    counts = series.rename("count").hist(
        bin_edges, include_category=False, include_breakpoint=False
    )[1:-1]
    return bin_centers, counts.to_numpy().T[0]

def plot_hist_of_series(series, bin_edges, axis=None, **plotkwarg):
    if axis is None:
        plt.figure()
        axis = plt.gca()
    bin_centers, step_size = midpoints_and_step_size(bin_edges)
    hist = series.rename("count").hist(
        bin_edges, include_category=False, include_breakpoint=False
    )[1:-1]
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
    import subprocess
    import sys
    import pathlib

    folder = pathlib.Path(__file__).parent.parent/"examples"
    # Prepare the command
    command = ["marimo", "edit", folder] + sys.argv[1:]
    
    # Execute the command
    print(f"launching marimo edit in {folder}")
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