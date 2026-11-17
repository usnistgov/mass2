import numpy as np
import polars as pl

import mass2


def test_midpoints_are_halfway_between_bin_edges():
    bin_edges = np.linspace(0, 10, 11)
    midpoints, step_size = mass2.misc.midpoints_and_step_size(bin_edges)
    assert step_size == 1.0
    expected = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    assert np.allclose(midpoints, expected), f"{midpoints=} should be halfway between edges"


def test_hist_of_series_bin_centers_fall_inside_their_bins():
    bin_edges = np.array([0.0, 2.0, 4.0, 6.0])
    series = pl.Series("x", [0.5, 2.5, 2.9, 5.0])
    bin_centers, counts = mass2.misc.hist_of_series(series, bin_edges)
    assert len(bin_centers) == len(counts)
    assert np.all(bin_centers > bin_edges[:-1])
    assert np.all(bin_centers < bin_edges[1:])
