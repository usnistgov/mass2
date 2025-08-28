"""
mass2.mathstat.utilities

Several math and plotting utilities:
* plot_as_stepped_hist
* plot_stepped_hist_poisson_errors

Joe Fowler, NIST

Started March 24, 2011
"""

from typing import Any
from numpy.typing import ArrayLike, NDArray
import numpy as np
import pylab as plt
from collections import namedtuple

__all__ = ["plot_as_stepped_hist", "plot_stepped_hist_poisson_errors", "find_svd_randomly", "find_range_randomly"]

# Create a module-local RNG. If you need to seed it to achieve repeatable tests,
# you can replace this.
rng = np.random.default_rng()


def plot_as_stepped_hist(axis: plt.Axes, data: ArrayLike, bins: ArrayLike, **kwargs: Any) -> None:
    """Plot data in stepped-histogram format.

    Args:
        axis: The pylab Axes object to plot onto.
        data: Bin contents.
        bins: An array of bin centers or of bin edges.  (Bin spacing will be
            inferred from the first two elements).  If len(bin_ctrs) == len(data)+1, then
            bin_ctrs will be assumed to be bin edges; otherwise it will be assumed to be
            the bin centers.
        **kwargs: All other keyword arguments will be passed to axis.plot().
    """
    data = np.asarray(data)
    bins = np.asarray(bins)
    if len(bins) == len(data) + 1:
        bin_edges = bins
        x = np.zeros(2 * len(bin_edges), dtype=float)
        x[0::2] = bin_edges
        x[1::2] = bin_edges
    else:
        x = np.zeros(2 + 2 * len(bins), dtype=float)
        dx = bins[1] - bins[0]
        x[0:-2:2] = bins - dx * 0.5
        x[1:-2:2] = bins - dx * 0.5
        x[-2:] = bins[-1] + dx * 0.5

    y = np.zeros_like(x)
    y[1:-1:2] = data
    y[2:-1:2] = data
    axis.plot(x, y, **kwargs)
    axis.set_xlim([x[0], x[-1]])


def plot_stepped_hist_poisson_errors(
    axis: plt.Axes, counts: ArrayLike, bin_ctrs: ArrayLike, scale: float = 1.0, offset: float = 0.0, **kwargs: Any
) -> None:
    """Use plot_as_stepped_hist to plot a histogram, where also
    an error band is plotted, assuming data are Poisson-distributed.

    Args:
        axis: The pylab Axes object to plot onto.
        data: Bin contents.
        bin_ctrs: An array of bin centers or of bin edges.  (Bin spacing will be
            inferred from the first two elements).  If len(bin_ctrs) == len(data)+1, then
            bin_ctrs will be assumed to be bin edges; otherwise it will be assumed to be
            the bin centers.
        scale: Plot counts*scale+offset if you need to convert counts to some physical units.
        offset: Plot counts*scale+offset if you need to convert counts to some physical units.
        **kwargs: All other keyword arguments will be passed to axis.plot().
    """
    counts = np.asarray(counts)
    bin_ctrs = np.asarray(bin_ctrs)
    if len(bin_ctrs) == len(counts) + 1:
        bin_ctrs = 0.5 * (bin_ctrs[1:] + bin_ctrs[:-1])
    elif len(bin_ctrs) != len(counts):
        raise ValueError("bin_ctrs must be either the same length as counts, or 1 longer.")
    smooth_counts = counts * scale
    errors = np.sqrt(counts) * scale
    fill_lower = smooth_counts - errors
    fill_upper = smooth_counts + errors
    fill_lower[fill_lower < 0] = 0
    fill_upper[fill_upper < 0] = 0
    axis.fill_between(bin_ctrs, fill_lower + offset, fill_upper + offset, alpha=0.25, **kwargs)
    plot_as_stepped_hist(axis, counts * scale + offset, bin_ctrs, **kwargs)


def find_range_randomly(A: ArrayLike, nl: int, q: int = 1) -> NDArray:
    """Find approximate range of matrix A using nl random vectors and
    with q power iterations (q>=0). Based on Halko Martinsson & Tropp Algorithm 4.3

    Suggest q=1 or larger particularly when the singular values of A decay slowly
    enough that the singular vectors associated with the smaller singular values
    are interfering with the computation.

    See "Finding structure with randomness: Probabilistic algorithms for constructing
    approximate matrix decompositions." by N Halko, P Martinsson, and J Tropp. *SIAM
    Review* v53 #2 (2011) pp217-288. http://epubs.siam.org/doi/abs/10.1137/090771806
    """
    if q < 0:
        msg = f"The number of power iterations q={q} needs to be at least 0"
        raise ValueError(msg)
    A = np.asarray(A)
    _m, n = A.shape
    Omega = rng.standard_normal((n, nl))
    Y = np.dot(A, Omega)
    for _ in range(q):
        Y = np.dot(A.T, Y)
        Y = np.dot(A, Y)
    Q, _R = np.linalg.qr(Y)
    return Q


def find_svd_randomly(A: ArrayLike, nl: int, q: int = 2) -> tuple[NDArray, NDArray, NDArray]:
    """Find approximate SVD of matrix A using nl random vectors and
    with q power iterations. Based on Halko Martinsson & Tropp Algorithm 5.1

    See "Finding structure with randomness: Probabilistic algorithms for constructing
    approximate matrix decompositions." by N Halko, P Martinsson, and J Tropp. *SIAM
    Review* v53 #2 (2011) pp217-288. http://epubs.siam.org/doi/abs/10.1137/090771806
    """
    A = np.asarray(A)
    Q = find_range_randomly(A, nl, q=q)
    B = np.dot(Q.T, A)
    SVD_B = np.linalg.svd(B, full_matrices=False)
    u = np.dot(Q, SVD_B.U)
    SVDResult = namedtuple("SVDResult", ["U", "W", "Vh"])
    return SVDResult(U=u, W=SVD_B.W, Vh=SVD_B.Vh)
