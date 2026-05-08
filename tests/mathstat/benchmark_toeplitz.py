from mass2.mathstat.toeplitz import ToeplitzSolver, SymmetricToeplitz
import numpy as np

rng = np.random.default_rng(6823)
SolverLength = 6400
NvectorsGS = 100
NvectorsLevinson = 10


def generate_autocorrelation(N):
    # Use autocorrelation to ensure we have a positive-definite test matrix
    x = rng.standard_normal(N)
    return np.correlate(x, x, mode="full")[N - 1 :]


r = generate_autocorrelation(SolverLength)


def test_ST_create(benchmark):
    benchmark(SymmetricToeplitz.fromFirstCol, r)


def test_ST_solve(benchmark):
    S = SymmetricToeplitz.fromFirstCol(r)
    B = rng.standard_normal((SolverLength, NvectorsGS))
    result = benchmark(S.solve, B)
    assert B.shape == result.shape


def test_TS_create(benchmark):
    benchmark(ToeplitzSolver, r)


def test_TS_solve(benchmark):
    S = ToeplitzSolver(r)
    B = rng.standard_normal((SolverLength, NvectorsLevinson))
    benchmark(S.__call__, B)


"""
def show_may2026_results():
    N = 25 * (2 ** np.arange(11))
    Tcreate = np.array([0.088, 0.18, 0.37, 0.734, 1.55, 3.534, 8.175, 20.97, 57.50, 189.7, 656])
    Tsolve = np.array([0.083, 0.172, 0.342, 0.707, 1.497, 3.419, 8.101, 21.27, 62.35, 207, 715])
    NS = 25 * (2 ** np.arange(13))
    Screate = np.array([0.126, 0.261, 0.503, 1.031, 2.191, 4.965, 15.304, 30.48, 114.51, 323.4, 1127, 4081, 15338])
    ScreateFast = np.array([
        0.070,
        0.136,
        0.279,
        0.575,
        1.163,
        2.531,
        5.641,
        13.456,
        39.26,
        124.38,
        546.89,
        1573.72,
        6117.1,
    ])
    NSs = 25 * (2 ** np.arange(14))
    Ssolve = np.array([0.11, 0.119, 0.119, 0.128, 0.159, 0.181, 0.325, 0.669, 1.145, 1.41, 3.036, 6.136, 13.579, 27.941])
    SsolveFast = np.array([
        0.126,
        0.167,
        0.215,
        0.161,
        0.149,
        0.269,
        0.253,
        0.394,
        0.813,
        2.055,
        3.036,
        5.893,
        13.239,
        27.878,
    ])

    import pylab as plt

    plt.clf()
    plt.loglog()
    plt.plot(N, Tcreate, "o-", color="b", label="create old solver")
    plt.plot(N, Tsolve, "o-", color="r", label="use old solver")
    plt.plot(NS, Screate, "o-", color="c", label="create new solver")
    plt.plot(NS, ScreateFast, "o-", color="g", label="create faster solver")
    plt.plot(NSs, Ssolve, "o-", color="orange", label="use new solver")
    plt.plot(NSs, SsolveFast, "o-", color="gold", label="use faster solver")
    plt.xlabel("Record length")
    plt.ylabel("Time (ms)")
    plt.plot([25, 100000], [0.1, 400], color="gray", lw=0.5)
    plt.plot([500, 100000], [0.12, 4800], color="gray", lw=0.5)
    plt.legend()
"""
