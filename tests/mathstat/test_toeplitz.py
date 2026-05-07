"""
Unit tests for MASS utilities.
(So far, only the Toeplitz solver)

Also contains a speed test object TestToeplitzSpeed

J. Fowler, NIST

March 30, 2011
"""

import pytest
from mass2.mathstat.toeplitz import ToeplitzSolver, LowerTriangularToeplitz, UpperTriangularToeplitz
from mass2.mathstat.toeplitz import levinson_durbin, SymmetricToeplitz
import numpy as np
from scipy import linalg
import time
import pylab as plt

rng = np.random.default_rng(6823)


class TestToeplitzSolverSmallSymmetric:
    """Test ToeplitzSolver on a 5x5 symmetric matrix."""

    def setup_method(self):
        self.autocorr = np.array((6.0, 4.0, 2.0, 1.0, 0.0))
        self.n = len(self.autocorr)
        self.solver = ToeplitzSolver(self.autocorr, symmetric=True)
        self.R = linalg.toeplitz(self.autocorr)

    def test_all_unit_vectors(self):
        for i in range(self.n):
            x_in = np.zeros(self.n, dtype=float)
            x_in[i] = 1.0
            y = self.R @ x_in
            x_out = self.solver(y)
            big_dif = np.abs(x_out - x_in).max()
            assert 0 == pytest.approx(big_dif, abs=1e-12)

    def test_arb_vectors(self):
        for _i in range(self.n):
            x_in = 5 * rng.standard_normal(self.n)
            y = self.R @ x_in
            x_out = self.solver(y)
            big_dif = np.abs(x_out - x_in).max()
            assert 0 == pytest.approx(big_dif, abs=1e-12)


class TestToeplitzSolverSmallAsymmetric:
    """Test ToeplitzSolver on a 5x5 non-symmetric matrix."""

    def setup_method(self):
        self.autocorr = np.asarray((-1, -2, 0, 3, 6.0, 4.0, 2.0, 1.0, 0.0))
        self.n = (len(self.autocorr) + 1) // 2
        self.solver = ToeplitzSolver(self.autocorr, symmetric=False)
        self.R = linalg.toeplitz(self.autocorr[self.n - 1 :], self.autocorr[self.n - 1 :: -1])

    def test_all_unit_vectors(self):
        for i in range(self.n):
            x_in = np.zeros(self.n, dtype=float)
            x_in[i] = 1.0
            y = self.R @ x_in
            x_out = self.solver(y)
            big_dif = np.abs(x_out - x_in).max()
            assert 0 == pytest.approx(big_dif, abs=1e-12)

    def test_arb_vectors(self):
        for _i in range(self.n):
            x_in = 5 * rng.standard_normal(self.n)
            y = self.R @ x_in
            x_out = self.solver(y)
            big_dif = np.abs(x_out - x_in).max()
            assert 0 == pytest.approx(big_dif, abs=1e-12)


class TestToeplitzSolver_32:
    """Test ToeplitzSolver on a 32x32 symmetric matrix."""

    def setup_method(self):
        self.n = 32
        t = np.arange(self.n)
        t[0] = 1
        pi = np.pi
        T = 1.0 * self.n
        self.autocorr = np.sin(pi * t / T) / (pi * t / T)
        self.autocorr[0] = 1
        self.autocorr[:5] *= np.arange(5, 0.5, -1)
        self.solver = ToeplitzSolver(self.autocorr, symmetric=True)
        self.R = linalg.toeplitz(self.autocorr)

    def test_all_unit_vectors(self):
        for i in range(self.n):
            x_in = np.zeros(self.n, dtype=float)
            x_in[i] = 1.0
            y = self.R @ x_in
            x_out = self.solver(y)
            big_dif = np.abs(x_out - x_in).max()
            assert 0 == pytest.approx(big_dif, abs=1e-10), f"Unit vector trial i={i:2d} gives x_out={x_out}"


class TestToeplitzSolver_512:
    """Test ToeplitzSolver on a 512x512 symmetric matrix."""

    def setup_method(self):
        self.n = 512
        t = np.arange(self.n)
        self.autocorr = 1.0 + 3.2 * np.exp(-t / 100.0)
        self.autocorr[0] = 9
        self.solver = ToeplitzSolver(self.autocorr, symmetric=True)
        self.R = linalg.toeplitz(self.autocorr)

    def test_some_unit_vectors(self):
        for i in (0, 20, 128, 256, 500, 512 - 1):
            x_in = np.zeros(self.n, dtype=float)
            x_in[i] = 1.0
            y = self.R @ x_in
            x_out = self.solver(y)
            big_dif = np.abs(x_out - x_in).max()
            assert 0 == pytest.approx(big_dif, abs=1e-10), f"Unit vector trial i={i:2d} gives x_out={x_out}"

    def test_arb_vectors(self):
        for _i in range(5):
            x_in = 5 * rng.standard_normal(self.n)
            y = self.R @ x_in
            x_out = self.solver(y)
            big_dif = np.abs(x_out - x_in).max()
            assert 0 == pytest.approx(big_dif, abs=1e-10), f"Random vector trial gives rms diff={(x_out - x_in).std()}"


class toeplitzSpeed:
    """Test the speed of the Toeplitz solver.

        This is NOT a unit test. Usage:

    >>> from mass2.mathstat.test import test_toeplitz
    >>> t=test_toeplitz.toeplitzSpeed()
    >>> t.plot()
    """

    def __init__(self, maxsize=8192):
        self.sizes = np.hstack((100, 200, np.arange(500, 5500, 500), 6144, 8192, 10000, 20000, 30000, 50000))
        t = np.arange(100000)
        self.autocorr = 1.0 + 3.2 * np.exp(-t / 100.0)
        self.autocorr[0] = 9

        self.ts_time = np.zeros(len(self.sizes), dtype=float)
        self.build_time = np.zeros_like(self.ts_time)
        self.mult_time = np.zeros_like(self.ts_time)
        self.solve_time = np.zeros_like(self.ts_time)
        self.lu_time = np.zeros_like(self.ts_time)
        for i, s in enumerate(self.sizes):
            times = self.test(s, maxsize)
            (self.ts_time[i], self.build_time[i], self.mult_time[i], self.solve_time[i], self.lu_time[i]) = times

    def test(self, size, maxsize=8192):
        if size > 150000:
            return 5 * [np.nan]

        ac = self.autocorr[:size]
        v = rng.standard_normal(size)

        t0 = time.time()
        solver = ToeplitzSolver(ac, symmetric=True)
        x = solver(v)
        dt = [time.time() - t0]

        if size <= maxsize:
            # dt[1] = creating R time
            t0 = time.time()
            R = linalg.toeplitz(ac)
            dt.append(time.time() - t0)

            # dt[2] = R * vector time
            t0 = time.time()
            v2 = R @ x
            dt.append(time.time() - t0)

            # dt[3] = solve(R,v) time
            t0 = time.time()
            x2 = np.linalg.solve(R, v)
            dt.append(time.time() - t0)

            t0 = time.time()
            lu_piv = linalg.lu_factor(R)
            x3 = linalg.lu_solve(lu_piv, v, overwrite_b=False)
            dt.append(time.time() - t0)
            print(f"rms rhs diff: {(v - v2).std():.3g}, solution diff: {(x - x2).std():.3g} {(x - x3).std():.3g}")

        else:
            dt.extend(4 * [np.nan])
        print(size, [f"{t:6.3f}" for t in dt])
        return dt

    def plot(self):
        plt.clf()
        plt.plot(self.sizes, self.ts_time, label="Toeplitz solver")
        plt.plot(self.sizes, self.build_time, label="Matrix build")
        plt.plot(self.sizes, self.mult_time, label="Matrix-vector mult")
        plt.plot(self.sizes, self.solve_time, label="Matrix solve")
        plt.plot(self.sizes, self.lu_time, label="LU solve")
        plt.legend(loc="upper left")
        plt.xlabel("Vector size")
        plt.ylabel("Time (sec)")
        plt.grid()
        plt.loglog()


def test_triangular_toeplitz():
    N = 40
    nvec = 10
    vec = rng.standard_exponential(N)

    L = LowerTriangularToeplitz(vec)
    assert N == L.N
    assert not L.isupper
    assert L.islower

    Lexact = L.tomatrix()
    for i in range(N):
        for j in range(N):
            if j <= i:
                assert Lexact[i, j] == L.firstcol[i - j]
            else:
                assert Lexact[i, j] == 0
    L2 = LowerTriangularToeplitz.fromLastRow(Lexact[-1, :])

    testv = rng.standard_normal(N)
    assert np.allclose(Lexact @ testv, L @ testv)
    assert np.allclose(Lexact @ testv, L2 @ testv)
    assert np.allclose(L2.firstcol, L.firstcol)

    testm = rng.standard_cauchy((N, nvec))
    assert np.allclose(Lexact @ testm, L @ testm)

    U = UpperTriangularToeplitz(vec)
    assert N == U.N
    assert U.isupper
    assert not U.islower
    Uexact = U.tomatrix()
    assert np.allclose(Uexact @ testv, U @ testv)
    assert np.allclose(Uexact @ testm, U @ testm)

    # Test inverse on small matrices
    Lsm = LowerTriangularToeplitz(vec[:15])
    Linv = Lsm.inverse()
    assert np.allclose(Linv @ Lsm.tomatrix(), np.eye(Lsm.N), atol=1e-4, rtol=0.001)
    Usm = UpperTriangularToeplitz(vec[N - 15 :])
    Uinv = Usm.inverse()
    assert np.allclose(Uinv @ Usm.tomatrix(), np.eye(Usm.N), atol=1e-5, rtol=0.001)


def generate_autocorrelation(N=50):
    # Use autocorrelation to ensure we have a positive-definite test matrix
    x = rng.standard_normal(N)
    return np.correlate(x, x, mode="full")[N - 1 :]


def test_levinson_durbin():
    N = 50
    r = generate_autocorrelation(N)
    R = linalg.toeplitz(r)
    fw1 = levinson_durbin(r, generate_whitener=False)
    fw, W = levinson_durbin(r, generate_whitener=True)
    bw = fw[::-1]
    assert np.allclose(fw1, fw)

    # Verify that forward and backward vectors are first and last column of inverse
    assert np.allclose(R @ bw, [0] * (N - 1) + [1])
    assert np.allclose(R @ fw, [1] + [0] * (N - 1))

    # Verify that whitener does what it promises
    WRWt = W @ R @ (W.T)
    print(WRWt)
    assert np.allclose(WRWt, np.eye(N))


def test_symmetric_toep():
    N = 30

    # Generate a positive-definite size-N SymmetricToeplitz matrix
    r = generate_autocorrelation(N)
    S = SymmetricToeplitz.fromFirstCol(r)
    R = S.tomatrix()
    assert np.allclose(R, linalg.toeplitz(r))

    # Test we can form SymmetricToeplitz from the last column, as well as the first.
    S2 = SymmetricToeplitz.fromLastCol(r[::-1])
    assert np.allclose(R, S2.tomatrix())

    # Check S@x == R@x for any x. Use a matrix, B
    nvecs = 5
    B = rng.standard_normal((N, nvecs))
    assert np.allclose(S @ B, R @ B)

    # Verify that whitener does what it promises
    W = S.whitener()
    WRWt = W @ R @ (W.T)
    assert np.allclose(WRWt, np.eye(N))

    # Check for solving Sx=b. Verify that solution obeys _both_ Sx=b and Rx=b.
    # This is testing the Gohberg-Semencul formula.
    b = rng.standard_normal(N)
    x = S.solve(b)
    assert np.allclose(b, R @ x)

    X = S.solve(B)
    assert np.allclose(B, S @ X)
