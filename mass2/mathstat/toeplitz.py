"""
toeplitz - General-purpose solver of Toeplitz matrices

Created on Nov 7, 2011

@author: fowlerj
"""

import numpy as np
from scipy.signal import fftconvolve, correlate
from scipy import linalg
from numpy.typing import ArrayLike, NDArray
from dataclasses import dataclass
from typing import overload, Literal


__all__ = [
    "ToeplitzSolver",
    "LowerTriangularToeplitz",
    "UpperTriangularToeplitz",
    "SymmetricToeplitz",
]


@dataclass(frozen=True)
class LowerTriangularToeplitz:
    """Represent a lower triangular Toeplitz matrix. Use FFT methods to multiply it by vectors or matrices."""

    firstcol: np.ndarray

    @classmethod
    def fromLastRow(cls, lastrow: ArrayLike) -> "LowerTriangularToeplitz":
        """Generate an `LowerTriangularToeplitz` from its last row, rather than the default (first column)"""
        vec = np.array(lastrow)[::-1]
        return cls(vec)

    @property
    def N(self) -> int:
        return len(self.firstcol)

    @property
    def isupper(self) -> bool:
        return False

    @property
    def islower(self) -> bool:
        return True

    def _vecmul(self, vec: np.ndarray) -> np.ndarray:
        assert len(vec) == self.N
        return fftconvolve(self.firstcol, vec)[: self.N]

    def _matmul(self, other: np.ndarray) -> np.ndarray:
        # Here are two implementations that both work. I thought that directly controlling the FFT
        # would be faster, but it absolutely was not. So use the simpler one.
        return np.column_stack([self._vecmul(col) for col in other.T])
        # nfft = 2 * self.N
        # k_fft = np.fft.rfft(self.firstcol, n=nfft)
        # other_fft = np.fft.rfft(other.T, n=nfft, axis=1)
        # conv = np.fft.irfft(other_fft * k_fft, n=nfft, axis=1).T
        # return conv[: self.N]

    def __matmul__(self, other: ArrayLike) -> np.ndarray:
        """Implement the `self @ other` syntax, taking the dot product of self with other (a matrix or vector)"""
        other = np.asarray(other)
        assert other.ndim in {1, 2}, "LowerTriangularToeplitz @ x requires x to be of dimension 1 or 2"
        if other.ndim == 1:
            return self._vecmul(other)
        return self._matmul(other)

    def tomatrix(self) -> np.ndarray:
        """Generate a concrete copy of the matrix represented by self

        Returns
        -------
        np.ndarray
            The NxN lower triangular Toeplitz matrix with `self.firstcol` as its first column.
        """
        return linalg.toeplitz(self.firstcol, [self.firstcol[0]] + (self.N - 1) * [0])

    def inverse(self) -> "LowerTriangularToeplitz":
        """Return the inverse of self.

        Returns
        -------
        LowerTriangularToeplitz
            A matrix representing the inverse of self
        """
        b = np.zeros(self.N, dtype=float)
        b[0] = 1.0 / self.firstcol[0]
        for i in range(1, self.N):
            b[i] = -b[0] * np.sum(self.firstcol[1 : i + 1] * b[i - 1 :: -1])
        return LowerTriangularToeplitz(b)

    @property
    def T(self) -> "UpperTriangularToeplitz":
        return UpperTriangularToeplitz(self.firstcol)


@dataclass(frozen=True)
class UpperTriangularToeplitz:
    """Represent an upper triangular Toeplitz matrix. Use FFT methods to multiply it by vectors or matrices."""

    toprow: np.ndarray

    @classmethod
    def fromLastCol(cls, lastcol: ArrayLike) -> "UpperTriangularToeplitz":
        """Generate an `UpperTriangularToeplitz` from its last column, rather than the default (top row)"""
        vec = np.array(lastcol)[::-1]
        return cls(vec)

    @property
    def N(self) -> int:
        return len(self.toprow)

    @property
    def isupper(self) -> bool:
        return True

    @property
    def islower(self) -> bool:
        return False

    def _vecmul(self, vec: np.ndarray) -> np.ndarray:
        assert len(vec) == self.N
        return correlate(vec, self.toprow, mode="full", method="fft")[self.N - 1 :]

    def _matmul(self, other: np.ndarray) -> np.ndarray:
        # Here are two implementations that both work. I thought that directly controlling the FFT
        # would be faster, but it absolutely was not. So use the simpler one.
        return np.column_stack([self._vecmul(col) for col in other.T])
        # nfft = 2 * self.N
        # k_fft = np.fft.rfft(self.toprow, n=nfft)
        # other_fft = np.fft.rfft(other.T, n=nfft, axis=1)
        # corr = np.fft.irfft(other_fft * np.conj(k_fft), n=nfft, axis=1).T
        # return corr[: self.N]

    def __matmul__(self, other: ArrayLike) -> np.ndarray:
        """Implement the `self @ other` syntax, taking the dot product of self with other (a matrix or vector)"""
        other = np.asarray(other)
        assert other.ndim in {1, 2}, "LowerTriangularToeplitz @ x requires x to be of dimension 1 or 2"
        if other.ndim == 1:
            return self._vecmul(other)
        return self._matmul(other)

    def tomatrix(self) -> np.ndarray:
        """Generate a concrete copy of the matrix represented by self

        Returns
        -------
        np.ndarray
            The NxN upper triangular Toeplitz matrix with `self.firstcol` as its first column.
        """
        return linalg.toeplitz([self.toprow[0]] + (self.N - 1) * [0], self.toprow)

    def inverse(self) -> "UpperTriangularToeplitz":
        """Return the inverse of self.

        Returns
        -------
        UpperTriangularToeplitz
            A matrix representing the inverse of self
        """
        b = np.zeros(self.N, dtype=float)
        b[0] = 1.0 / self.toprow[0]
        for i in range(1, self.N):
            b[i] = -b[0] * np.sum(self.toprow[1 : i + 1] * b[i - 1 :: -1])
        return UpperTriangularToeplitz(b)

    @property
    def T(self) -> LowerTriangularToeplitz:
        return LowerTriangularToeplitz(self.toprow)


@overload
def levinson_durbin(r: NDArray, generate_whitener: Literal[False]) -> np.ndarray:
    pass


@overload
def levinson_durbin(r: NDArray, generate_whitener: Literal[True]) -> tuple[np.ndarray, np.ndarray]:
    pass


def levinson_durbin(r: NDArray, generate_whitener: bool = False) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Run the Levinson-Durbin recursion for a symmetric Toeplitz matrix R with the given first column. Find the
    final "forward vector" `f` such that Rf=[1, 0, 0....0]. That forward vector is the first column of inv(R).
    Its reverse (called the "backward vector") `b` satisfies Rb = [0, 0, .... 1], so it is the last column of inv(R).

    Optionally also return the whitening transformation, a square matrix that obeys W @ R @ (W.T) = I. While this
    matrix is implicitly computed as part of the Levinson-Durbin algorithm, storing it explicitly requires O(n^2)
    memory (for input vector `r` having length n). It is best not to compute it fully unless you need it.

    Parameters
    ----------
    r : np.ndarray
        First column and first row of the symmetric Toeplitz matrix being analyzed, of length `n`
    generate_whitener : bool, optional
        Whether to compute and return the exact whitening transformation (an `n`x`n` matrix), by default False.
        The whitener uses `n^2` space, so it should not be created and returned unless the user actually wants it

    Returns
    -------
    np.ndarray | tuple[np.ndarray, np.ndarray]
        Either the final forward vector `f`, or (if `generate_whitener` is True) the
        tuple `(f, W)` where `W` is the `n`x`n` exact whitening matrix.
    """
    n = len(r)
    f = np.zeros(n, dtype=float)
    b = np.zeros(n, dtype=float)

    if generate_whitener:
        W = np.zeros((n, n), dtype=float)
        W[0, 0] = r[0] ** -0.5

    f[0] = b[0] = 1.0 / r[0]
    for k in range(1, n):
        error_k = b[:k] @ r[1 : k + 1]
        scale = 1.0 / (1.0 - error_k**2)
        f[1 : k + 1] -= error_k * b[:k]
        f[: k + 1] *= scale
        b[: k + 1] = f[k::-1]
        if generate_whitener:
            W[k, : k + 1] = b[: k + 1] / np.sqrt(b[k])

    if generate_whitener:
        return (f, W)
    return f


@dataclass(frozen=True)
class SymmetricToeplitz:
    """Represent a Symmetric Toeplitz matrix.

    Create using one of the class methods:
    >>> cfirst = np.array([10, 5, 3, 1])
    >>> clast = cfirst[::-1]
    >>> S = SymmetricToeplitz.fromFirstCol(cfirst)
    >>> S = SymmetricToeplitz.fromLastCol(clast)

    When created in this way, the "forward" and "backward" vectors (the first and last columns of the inverse
    of this matrix) will be computed. This one-time startup cost makes solving this matrix very fast.
    """

    firstcol: np.ndarray
    fvec: np.ndarray
    bvec: np.ndarray
    L1: LowerTriangularToeplitz
    L2: LowerTriangularToeplitz

    @classmethod
    def fromFirstCol(cls, firstcol: ArrayLike) -> "SymmetricToeplitz":
        """Generate a `SymmetricToeplitz` from its first column."""
        firstcol = np.asarray(firstcol)

        # Run the Levinson-Durbin, so we can store the last fw and bw vectors
        fvec = levinson_durbin(firstcol, generate_whitener=False)
        bvec = fvec[::-1]
        l1column = np.asarray(fvec) / fvec[0] ** 0.5
        l2column = np.hstack(([0], bvec[:-1])) / fvec[0] ** 0.5
        L1 = LowerTriangularToeplitz(l1column)
        L2 = LowerTriangularToeplitz(l2column)
        return cls(firstcol, fvec, bvec, L1, L2)

    @classmethod
    def fromLastCol(cls, lastcol: ArrayLike) -> "SymmetricToeplitz":
        """Generate a `SymmetricToeplitz` from its last column."""
        firstcol = np.array(lastcol)[::-1]
        return cls.fromFirstCol(firstcol)

    @property
    def N(self) -> int:
        return len(self.firstcol)

    @property
    def T(self) -> "SymmetricToeplitz":
        return self

    def tomatrix(self) -> np.ndarray:
        """Generate a concrete copy of the matrix represented by self

        Returns
        -------
        np.ndarray
            The NxN symmetric Toeplitz matrix with `self.firstcol` as its first column.
        """
        return linalg.toeplitz(self.firstcol)

    def __matmul__(self, other: ArrayLike) -> np.ndarray:
        """Implement the `self @ other` syntax, taking the dot product of self with other (a matrix or vector)"""
        other = np.asarray(other)
        assert other.ndim in {1, 2}
        if other.ndim == 1:
            return self._multbyvec(other)
        result = np.zeros_like(other)
        for i in range(other.shape[1]):
            result[:, i] = self._multbyvec(other[:, i])
        return result

    def _multbyvec(self, x: np.ndarray) -> np.ndarray:
        """Implement the `self @ other` syntax, taking the dot product of self with a single vector `x`.

        Not for API use, as it assumes `x` is already checked for being 1-dimensional and is a np.ndarray.
        """
        N = len(x)
        y = np.zeros_like(x)
        y[0] = self.firstcol @ x
        for i in range(1, N):
            y[i] = self.firstcol[:-i] @ x[i:]
            y[i] += self.firstcol[1 : 1 + i] @ x[i - 1 :: -1]
        return y

    def solve(self, other: ArrayLike) -> np.ndarray:
        other = np.asarray(other)
        assert other.ndim in {1, 2}
        if other.ndim == 1:
            return self._solvevec(other)
        return np.column_stack([self._solvevec(col) for col in other.T])

    def _solvevec(self, x: np.ndarray) -> np.ndarray:
        """Implement the `inv(self) @ other` or solving self * y = x, for a single right-hand-side vector `x`.

        Not for API use, as it assumes `x` is already checked for being 1-dimensional and is a np.ndarray.
        """
        assert x.ndim == 1
        y = self.L1 @ (self.L1.T @ x)
        y -= self.L2 @ (self.L2.T @ x)
        return y

    def whitener(self) -> np.ndarray:
        return self.Whitener(self.firstcol)

    @staticmethod
    def Whitener(firstcol: ArrayLike) -> np.ndarray:
        _, W = levinson_durbin(np.asarray(firstcol), generate_whitener=True)
        return W


class ToeplitzSolver:
    """Solve a Toeplitz matrix for one or more vectors.

    A Toeplitz matrix is an NxN square matrix where T_ij = R_(i-j) for some
    vector R_k  with k=-(N-1),-(N-2),...-1,0,1,2,...(N-1).
    A symmetric Toeplitz matrix has R_k = R_(-k).

    Initialize the object with the R vector. Careful!  Notice that R is to be specified
    differently depending on the choice of symmetric vs asymmetric matrix.

    Typical usage for a symmetric Toeplitz matrix:
    ac = compute_autocorrelation(...) # ac[0] is 0-lag, ac[1] is lag-1, etc..
    rhs_vect = compute_rhs_vector(...)  # rhs_vect and ac should have same length
    ts = ToeplitzSolver(ac, symmetric=True)
    solution_vect = ts(rhs_vect)

    The solver uses Levinson's algorithm, as explained in Numerical Recipes, 3rd
    Edition section 2.8.2.  For my exact notation, see Joe Fowler's NIST lab book 2
    pages 148-151 (March 30, 2011).

    Timing results from my 4-core 2010-era Mac show that the calculation (as implemented
    on March 31, 2011; with changes, your results may vary) could solve a symmetric
    N=3000 system once in 0.25 seconds, N=5000 in 0.50 seconds, N=8192 in 1.0 seconds,
    N=10k in 1.4 seconds, and N=20k in 4.6 seconds.  Additional solutions to the same
    matrix should take between 0.5 and 0.6 times as long, since the one-time precomputation
    step is approximately as long as the per-solution computations.
    """

    def __init__(self, T: ArrayLike, symmetric: bool = True):
        """Initialize a Toeplitz matrix solver.

        Parameters
        ----------
        T : ArrayLike
            The values in an NxN Toeplitz matrix. The meaning of `T` depends on symmetric.
        symmetric : bool, optional
            Whether the Toeplitz matrix is symmetric, by default True

        When `symmetric` is True, `T` is of length N and gives both the top row and
        the left column of the matrix, which are equal.

        When `symmetric` is False, `T` is of length (2N-1), and matrix T_ij is represented by
        T[i-j+N-1].  Thus T[N-1] is the main diagonal, T[0] is the upper right value of T,
        and T[2*N-2] is the lower left value of T.
        """

        # Whether this Toeplitz matrix is symmetric.
        # Governs how we compute solutions and store the values.
        self.symmetric = symmetric

        # The dimension of the square matrix T.
        T = np.asarray(T)
        self.n = len(T)
        if not symmetric:
            # T needs to be of length 2n-1 for integer n
            assert len(T) % 2 == 1
            self.n = (len(T) + 1) // 2

        # For symmetric matrices, T_(0,0) and T_(1,0) are R[0] and R[1].
        # For non-symmetric, they are R[n-1] and R[n].
        # The non-redundant elements of the Toeplitz matrix.  This will be the top
        # row if symmetric, or otherwise the first column (bottom to top) appended to the rest of
        # the top row.
        self.T = np.array(T).astype(float)

        # It would be good to have a precomputation step for asymmetric matrices, too,
        # but I don't need it now and don't want to spend the time!
        if symmetric:
            self.__precompute_symmetric()

    def mult(self, x: ArrayLike) -> np.ndarray:
        """Return y=Tx

        Currently supported only for symmetric matrices."""
        if not self.symmetric:
            raise NotImplementedError("ToeplitzeSolver.mult(x) is not implemented for asymmetric matrices")
        x = np.asarray(x)
        N = len(x)
        assert N == self.n
        y = np.zeros_like(x)
        y[0] = self.T @ x
        for i in range(1, N):
            y[i] = self.T[:-i] @ x[i:]
            y[i] += self.T[1 : 1 + i] @ x[i - 1 :: -1]
        return y

    def __call__(self, y: ArrayLike) -> np.ndarray:
        """Return the solution x for Tx=y"""
        if self.symmetric:
            return self.__solve_symmetric(y)
        return self.__solve_asymmetric(y)

    def __solve_asymmetric(self, y: ArrayLike) -> np.ndarray:
        """Return the solution x when Tx=y for an asymmetric Toeplitz matrix T."""
        n = self.n
        y = np.asarray(y)
        assert len(y) == n

        x = np.zeros(n, dtype=float)
        g = np.zeros(n, dtype=float)
        h = np.zeros(n, dtype=float)
        xh_denom = np.zeros(n, dtype=float)

        T0 = self.T[n - 1]
        x[0] = y[0] / T0
        g[0] = self.T[n - 2] / T0
        h[0] = self.T[n] / T0

        for K in range(1, n):  # i = m+1
            # Steps b, c, and d (exit test)
            xh_denom[K] = (self.T[n : K + n] * g[:K]).sum() - T0
            x[K] = ((self.T[K + n - 1 : n - 1 : -1] * x[:K]).sum() - y[K]) / xh_denom[K]
            x[:K] -= x[K] * g[K - 1 :: -1]
            if K == n - 1:
                return x

            # Step e
            g_denom = (self.T[n - K - 1 : n - 1] * h[K - 1 :: -1]).sum() - T0
            h[K] = ((self.T[n + K - 1 : n - 1 : -1] * h[:K]).sum() - self.T[K + n]) / xh_denom[K]
            g[K] = ((self.T[n - K - 1 : n - 1] * g[:K]).sum() - self.T[n - K - 2]) / g_denom

            # Step f (careful not to clobber the prev iteration of g)
            gsave = g[:K].copy()
            g[:K] -= g[K] * h[K - 1 :: -1]
            h[:K] -= h[K] * gsave[K - 1 :: -1]
        raise ValueError("unreachable")

    def __precompute_symmetric(self) -> None:
        """Precompute some data so that the solve_symmetric method can be done in
        roughly half the time per solve."""
        n = self.n
        assert self.symmetric

        g = np.zeros(n, dtype=float)
        # The constant denominator of the x_g computation
        self.xg_denom = np.zeros(n, dtype=float)
        # The constant leading value g[K] for each iteration K
        self.gK_leading = np.zeros(n, dtype=float)

        T0 = self.T[0]
        g[0] = self.T[1] / T0

        for K in range(1, n):  # K = M+1
            self.xg_denom[K] = (self.T[1 : K + 1] * g[:K]).sum() - T0
            if K == n - 1:
                return
            g[K] = ((self.T[K:0:-1] * g[:K]).sum() - self.T[K + 1]) / self.xg_denom[K]
            self.gK_leading[K] = g[K]
            g[:K] -= g[K] * g[K - 1 :: -1]
        raise ValueError("unreachable")

    def __solve_symmetric(self, y: ArrayLike) -> np.ndarray:
        """Return the solution x when Tx=y for a symmetric Toeplitz matrix T."""
        y = np.asarray(y)
        if y.ndim == 2:
            result = np.vstack([self.__solve_symmetric(ycol) for ycol in y.T])
            return result.T
        if y.ndim > 2:
            raise ValueError("argument y must be of dimension 1 or 2")

        n = self.n
        assert len(y) == n
        assert self.symmetric

        x = np.zeros(n, dtype=float)
        g = np.zeros(n, dtype=float)

        T = self.T
        T0 = T[0]
        x[0] = y[0] / T0
        g[0] = T[1] / T0

        for K in range(1, n):  # K = M+1
            # Steps b, c, and d (the exit test)
            x[K] = ((T[K:0:-1] * x[:K]).sum() - y[K]) / self.xg_denom[K]
            x[:K] -= x[K] * g[K - 1 :: -1]
            if K == n - 1:
                return x

            # Steps e and f
            g[K] = self.gK_leading[K]
            g[:K] -= g[K] * g[K - 1 :: -1]
        raise ValueError("unreachable")
