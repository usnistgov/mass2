"""gcss.py

Generalized column subset selection (GCSS) is the problem of selecting a small number of columns
from a "source" matrix A, such that their span optimally captures the columns of a "target" matrix B.

Specifically, we measure "optimal" by the Frobenius norm. Equivalently, we either maximize the norm
of B projected onto the span of the chosen columns, or minimize the norm of the residual (the kernel)
between B and the projected B.

The truly optimal choice is a combinatorically large problem. If there are l columns being chosen from n
columns in A, then only trying all n-choose-l possibilities will find the exact optimum.

However, the obvious greedy algorithm can be used: select columns one at a time according to which would
improve the penalty function the most. Such a function is easy to implement inefficiently, but the paper
by Farahat et al (2013) at https://arxiv.org/abs/1312.6820/ shows a fast method that avoids computing
unneeded intermediate results.
"""

import numpy as np
from numpy.typing import ArrayLike, NDArray


def gcss(A: ArrayLike, B: ArrayLike, ncol: int) -> NDArray:  # noqa: PLR0914
    """Greedy generalized column subset selection (GCSS).

    Find the size-`ncol` subset of columns of `A`, such that a projection of `B` onto the span of the selected
    columns has the largest possible Frobenius norm (or the difference between `B` and that projection has minimal
    norm).

    This function returns an approximation to the optimal GCSS solution, found by selecting one new column at a time,
    choosing whichever most improves the intermediate result. The algorithm for fast computation is that given by
    Farahat et al (2013) at https://arxiv.org/abs/1312.6820/

    Parameters
    ----------
    A : ArrayLike
        A source matrix of size (m, n), from which to select `ncol` of `n` columns
    B : ArrayLike
        A target matrix of size (m, r), which should be well-explained by the selected columns of the source.
    ncol : int
        How many columns to choose from the source. Require 1 <= ncol < n.

    Returns
    -------
    NDArray
        The indices of the selected columns
    """
    A = np.asarray(A)
    B = np.asarray(B)
    m, n = A.shape
    mb, r = B.shape
    assert m == mb, "A and B must have equal number of rows"
    assert ncol > 0, "Must select at least one column of A"
    assert ncol <= n, f"Must select no more than the number of columns in A, {n}"

    BtA = B.T @ A
    AtA = A.T @ A

    g = np.diag(AtA).copy()
    f = (BtA**2).sum(axis=0)

    chosen: list[int] = []
    omega = np.zeros((ncol, n), dtype=float)
    v = np.zeros((ncol, r), dtype=float)
    first_delta = 0.0

    for iter in range(ncol):
        p = (f / g).argmax()
        assert p not in chosen

        delta = AtA[:, p].copy()
        gamma = BtA[:, p].copy()
        if iter == 0:
            first_delta = delta[p]
        for k in range(iter):
            delta -= omega[k, p] * omega[k]
            gamma -= omega[k, p] * v[k]

        # Test whether update rule is starting to fail: delta[p] should not be negative,
        # but in practice it gets so close as to cause numerical problems.
        if delta[p] <= first_delta * 1e-15:
            # Now that it is failing, we have to call gcss recursively with the chosen columns fully eliminated from
            # the source matrix and projected out of the data matrix. This might be inefficient, but I have no
            # better ideas on how to select further columns.
            retained = np.ones(n, dtype=bool)
            for c in chosen:
                retained[c] = False

            newA = A[:, retained]
            selected_col = A[:, ~retained]
            coef = np.linalg.lstsq(selected_col, B)[0]
            newB = B - selected_col @ coef
            subresult = gcss(newA, newB, ncol=ncol - len(chosen))

            # subresult numbers columns from the reduced set found in newA. We have to renumber
            # them according to the previous indices, which we compute in `idx_retained`.
            idx_retained = np.array([i for i in range(n) if retained[i]])
            sub_chosen = idx_retained[subresult]
            assert np.all([s not in chosen for s in sub_chosen])
            return np.sort(np.hstack((chosen, sub_chosen)))

        chosen.append(p)
        # Avoid divide-by-zero problems on next iteration, or accidentally choosing the same column twice.
        g[p] = np.inf

        rescaling = delta[p] ** -0.5
        omega[iter] = delta * rescaling
        v[iter] = gamma * rescaling

        Htv = v[iter] @ BtA
        for rr in range(iter):
            Htv -= (v[rr] @ v[iter]) * omega[rr]
        f -= 2 * omega[iter] * Htv
        f += (v[iter] @ v[iter]) * omega[iter] ** 2
        g -= omega[iter] ** 2
        f[chosen] = 0

    return np.sort(chosen)


def css(A: ArrayLike, ncol: int) -> NDArray:
    """Greedy column subset selection (CSS).

    Find the size-`ncol` subset of columns of `A`, such that a projection of `A` onto the span of the selected
    columns has the largest possible Frobenius norm (or the difference between `A` and that projection has minimal
    norm).

    This function returns an approximation to the optimal CSS solution, found by selecting one new column at a time,
    choosing whichever most improves the intermediate result. The algorithm for fast computation is that given by
    Farahat et al (2013) at https://arxiv.org/abs/1312.6820/

    TODO: this function currently calls `gcss(A, A, ncol)`. It is possible that such an approach performs
    redundant calculations. Consider rewriting rather than calling `gcss`.

    Parameters
    ----------
    A : ArrayLike
        A source matrix of size (m, n), from which to select `ncol` of `n` columns
    ncol : int
        How many columns to choose from the source. Require 1 <= ncol < n.

    Returns
    -------
    NDArray
        The indices of the selected columns
    """
    return gcss(A, A, ncol)
