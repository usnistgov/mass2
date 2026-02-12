"""
Unit tests for MASS utilities.
(So far, only the find_svd_randomly)

J. Fowler, NIST

February 12, 2026
"""

from mass2.mathstat.utilities import find_svd_randomly
import numpy as np

rng = np.random.default_rng()


def test_find_svd_randomly():
    A = rng.standard_normal((5, 5))
    _ = find_svd_randomly(A, 2)
