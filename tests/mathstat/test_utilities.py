"""
Unit tests for MASS utilities.
(So far, only the find_svd_randomly)

J. Fowler, NIST

February 12, 2026
"""

from mass2.mathstat.utilities import find_svd_randomly
import numpy as np
import pytest

rng = np.random.default_rng(2783)


def test_find_svd_randomly():
    A = rng.standard_normal((5, 2)) @ rng.standard_normal((2, 5))
    u, s, vh = find_svd_randomly(A, 2)
    assert u @ np.diag(s) @ vh == pytest.approx(A, abs=1e-10)
    assert s == pytest.approx(np.linalg.svd(A, compute_uv=False)[:2], rel=1e-10)
