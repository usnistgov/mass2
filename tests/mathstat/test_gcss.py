import pytest
import numpy as np
from mass2.mathstat.gcss import gcss, css

rng = np.random.default_rng()


def test_gcss():
    """Test the Generalized Column Subset Selection functions"""
    m = 10
    n = 10
    r = 40

    # Make a B where the first 3 columns of A are a perfect match
    A = rng.standard_normal((m, n))
    B = A[:, :3] @ rng.standard_normal((3, r))
    selected = gcss(A, B, 3)
    assert np.all(selected == [0, 1, 2])

    # Now reverse the columns of the source matrix. Expect the GCSS answer to be the last 3.
    Arevcol = A[:, ::-1]
    selected = gcss(Arevcol, B, 3)
    assert np.all(selected == [n - 3, n - 2, n - 1])

    # All of the following inputs should fail assertions
    badarguments = ((A, B, -1), (A, B, n + 1), (A[:-2], B, 3), (A, B[:-2], 3))

    for a, b, ncol in badarguments:
        with pytest.raises(AssertionError):
            gcss(a, b, ncol)

    assert np.all(gcss(A, A, 4) == css(A, 4))
