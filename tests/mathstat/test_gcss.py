import pytest
import numpy as np
from mass2.mathstat.gcss import gcss, css

rng = np.random.default_rng(seed=1432)


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


def test_gcss_iterative():
    "We know that this will require iterative calling of gcss from itself. Make sure it works"
    m = 100
    n = 30
    r = 100
    t = np.arange(m)
    A = np.vstack([np.exp(-t * rate) for rate in np.linspace(0, 10 / m, n)]).T

    pulse1 = t * np.exp(-t * 5 / m)
    B = np.outer(pulse1, rng.uniform(100, 1000, size=r)) + rng.standard_normal((m, r)) * 10

    for ncol in (2, 5, 8, 12, 16, 20, 24):
        columns = gcss(A, B, ncol)
        assert len(columns) == ncol
        assert columns.min() >= 0
        assert columns.max() < n
