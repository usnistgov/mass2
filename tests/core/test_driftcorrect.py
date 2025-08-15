import mass2
import numpy as np
import pytest


def test_drift_correct(N=2e5, a0=100, b0=100, slope=0.01):
    N = int(N)
    rng = np.random.default_rng(42)
    a = rng.standard_normal(N) + a0
    b = rng.standard_normal(N) + b0
    dc_perfect = mass2.core.drift_correction.DriftCorrection(slope=-slope, offset=a0)
    b_tilted = dc_perfect(a, b)
    dc = mass2.core.drift_correct(indicator=a, uncorrected=b_tilted)
    assert dc.slope == pytest.approx(slope, abs=1e-3)
    assert dc.offset == pytest.approx(b0, abs=1e-2)
