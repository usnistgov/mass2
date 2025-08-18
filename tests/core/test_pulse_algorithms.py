import numpy as np
import pytest
from mass2.core import pulse_algorithms


def test_fit_pulse_2exp_with_tail():
    # Test parameters
    t0 = 50
    a_tail = 5.0
    tau_tail = 100.0
    a = 1000.0
    tau_rise = 10.0
    tau_fall_factor = 2
    baseline = -32.7

    # Create a time array
    t = np.arange(100)

    # Calculate the expected values using the function
    expected_values = pulse_algorithms.pulse_2exp_with_tail(t, t0, a_tail, tau_tail, a, tau_rise, tau_fall_factor, baseline)

    assert expected_values[0] == pytest.approx(a_tail + baseline, rel=1e-2)
    assert np.amax(expected_values) == pytest.approx(a + baseline, rel=1e-2)

    result = pulse_algorithms.fit_pulse_2exp_with_tail(expected_values, npre=50, dt=1)

    assert result.params["t0"].value == pytest.approx(t0, rel=1e-2)
    assert result.params["a_tail"].value == pytest.approx(a_tail, rel=1e-2)
    assert result.params["tau_tail"].value == pytest.approx(tau_tail, rel=1e-2)
    assert result.params["a"].value == pytest.approx(a, rel=1e-2)
    assert result.params["tau_rise"].value == pytest.approx(tau_rise, rel=1e-2)
    assert result.params["tau_fall_factor"].value == pytest.approx(tau_fall_factor, rel=1e-2)
    assert result.params["baseline"].value == pytest.approx(baseline, rel=1e-2)
