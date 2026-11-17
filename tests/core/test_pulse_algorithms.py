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


def test_pulse_2exp_with_tail_equal_taus():
    t = np.arange(200.0)
    args = dict(t0=20, a_tail=0, tau_tail=50, a=1000, tau_rise=10, baseline=0)
    limit = pulse_algorithms.pulse_2exp_with_tail(t, tau_fall_factor=1.0, **args)
    near = pulse_algorithms.pulse_2exp_with_tail(t, tau_fall_factor=1 + 1e-6, **args)
    assert np.amax(limit) == pytest.approx(1000)
    assert np.allclose(limit, near, atol=1e-2)


def test_pulse_2exp_with_tail_no_overflow():
    t = np.arange(1500.0)
    f = pulse_algorithms.pulse_2exp_with_tail
    v = f(t, t0=1000, a_tail=0, tau_tail=100, a=1000, tau_rise=1.0, tau_fall_factor=20, baseline=100)
    assert np.all(v[:1001] == 100)
    v = f(t, t0=1000, a_tail=5, tau_tail=1.0, a=1000, tau_rise=5.0, tau_fall_factor=20, baseline=100)
    assert np.all(np.isfinite(v))
    assert v[0] == pytest.approx(105)
    assert v[1] == pytest.approx(100 + 5 * np.exp(-1))


@pytest.mark.parametrize("offset, scale", [(0, 2.0), (-45000, 1), (-45000.0, 1.0), (0, 1e-3)])
def test_summarize_data_numba_signed_and_float(offset, scale):
    rng = np.random.default_rng(0)
    npre, nsamp = 50, 200
    t = np.arange(nsamp) - npre
    pulse = np.where(t > 0, 3000 * (np.exp(-t / 40) - np.exp(-t / 5)), 0)
    raw = (40000 + pulse + rng.normal(0, 5, (20, nsamp))).astype(np.uint16)
    x = raw.astype(np.int64) * scale + offset
    s = pulse_algorithms.summarize_data_numba(x, 1e-5, npre + 20, 0, npre)
    base = pulse_algorithms.summarize_data_numba(raw, 1e-5, npre + 20, 0, npre)
    assert np.allclose(s["min_value"], x.min(axis=1))
    assert np.array_equal(s["peak_index"], x.argmax(axis=1))
    baseline = x[:, :npre].mean(axis=1)
    if x.dtype.kind in "iu":
        baseline = np.floor(baseline + 0.5)
    assert np.allclose(s["peak_value"], x.max(axis=1) - baseline)
    assert np.allclose(s["rise_time"], base["rise_time"], rtol=1e-3)
    assert np.allclose(s["promptness"], base["promptness"], rtol=1e-3)
