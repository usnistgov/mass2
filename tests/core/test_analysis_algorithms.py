from mass2.core.analysis_algorithms import estimateRiseTime, resample_one_pulse, resample_pulses
import numpy as np
import pytest


def test_resample_pulses():
    "Test that resampling pulses works as intended."
    raw = np.arange(0, 300, 10)
    tests = (
        (0, raw),
        (1, np.hstack((raw[0], raw[:-1]))),
        (2, np.hstack((raw[0], raw[0], raw[:-2]))),
        (-1, np.hstack((raw[1:], raw[-1]))),
        (-2, np.hstack((raw[2:], raw[-1], raw[-1]))),
        (0.6, np.hstack((raw[0], 4 + raw[:-1]))),
        (1.2, np.hstack((raw[0], raw[0], 8 + raw[:-2]))),
        (1.5, np.hstack((raw[0], raw[0], 5 + raw[:-2]))),
        (1.8, np.hstack((raw[0], raw[0], 2 + raw[:-2]))),
        (-0.6, np.hstack((6 + raw[:-1], raw[-1]))),
        (-1.6, np.hstack((16 + raw[:-2], raw[-1], raw[-1]))),
        (-2.3, np.hstack((23 + raw[:-3], raw[-1], raw[-1], raw[-1]))),
    )
    for shift, expected in tests:
        y = raw.copy()
        assert np.all(resample_one_pulse(y, shift) == expected)

    # Now make sure the function works that does N pulses in one call.
    shifts = np.array([s for (s, _) in tests])
    pulses = np.vstack([raw.copy() for _ in range(len(shifts))])
    resample_pulses(pulses, shifts)
    for (_, expected), p in zip(tests, pulses):
        assert np.all(p == expected)


def test_resample_float_pulse_and_scalar_shift():
    "Float pulses are interpolated without rounding, and a scalar shift applies to every pulse."
    raw = np.linspace(0, 1, 6)
    x = np.arange(len(raw))
    for shift in (0.5, -0.5, 1.25, -2.75):
        assert np.allclose(resample_one_pulse(raw.copy(), shift), np.interp(x - shift, x, raw))
    pulses = resample_pulses(np.tile(raw, (3, 1)), 1.5)
    assert np.allclose(pulses, np.interp(x - 1.5, x, raw))


def test_estimateRiseTime_timebase_units():
    """A linear ramp over 20 samples rises in 20 samples, whatever the unit of timebase."""
    pulse = np.zeros(100)
    pulse[10:31] = np.linspace(0, 1000, 21)
    pulse[31:] = 1000 * np.exp(-np.arange(69) / 50)
    for timebase in [1e-6, 1.0, 2.0]:
        assert estimateRiseTime(pulse, timebase, 8)[0] == pytest.approx(20 * timebase)
