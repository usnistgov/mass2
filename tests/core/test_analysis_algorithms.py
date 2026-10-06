from mass2.core.analysis_algorithms import resample_one_pulse, resample_pulses
import numpy as np


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
