import numpy as np
import polars as pl

import mass2
import pytest
import logging

LOG = logging.getLogger("mass")

rng = np.random.default_rng(19324234)  # make tests not fail randomly


def make_arrival_times(cps, duration_s):
    """Return a distribution of arrival times where time differences are drawn from
    from an exponential distribution to reflect real x-ray arrival times. The last time
    will be less than duration_s, the first time will be zero.
    """
    nmult = 2
    while True:
        tdiffs = rng.exponential(1 / float(cps), int(np.ceil(nmult * cps * duration_s)))
        t = np.cumsum(np.hstack((0, tdiffs)))
        if t[-1] > duration_s:
            return t[t < duration_s]


def make_drifting_data(distrib, res_fwhm_ev, cps, duration_s, gain_of_t):
    """Return a tuple (t, energies) where t is an array of arrival times with realistic
    distribution with counts per second cps and duration duration_s, and energies
    is drawn from distribution, has added detector energy resoltion res_fwhm_ev,
    and the energies are warped by a gain equal to gain_of_t(t).
    """
    res_sigma = res_fwhm_ev / 2.3548
    t = make_arrival_times(cps, duration_s)
    energies0 = distrib.rvs(size=len(t), instrument_gaussian_fwhm=0, rng=rng)
    gain = gain_of_t(t)
    energies = gain * energies0 + rng.standard_normal(len(t)) * res_sigma
    return t, energies


class TestTimeDriftCorrection:
    @staticmethod
    def test_make_arrival_times():
        for cps in [0.1, 1, 10, 100]:
            for duration_s in [10, 100, 1000, 10000]:
                t = make_arrival_times(cps, duration_s)
                assert t[-1] < duration_s
                # t should have N=cps*duration_s entries with std deviation sqrt(N)
                # assert that it is within 10 stdevs
                Nexpected = cps * duration_s
                assert np.abs(len(t) - Nexpected) < 10 * np.sqrt(Nexpected)

    @staticmethod
    def gain_of_t(t):
        return 1 + 0.005 * np.sin(2 * np.pi * t / 10000.0)

    def test_make_drifting_data(self):
        distrib = mass2.calibration.MnKAlpha
        res_fwhm_ev = 3.0
        cps = 1
        duration_s = 10000

        _t, _energy = make_drifting_data(distrib, res_fwhm_ev, cps, duration_s, self.gain_of_t)

    def test_time_drift_correct(self):
        """Simulate MnKAlpha data with a sinusoidal gain drift and check that time_drift_correct removes most of it."""
        t, energy = make_drifting_data(mass2.calibration.MnKAlpha, 3.0, 1, 10000, self.gain_of_t)
        df = pl.DataFrame({"timestamp": pl.from_epoch(pl.Series(t * 1e6).cast(pl.Int64), "us"), "energy": energy})
        header = mass2.ChannelHeader("drifting", None, 1, 1e-5, 10, 100, pl.DataFrame())
        ch = mass2.Channel(df, header, len(df)).time_drift_correct("timestamp", "energy", "energy_tdc")
        raw = ch.linefit("MnKAlpha", "energy", dlo=100, dhi=100)
        corrected = ch.linefit("MnKAlpha", "energy_tdc", dlo=100, dhi=100)
        assert corrected.params["fwhm"].value < 0.15 * raw.params["fwhm"].value
        assert corrected.params["peak_ph"].value == pytest.approx(mass2.calibration.MnKAlpha.peak_energy, abs=1)


def test_time_drift_correct_scale_invariant():
    """The correction must not depend on the units of the data, even when the entropy is negative."""
    r = np.random.default_rng(5)
    t = np.sort(r.uniform(0, 40000, 5000))
    ph = (5898.8 + r.normal(0, 2, len(t))) * (1 + 0.003 * np.sin(t / 40000 * 3))
    corrections = []
    for scale in [1.0, 1e-3]:
        info = mass2.core.analysis_algorithms.time_drift_correct(t, ph * scale, w=0.5 * scale)
        corrections.append(info["model"](info["normalize"](t)))
    assert np.std(ph * (1 + corrections[0])) < 0.6 * np.std(ph)
    assert np.allclose(corrections[1], corrections[0], atol=1e-6)
