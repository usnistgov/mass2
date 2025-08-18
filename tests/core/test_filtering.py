import numpy as np
import pytest

from mass2.core import Filter, FilterMaker, ToeplitzWhitener


def ATSF(pulse, npre, noise, sample_time_sec, peak=0.0, f_3db=None, cut_pre=0, cut_post=0):
    assert pulse.shape[1] == 2
    maker = FilterMaker(pulse[:, 0], npre, noise, dt_model=pulse[:, 1], sample_time_sec=sample_time_sec, peak=peak)
    return maker.compute_ats(f_3db=f_3db, cut_pre=cut_pre, cut_post=cut_post)


@pytest.mark.filterwarnings("ignore:invalid value encountered")
def test_dc_insensitive():
    """When f_3db or fmax applied, filter should not become DC-sensitive.
    Tests for issue #176."""
    nSamples = 100
    nPresamples = 50
    nPost = nSamples - nPresamples

    # Some fake data, fake noise, and a fake noise spectrum
    pulse_like = np.append(np.zeros(nPresamples), np.linspace(nPost - 1, 0, nPost))
    deriv_like = np.append(np.zeros(nPresamples), -np.ones(nPost))

    fake_noise = np.random.default_rng().standard_normal(nSamples)
    fake_noise[0] = 10.0
    dt = 6.72e-6

    nPSD = 1 + (nSamples // 2)
    fPSD = np.linspace(0, 0.5, nPSD)
    PSD = 1 + 10 / (1 + (fPSD / 0.1) ** 2)

    maker_no_psd = FilterMaker(pulse_like, nPresamples, fake_noise, dt_model=deriv_like, sample_time_sec=dt, peak=np.max(pulse_like))
    with pytest.raises(ValueError):
        maker_no_psd.compute_fourier()  # impossible with no PSD

    maker = FilterMaker(
        pulse_like, nPresamples, fake_noise, dt_model=deriv_like, noise_psd=PSD, sample_time_sec=dt, peak=np.max(pulse_like)
    )

    def compute_5lag_noexp(f_3db=None, fmax=None):
        expmodel = np.exp(-np.linspace(0, 1, nSamples - 4))
        return maker.compute_constrained_5lag(expmodel, f_3db=f_3db, fmax=fmax)

    for computer in (maker.compute_ats, maker.compute_fourier, maker.compute_5lag, compute_5lag_noexp):
        filter_to_test = computer(f_3db=None, fmax=None)
        std = np.median(np.abs(filter_to_test.values))
        mean = filter_to_test.values.mean()
        assert np.abs(mean) < 1e-10 * std, f"{filter_to_test} failed DC test w/o lowpass"

        filter_to_test = computer(f_3db=1e4, fmax=None)
        mean = filter_to_test.values.mean()
        assert np.abs(mean) < 1e-10 * std, f"{filter_to_test} failed DC test w/ f_3db"

        filter_to_test = computer(f_3db=None, fmax=1e4)
        mean = filter_to_test.values.mean()
        assert np.abs(mean) < 1e-10 * std, f"{filter_to_test} failed DC test w/ fmax"


def test_constrained_filtering():  # noqa: PLR0914
    """Make sure that filters are insensitive to a given exponential when required (but not in general)
    and also general constraints"""
    nSamples = 100
    nPresamples = 50
    nPost = nSamples - nPresamples

    # Some fake data, fake noise, and a fake noise spectrum
    pulse_like = np.append(np.zeros(nPresamples), np.linspace(nPost - 1, 0, nPost))
    deriv_like = np.append(np.zeros(nPresamples), -np.ones(nPost))

    fake_noise = np.random.default_rng().standard_normal(nSamples)
    fake_noise[0] = 10.0
    dt = 6.72e-6

    nPSD = 1 + (nSamples // 2)
    fPSD = np.linspace(0, 0.5, nPSD)
    PSD = 1 + 10 / (1 + (fPSD / 0.1) ** 2)

    maker_no_psd = FilterMaker(pulse_like, nPresamples, fake_noise, dt_model=deriv_like, sample_time_sec=dt, peak=np.max(pulse_like))
    with pytest.raises(ValueError):
        maker_no_psd.compute_fourier()  # impossible with no PSD

    maker = FilterMaker(
        pulse_like, nPresamples, fake_noise, dt_model=deriv_like, noise_psd=PSD, sample_time_sec=dt, peak=np.max(pulse_like)
    )

    # Make a data vector with decaying exponental exp(-t/tau) and filters orthogonal to that
    tau = 0.001
    expdata = np.exp(-np.linspace(0, (nSamples - 1) * dt / tau, nSamples))
    expmodel = expdata[2:-2]
    f_usual = maker.compute_5lag()
    f_noexp = maker.compute_5lag_noexp(tau)
    f_constrained = maker.compute_constrained_5lag(expmodel)

    assert np.abs(f_usual.filter_records(expdata)[0]) > 1e-4, "compute_5lag is insensitive to an exponential"
    assert np.abs(f_noexp.filter_records(expdata)[0]) < 1e-10, "compute_5lag_noexp is sensitive to an exponential"
    assert np.abs(f_constrained.filter_records(expdata)[0]) < 1e-10, "compute_constrained_5lag is sensitive to an exponential"

    # Now make multiple exponential constraints
    insensitive_models = [expdata, 1 - expdata**1.5, expdata**3.5]
    constraints = [m[2:-2] for m in insensitive_models]
    f_constrained = maker.compute_constrained_5lag(constraints)
    msg1 = "compute_5lag is unexpectedly insensitive to an arbitrary shape"
    msg2 = "compute_constrained_5lag is sensitive to constraint"
    assert np.all(np.abs(f_usual.filter_records(insensitive_models)[0]) > 1e-4), msg1
    assert np.all(np.abs(f_constrained.filter_records(insensitive_models)[0]) < 1e-10), msg2

    # And add a non-exponential constraint. This won't be strictly insensitive when we 5-lag filter it.
    # But it _will_ have zero inner product with the shortened-by-4 model. So test only that
    insensitive_models.append(np.cos(np.linspace(0, 7, nSamples)))
    constraints.append(insensitive_models[-1][2:-2])
    f_constrained = maker.compute_constrained_5lag(constraints)
    for i, vec in enumerate(constraints):
        msg2 = f"compute_constrained_5lag filter values are not orthogonal to constraint # {i}"
        assert np.abs(f_constrained.values.dot(vec)) < 1e-10, msg2


def test_no_concrete_baseFilter():
    "Make sure that mass2.core.Filter is an abstract base class, can't be instantiated directly."
    with pytest.raises(TypeError):
        _ = Filter(np.zeros(100), 100.0, 100.0, 100.0)


class TestWhitener:
    """Test ToeplitzWhitener."""

    @staticmethod
    def test_trivial():
        """Be sure that the trivial whitener does nothing."""
        w = ToeplitzWhitener([1.0], [1.0])  # the trivial whitener
        r = np.random.default_rng().standard_normal(100)
        assert np.allclose(r, w(r))
        assert np.allclose(r, w.solveW(r))
        assert np.allclose(r, w.applyWT(r))
        assert np.allclose(r, w.solveWT(r))

    @staticmethod
    def test_reversible():
        """Use a nontrivial whitener, and make sure that inverse operations are inverses."""
        w = ToeplitzWhitener([1.0, -1.7, 0.72], [1.0, 0.95])
        r = np.random.default_rng().standard_normal(100)
        assert np.allclose(r, w.solveW(w(r)))
        assert np.allclose(r, w(w.solveW(r)))
        assert np.allclose(r, w.solveWT(w.applyWT(r)))
        assert np.allclose(r, w.applyWT(w.solveWT(r)))

        # Also check that w isn't trivial
        assert not np.allclose(r, w(r))
        assert not np.allclose(r, w.solveW(r))
        assert not np.allclose(r, w.applyWT(r))
        assert not np.allclose(r, w.solveWT(r))

        # Check that no operations applied twice cancel out.
        assert not np.allclose(r, w(w(r)))
        assert not np.allclose(r, w.solveW(w.solveW(r)))
        assert not np.allclose(r, w.applyWT(w.applyWT(r)))
        assert not np.allclose(r, w.solveWT(w.solveWT(r)))

    @staticmethod
    def test_causal():
        """Make sure that the whitener and its inverse are causal,
        and that WT and its inverse anti-causal."""
        w = ToeplitzWhitener([1.0, -1.7, 0.72], [1.0, 0.95])
        Nzero = 100
        z = np.zeros(Nzero, dtype=float)
        r = np.hstack([z, np.random.default_rng().standard_normal(100), z])

        # Applying and solving W are causal operations.
        wr = w(r)
        wir = w.solveW(r)
        assert np.all(r[:Nzero] == 0)
        assert np.all(wr[:Nzero] == 0)
        assert np.all(wir[:Nzero] == 0)
        assert not np.all(wr[Nzero:] == 0)
        assert not np.all(wir[Nzero:] == 0)

        # Applying and solving WT are anti-causal operations.
        wtr = w.applyWT(r)
        wtir = w.solveWT(r)
        assert np.all(r[-Nzero:] == 0)
        assert np.all(wtr[-Nzero:] == 0)
        assert np.all(wtir[-Nzero:] == 0)
        assert not np.all(wtr[:-Nzero] == 0)
        assert not np.all(wtir[:-Nzero] == 0)
