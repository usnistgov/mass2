import numpy as np
import scipy as sp
import pytest

from mass2.core import Filter, FilterMaker, ToeplitzWhitener

rng = np.random.default_rng(7384)


def generate_autocorrelation(N=50):
    # Use autocorrelation to ensure we have a positive-definite test matrix
    x = rng.standard_normal(N)
    return np.correlate(x, x, mode="full")[N - 1 :]


def ATSF(pulse, npre, noise, sample_time_sec, peak=0.0, f_3db=None, cut_pre=0, cut_post=0):
    assert pulse.shape[1] == 2
    maker = FilterMaker(pulse[:, 0], npre, noise, dt_model=pulse[:, 1], sample_time_sec=sample_time_sec, peak=peak)
    return maker.compute_ats(f_3db=f_3db, cut_pre=cut_pre, cut_post=cut_post)


def test_ATSF():
    """Test the simple ATSF filter computation does not error, and is insensitive to DC."""
    nSamples = 100
    nPresamples = 50
    nPost = nSamples - nPresamples

    # Some fake data, fake noise, and a fake noise spectrum
    pulse_like = np.append(np.zeros(nPresamples), np.linspace(nPost - 1, 0, nPost))
    deriv_like = np.append(np.zeros(nPresamples), -np.ones(nPost))
    pulse = np.vstack((pulse_like, deriv_like)).T

    fake_noise_autocorr = generate_autocorrelation(nSamples)
    dt = 6.72e-6
    autocorr = np.zeros(nSamples)
    autocorr[0] = 1.0
    f = ATSF(pulse, nPresamples, autocorr, dt)
    assert np.isclose(f.values.sum(), 0)

    nPSD = 1 + (nSamples // 2)
    fPSD = np.linspace(0, 0.5, nPSD)
    PSD = 1 + 10 / (1 + (fPSD / 0.1) ** 2)

    maker_no_psd = FilterMaker(
        pulse_like, nPresamples, fake_noise_autocorr, dt_model=deriv_like, sample_time_sec=dt, peak=np.max(pulse_like)
    )
    with pytest.raises(ValueError):
        maker_no_psd.compute_fourier()  # impossible with no PSD

    maker = FilterMaker(
        pulse_like, nPresamples, fake_noise_autocorr, dt_model=deriv_like, noise_psd=PSD, sample_time_sec=dt, peak=np.max(pulse_like)
    )

    for computer in (maker.compute_ats, maker.compute_fourier):
        filter_to_test = computer(f_3db=None)
        assert isinstance(filter_to_test, Filter)
        assert np.isclose(filter_to_test.values.sum(), 0.0), f"{filter_to_test} failed DC test w/o lowpass"

        filter_to_test = computer(f_3db=1e4)
        assert isinstance(filter_to_test, Filter)
        assert np.isclose(filter_to_test.values.sum(), 0.0), f"{filter_to_test} failed DC test w/ f_3db"


def test_dc_insensitive():
    """When f_3db or fmax applied, filter should not become DC-sensitive.
    Tests for issue #176."""
    nSamples = 100
    nPresamples = 50
    nPost = nSamples - nPresamples

    # Some fake data, fake noise, and a fake noise spectrum
    pulse_like = np.append(np.zeros(nPresamples), np.linspace(nPost - 1, 0, nPost))
    deriv_like = np.append(np.zeros(nPresamples), -np.ones(nPost))

    fake_noise_autocorr = generate_autocorrelation(nSamples)
    dt = 6.72e-6

    nPSD = 1 + (nSamples // 2)
    fPSD = np.linspace(0, 0.5, nPSD)
    PSD = 1 + 10 / (1 + (fPSD / 0.1) ** 2)

    maker_no_psd = FilterMaker(
        pulse_like, nPresamples, fake_noise_autocorr, dt_model=deriv_like, sample_time_sec=dt, peak=np.max(pulse_like)
    )
    with pytest.raises(ValueError):
        maker_no_psd.compute_fourier()  # impossible with no PSD

    maker = FilterMaker(
        pulse_like, nPresamples, fake_noise_autocorr, dt_model=deriv_like, noise_psd=PSD, sample_time_sec=dt, peak=np.max(pulse_like)
    )

    def compute_5lag_noexp(f_3db=None, fmax=None):
        expmodel = np.exp(-np.linspace(0, 1, nSamples - 4))
        return maker.compute_constrained_5lag(expmodel, f_3db=f_3db, fmax=fmax)

    for computer in (maker.compute_ats, maker.compute_fourier, maker.compute_5lag, compute_5lag_noexp, maker.compute_1lag):
        filter_to_test = computer(f_3db=None, fmax=None)
        std = np.median(np.abs(filter_to_test.values))
        mean = filter_to_test.values.mean()
        assert np.abs(mean) < 1e-9 * std, f"{filter_to_test} failed DC test w/o lowpass"

        filter_to_test = computer(f_3db=1e4, fmax=None)
        mean = filter_to_test.values.mean()
        assert np.abs(mean) < 1e-9 * std, f"{filter_to_test} failed DC test w/ f_3db"

        filter_to_test = computer(f_3db=None, fmax=1e4)
        mean = filter_to_test.values.mean()
        assert np.abs(mean) < 1e-9 * std, f"{filter_to_test} failed DC test w/ fmax"


def test_constrained_filtering():  # noqa: PLR0914
    """Make sure that filters are insensitive to a given exponential when required (but not in general)
    and also general constraints"""
    nSamples = 100
    nPresamples = 50
    nPost = nSamples - nPresamples

    # Some fake data, fake noise, and a fake noise spectrum
    pulse_like = np.append(np.zeros(nPresamples), np.linspace(nPost - 1, 0, nPost))
    deriv_like = np.append(np.zeros(nPresamples), -np.ones(nPost))

    fake_noise_autocorr = generate_autocorrelation(nSamples)
    dt = 6.72e-6

    nPSD = 1 + (nSamples // 2)
    fPSD = np.linspace(0, 0.5, nPSD)
    PSD = 1 + 10 / (1 + (fPSD / 0.1) ** 2)

    maker_no_psd = FilterMaker(
        pulse_like, nPresamples, fake_noise_autocorr, dt_model=deriv_like, sample_time_sec=dt, peak=np.max(pulse_like)
    )
    with pytest.raises(ValueError):
        maker_no_psd.compute_fourier()  # impossible with no PSD

    maker = FilterMaker(
        pulse_like, nPresamples, fake_noise_autocorr, dt_model=deriv_like, noise_psd=PSD, sample_time_sec=dt, peak=np.max(pulse_like)
    )

    # Make a data vector with decaying exponental exp(-t/tau) and filters orthogonal to that
    tau = 0.001
    expdata = 100 * np.exp(-np.linspace(0, (nSamples - 1) * dt / tau, nSamples))
    expmodel = expdata[2:-2]
    f_usual = maker.compute_5lag()
    f_noexp = maker.compute_5lag_noexp(tau)
    f_constrained5 = maker.compute_constrained_5lag(expmodel)
    f_constrained1 = maker.compute_constrained_1lag(expdata)

    assert np.abs(f_usual.filter_records(expdata)[0]) > 1e-4, "compute_5lag is insensitive to an exponential"
    assert np.abs(f_noexp.filter_records(expdata)[0]) < 1e-8, "compute_5lag_noexp is sensitive to an exponential"
    assert np.abs(f_constrained5.filter_records(expdata)[0]) < 1e-8, "compute_constrained_5lag is sensitive to an exponential"
    assert np.abs(f_constrained1.filter_records(expdata)[0]) < 1e-8, "compute_constrained_1lag is sensitive to an exponential"

    # Now make multiple exponential constraints
    insensitive_models = [expdata, 30 - expdata**1.5, expdata**2]
    constraints = [m[2:-2] for m in insensitive_models]
    f_constrained = maker.compute_constrained_5lag(constraints)
    msg1 = "compute_5lag is unexpectedly insensitive to an arbitrary shape"
    msg2 = "compute_constrained_5lag is sensitive to constraint"
    assert np.all(np.abs(f_usual.filter_records(insensitive_models)[0]) > 1e-4), msg1
    assert np.all(np.abs(f_constrained.filter_records(insensitive_models)[0]) < 1e-7), msg2

    # And add a non-exponential constraint. This won't be strictly insensitive when we 5-lag filter it.
    # But it _will_ have zero inner product with the shortened-by-4 model. So test only that
    insensitive_models.append(200 * np.cos(np.linspace(0, 7, nSamples)))
    constraints.append(insensitive_models[-1][2:-2])
    f_constrained = maker.compute_constrained_5lag(constraints)
    for i, vec in enumerate(constraints):
        msg2 = f"compute_constrained_5lag filter values are not orthogonal to constraint # {i}"
        assert np.abs(f_constrained.values @ vec) < 1e-8, msg2


def test_no_concrete_baseFilter():
    "Make sure that mass2.core.Filter is an abstract base class, can't be instantiated directly."
    with pytest.raises(TypeError):
        _ = Filter(np.zeros(100), 100.0, 100.0, 100.0, 100.0, 0, None, None)  # type: ignore


class TestWhitener:
    """Test ToeplitzWhitener."""

    @staticmethod
    def test_trivial():
        """Be sure that the trivial whitener does nothing."""
        w = ToeplitzWhitener(np.array([1.0]), np.array([1.0]))  # the trivial whitener
        r = np.random.default_rng().standard_normal(100)
        assert np.allclose(r, w(r))
        assert np.allclose(r, w.solveW(r))
        assert np.allclose(r, w.applyWT(r))
        assert np.allclose(r, w.solveWT(r))

    @staticmethod
    def test_reversible():
        """Use a nontrivial whitener, and make sure that inverse operations are inverses."""
        w = ToeplitzWhitener(np.array([1.0, -1.7, 0.72]), np.array([1.0, 0.95]))
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
        w = ToeplitzWhitener(np.array([1.0, -1.7, 0.72]), np.array([1.0, 0.95]))
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


def test_VoverdV():
    Nsamp = 500
    Npulses = 2000
    white = rng.standard_normal((Nsamp, Npulses))
    t = np.arange(Nsamp)
    acorr = 10 * np.exp(-t / 50)
    acorr[0] *= 3
    R = sp.linalg.toeplitz(acorr)
    L = sp.linalg.cholesky(R, lower=True)
    noise = L @ white

    pulse = np.exp(-t[: Nsamp // 2] / 200) - np.exp(-t[: Nsamp // 2] / 20)
    pulse /= pulse.max()
    pulse = np.hstack((np.zeros(Nsamp - len(pulse)), pulse))

    # Filter using mass2 FilterMaker
    fm = FilterMaker(pulse, Nsamp // 2, acorr, None, peak=1000)
    filter5lag = fm.compute_5lag()
    assert np.abs(filter5lag.values.sum()) < 1e-12
    assert np.abs((filter5lag.values @ (1000 * pulse[2:-2])) - 1000) < 0.2
    noise_filt, phase = filter5lag.filter_records(noise.T + 1000 * pulse)
    predictedvar = filter5lag.variance
    assert (np.var(noise_filt, ddof=1) / predictedvar - 1) < 0.05

    # Filter outside the mass2 FilterMaker framework
    M = np.column_stack([pulse, np.ones_like(pulse)])
    Mtilde = np.linalg.lstsq(L, M)[0]
    filter1 = (np.linalg.pinv(Mtilde) @ np.linalg.inv(L))[0]
    peaks1 = filter1 @ noise + filter1 @ (pulse * 1000 + 5000)
    assert np.abs(peaks1.mean() - 1000) < 0.5
    assert (np.var(peaks1, ddof=1) / predictedvar - 1) < 0.05


def test_VoverdV_marcel():
    """Verify that we get roughly the same answer found in Marcel VandenBerg's memo
    "Optimal filtering in IGOR, version 3.0" dated 11 January 2005. He found V/dV = 4422 in IGOR.
    We require at least 4000 and no more than 4500, to be sure we aren't off by factors of
    2 or sqrt(8ln2) or some other large factor.
    """
    dt = 500e-9
    peak = 10
    Nsamp = 4096
    Npre = 255
    whitenoise = 0.01
    t = np.arange(-Npre, Nsamp - Npre) * dt
    pulse = peak * (np.exp(-t / 1e-4) - np.exp(-t / 1e-6))
    pulse[t < 0] = 0
    # pulse *= peak / pulse.max()
    autocorr = np.zeros_like(pulse)
    autocorr[0] = whitenoise**2
    fm = FilterMaker(pulse, Npre, autocorr, peak=peak, sample_time_sec=dt)
    filter1 = fm.compute_1lag()
    filter5 = fm.compute_5lag()
    filter5a = fm.compute_5lag(f_3db=10000.0)
    for f in (filter1, filter5, filter5a):
        vdv = f.predicted_v_over_dv
        assert vdv > 4000 and vdv < 4500
