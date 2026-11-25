"""
Pytest suite for ARNoiseWhitener in noise_algorithms.py.

Covers:
  1. Constructor & method input validation
  2. White noise modeling at fs = 250 kHz (PSD, ACF, reflection noise floor, whitening)
  3. ARMA(2,2) low-pass noise (2 kHz & 8 kHz poles at fs = 250 kHz):
     - Spectral & autocorrelation accuracy vs. analytical ground truth
     - Offset-invariant length-p whitening (1D and 2D column-vector batches)
     - GLS linear parameter recovery and predicted error-bar calibration
  4. Long-vector whitening (whiten_long in 'toeplitz' and 'exact_ar' modes)
  5. Digest contents and Matplotlib 4-panel plotting (plot_digest)
  6. Multi-row Parquet and Arrow IPC persistence (save, save_many, load, load_all)
"""

import matplotlib

matplotlib.use("Agg")  # non-interactive backend for headless test runs
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
import pytest
from scipy import linalg, signal

from mass2.core.noise_algorithms import ARNoiseWhitener


# ----------------------------------------------------------------------
# Noise simulation helpers & fixtures
# ----------------------------------------------------------------------

FS_HZ = 250_000.0  # 250 ksamples/s (>= 200,000 samples/s)
PULSE_LEN = 2048  # p >> q (q <= 256)
N_TRAIN_SEQS = 32
TRAIN_SEQ_LEN = 4 * PULSE_LEN


def simulate_white_noise(
    rng: np.random.Generator,
    n_seqs: int,
    seq_len: int,
    sigma: float = 2.5,
    add_dc_offsets: bool = True,
) -> np.ndarray:
    """Generate Gaussian white noise with optional large per-sequence DC offsets."""
    data = sigma * rng.standard_normal((n_seqs, seq_len))
    if add_dc_offsets:
        data += rng.uniform(-500.0, 500.0, size=(n_seqs, 1))
    return data


def simulate_arma22_lowpass_noise(  # ruff: ignore[too-many-positional-arguments]
    rng: np.random.Generator,
    n_seqs: int,
    seq_len: int,
    fs: float = FS_HZ,
    f_pole1_hz: float = 2_000.0,
    f_pole2_hz: float = 8_000.0,
    f_zero_hz: float = 30_000.0,
    r_zero: float = 0.85,
    sigma_w: float = 1.5,
    add_dc_offsets: bool = True,
) -> tuple[np.ndarray, list[float], list[float]]:
    """
    Generate stationary ARMA(2,2) noise with two real low-pass poles in 1-10 kHz
    and a pair of zeros, discarding startup transient burn-in.
    """
    p1 = float(np.exp(-2.0 * np.pi * f_pole1_hz / fs))
    p2 = float(np.exp(-2.0 * np.pi * f_pole2_hz / fs))
    omega_z = 2.0 * np.pi * f_zero_hz / fs

    b = [1.0, -2.0 * r_zero * float(np.cos(omega_z)), r_zero**2]
    a = [1.0, -(p1 + p2), p1 * p2]

    burn_in = 2048
    innovations = sigma_w * rng.standard_normal((n_seqs, seq_len + burn_in))
    filtered = signal.lfilter(b, a, innovations, axis=1)
    assert isinstance(filtered, np.ndarray)

    colored = filtered[:, burn_in:]
    if add_dc_offsets:
        colored += rng.uniform(-500.0, 500.0, size=(n_seqs, 1))
    return colored, b, a


@pytest.fixture(scope="module")
def white_noise_setup() -> dict[str, object]:
    rng = np.random.default_rng(101)
    sigma = 2.5
    train_seqs = simulate_white_noise(rng, N_TRAIN_SEQS, TRAIN_SEQ_LEN, sigma=sigma)
    model = ARNoiseWhitener(
        sequences=train_seqs,
        pulse_length=PULSE_LEN,
        max_order=256,
        sample_rate=FS_HZ,
        project_offset=True,
        calibrate=True,
        label="white_250kHz",
        metadata={"noise_type": "white", "sigma": sigma},
    )
    return {"model": model, "sigma": sigma, "fs": FS_HZ, "p": PULSE_LEN}


@pytest.fixture(scope="module")
def arma22_noise_setup() -> dict[str, object]:
    rng = np.random.default_rng(202)
    sigma_w = 1.5
    train_seqs, b, a = simulate_arma22_lowpass_noise(
        rng,
        n_seqs=N_TRAIN_SEQS,
        seq_len=TRAIN_SEQ_LEN,
        fs=FS_HZ,
        f_pole1_hz=2_000.0,
        f_pole2_hz=8_000.0,
        f_zero_hz=30_000.0,
        r_zero=0.85,
        sigma_w=sigma_w,
    )
    model = ARNoiseWhitener(
        sequences=train_seqs,
        pulse_length=PULSE_LEN,
        max_order=256,
        sample_rate=FS_HZ,
        project_offset=True,
        calibrate=True,
        label="arma22_lp_2k_8k",
        metadata={"noise_type": "arma22", "poles_hz": [2000.0, 8000.0]},
    )

    # Exact theoretical one-sided PSD & autocovariance for ground-truth comparisons
    freqs_true: NDArray[np.float64] = model.digest["freqs_ar"]
    n_psd_bins = 2 * (len(freqs_true) - 1)
    h_freq = np.fft.rfft(b, n=n_psd_bins) / np.fft.rfft(a, n=n_psd_bins)
    psd_true = (2.0 * sigma_w**2 / FS_HZ) * (np.abs(h_freq) ** 2)
    psd_true[0] *= 0.5
    psd_true[-1] *= 0.5

    imp_len = 16_384
    impulse = np.zeros(imp_len)
    impulse[0] = 1.0
    h_imp = signal.lfilter(b, a, impulse)
    assert isinstance(h_imp, np.ndarray)
    gamma_true = (sigma_w**2) * np.array([np.sum(h_imp[: imp_len - k] * h_imp[k:]) for k in range(PULSE_LEN)])
    acf_true = gamma_true / gamma_true[0]

    return {
        "model": model,
        "b": b,
        "a": a,
        "sigma_w": sigma_w,
        "fs": FS_HZ,
        "p": PULSE_LEN,
        "psd_true": psd_true,
        "gamma_true": gamma_true,
        "acf_true": acf_true,
    }


# ----------------------------------------------------------------------
# 1. Input validation & edge cases
# ----------------------------------------------------------------------


def test_input_validation() -> None:
    rng = np.random.default_rng(1)
    seqs = rng.standard_normal((4, 1024))

    with pytest.raises(ValueError, match="2D array"):
        ARNoiseWhitener(rng.standard_normal((2, 4, 1024)), pulse_length=512)

    with pytest.raises(ValueError, match="at least 16"):
        ARNoiseWhitener(seqs, pulse_length=8, max_order=4)

    with pytest.raises(ValueError, match="length >= pulse_length"):
        ARNoiseWhitener(seqs, pulse_length=2048)

    with pytest.raises(ValueError, match="between 1 and 256"):
        ARNoiseWhitener(seqs, pulse_length=512, max_order=300)

    with pytest.raises(ValueError, match="greater than `max_order`"):
        ARNoiseWhitener(seqs, pulse_length=256, max_order=256)

    with pytest.raises(ValueError, match="zero variance"):
        ARNoiseWhitener(np.ones((4, 1024)), pulse_length=512, max_order=32)


def test_1d_single_sequence_and_uncalibrated_mode() -> None:
    rng = np.random.default_rng(2)
    single_seq = rng.standard_normal(4096)
    model = ARNoiseWhitener(
        sequences=single_seq,
        pulse_length=1024,
        max_order=64,
        sample_rate=FS_HZ,
        project_offset=False,
        calibrate=True,  # falls back cleanly when n_sequences == 1
    )
    assert model.n_sequences == 1
    assert model.calibration_alpha == 1.0
    assert model.w_hat is None
    z = model.whiten(rng.standard_normal(1024))
    assert z.shape == (1024,)
    assert np.isclose(np.var(z), 1.0, rtol=0.12)


# ----------------------------------------------------------------------
# 2. White noise tests (fs = 250 kHz)
# ----------------------------------------------------------------------


def test_white_noise_model_and_digest(white_noise_setup: dict[str, object]) -> None:
    model: ARNoiseWhitener = white_noise_setup["model"]  # type: ignore[assignment]
    sigma: float = white_noise_setup["sigma"]  # type: ignore[assignment]
    fs: float = white_noise_setup["fs"]  # type: ignore[assignment]
    d = model.digest

    # Reflection coefficients should fluctuate around the statistical noise floor 1/sqrt(N_eff)
    rms_kappa = float(np.sqrt(np.mean(model.reflection_coeffs[1:] ** 2)))
    assert rms_kappa < 1.5 * model.noise_floor_kappa

    # Total RMS and innovation STD should both recover sigma = 2.5
    assert np.isclose(d["rms_noise"], sigma, rtol=0.03)
    assert np.isclose(d["innovation_std"], sigma, rtol=0.03)

    # One-sided PSD includes an exact 3.01 dB (2x) step between DC/Nyquist and interior bins
    assert d["spectral_dynamic_range_db"] < 4.5
    interior_dyn_range_db = float(10.0 * np.log10(np.max(d["psd_ar"][1:-1]) / np.min(d["psd_ar"][1:-1])))
    assert interior_dyn_range_db < 1.5

    # One-sided white PSD should match 2 * sigma^2 / fs across interior bins
    expected_psd = 2.0 * sigma**2 / fs
    assert np.isclose(np.mean(d["psd_ar"][1:-1]), expected_psd, rtol=0.05)
    assert np.isclose(np.median(d["psd_welch"][1:-1]), expected_psd, rtol=0.08)

    # Autocorrelation should be 1 at lag 0 and near 0 for all lags >= 1
    assert d["autocorr"][0] == 1.0
    assert np.max(np.abs(d["autocorr"][1:])) < 0.05


def test_white_noise_whitening_and_offset_invariance(white_noise_setup: dict[str, object]) -> None:
    model: ARNoiseWhitener = white_noise_setup["model"]  # type: ignore[assignment]
    sigma: float = white_noise_setup["sigma"]  # type: ignore[assignment]
    p: int = white_noise_setup["p"]  # type: ignore[assignment]

    rng = np.random.default_rng(303)
    # Column-oriented batch of shape (p, 300)
    test_cols = simulate_white_noise(rng, n_seqs=300, seq_len=p, sigma=sigma, add_dc_offsets=True).T
    Z = model.whiten(test_cols)

    assert Z.shape == test_cols.shape
    # Unit variance in the (p - 1)-dimensional offset-free subspace
    mean_energy = float(np.mean(np.sum(Z**2, axis=0)))
    assert np.isclose(mean_energy, p - 1, rtol=0.04)

    # Strict invariance to huge constant offsets (float64 roundoff on ~500 DC leaves ~5e-11)
    Z_shifted = model.whiten(test_cols + 10_000.0)
    assert np.allclose(Z, Z_shifted, atol=1e-10)
    assert model.w_hat is not None
    assert np.max(np.abs(model.w_hat @ Z)) < 1e-9


# ----------------------------------------------------------------------
# 3. ARMA(2,2) low-pass noise tests (2 kHz & 8 kHz poles, fs = 250 kHz)
# ----------------------------------------------------------------------


def test_arma22_spectrum_and_autocorrelation(arma22_noise_setup: dict[str, object]) -> None:
    model: ARNoiseWhitener = arma22_noise_setup["model"]  # type: ignore[assignment]
    sigma_w: float = arma22_noise_setup["sigma_w"]  # type: ignore[assignment]
    psd_true: np.ndarray = arma22_noise_setup["psd_true"]  # type: ignore[assignment]
    gamma_true: np.ndarray = arma22_noise_setup["gamma_true"]  # type: ignore[assignment]
    acf_true: np.ndarray = arma22_noise_setup["acf_true"]  # type: ignore[assignment]
    d = model.digest

    # AR(q) needs higher order than 2 to approximate the 30 kHz MA zero pair, while staying < max_order
    assert 10 <= model.selected_order <= model.max_order
    assert np.isclose(d["innovation_std"], sigma_w, rtol=0.03)
    assert np.isclose(d["rms_noise"], np.sqrt(gamma_true[0]), rtol=0.08)

    # Log-spectral error between AR(q) PSD and exact ARMA(2,2) PSD across [500 Hz, 100 kHz]
    freqs = d["freqs_ar"]
    band = (freqs >= 500.0) & (freqs <= 100_000.0)
    log_err_db = 10.0 * np.abs(np.log10(d["psd_ar"][band] / psd_true[band]))
    assert np.mean(log_err_db) < 0.8  # < 0.8 dB mean spectral error across 40+ dB range

    # Autocorrelation out to length p should track theoretical ARMA(2,2) ACF
    assert np.max(np.abs(d["autocorr"] - acf_true)) < 0.04


def test_arma22_whitening_and_parameter_uncertainty_calibration(  # ruff: ignore[too-many-locals]
    arma22_noise_setup: dict[str, object],
) -> None:
    model: ARNoiseWhitener = arma22_noise_setup["model"]  # type: ignore[assignment]
    sigma_w: float = arma22_noise_setup["sigma_w"]  # type: ignore[assignment]
    fs: float = arma22_noise_setup["fs"]  # type: ignore[assignment]
    p: int = arma22_noise_setup["p"]  # type: ignore[assignment]

    rng = np.random.default_rng(404)
    n_trials = 400
    noise_pulses, _, _ = simulate_arma22_lowpass_noise(rng, n_seqs=n_trials, seq_len=p, fs=fs, sigma_w=sigma_w, add_dc_offsets=True)
    noise_cols = noise_pulses.T  # shape (p, n_trials)

    # 1. Whitened noise energy & residual autocorrelation check
    Z = model.whiten(noise_cols)
    assert Z.shape == (p, n_trials)
    assert np.isclose(np.mean(np.sum(Z**2, axis=0)), p - 1, rtol=0.05)
    assert np.allclose(Z, model.whiten(noise_cols + 25_000.0), atol=1e-10)

    # Out-of-sample whitened residual ACF from digest should be near zero
    w_acf = model.digest["whitened_acf"][1:]
    assert np.sqrt(np.mean(w_acf**2)) < 0.03

    # 2. Inject a microcalorimeter-like pulse (10 us rise, 200 us decay -> energy in 1-10 kHz band)
    t = np.arange(p) / fs
    template = np.exp(-t / 200e-6) - np.exp(-t / 10e-6)
    X = template[:, None]

    cov_pred = model.predict_parameter_cov(X)
    se_pred = float(np.sqrt(cov_pred[0, 0]))

    true_amp = 4.5
    noisy_records = noise_pulses + true_amp * template[None, :]
    estimates = np.empty(n_trials)
    for i in range(n_trials):
        beta_hat, cov_i = model.fit_linear_model(noisy_records[i], X)
        estimates[i] = beta_hat[0]
        assert np.allclose(cov_i, cov_pred)

    se_empirical = float(np.std(estimates, ddof=1))
    # Unbiased amplitude estimate and calibrated error bar prediction
    assert np.isclose(np.mean(estimates), true_amp, atol=3.5 * se_pred / np.sqrt(n_trials))
    assert np.isclose(se_pred, se_empirical, rtol=0.12)


def test_shape_mismatch_errors(arma22_noise_setup: dict[str, object]) -> None:
    model: ARNoiseWhitener = arma22_noise_setup["model"]  # type: ignore[assignment]
    p: int = arma22_noise_setup["p"]  # type: ignore[assignment]

    with pytest.raises(ValueError, match="length p="):
        model.whiten(np.zeros(p + 10))

    # Row-oriented 2D array (n_vectors, p) must be rejected when n_vectors != p
    with pytest.raises(ValueError, match="column vectors"):
        model.whiten(np.zeros((5, p)))

    with pytest.raises(ValueError, match="length >= p="):
        model.whiten_long(np.zeros(p - 1))

    with pytest.raises(ValueError, match="Design matrix X"):
        model.fit_linear_model(np.zeros(p), np.zeros((p + 1, 2)))


def test_optimal_fitter_matches_explicit_pinv_and_gls(  # ruff: ignore[too-many-locals]
    arma22_noise_setup: dict[str, object],
) -> None:
    model: ARNoiseWhitener = arma22_noise_setup["model"]  # type: ignore[assignment]
    sigma_w: float = arma22_noise_setup["sigma_w"]  # type: ignore[assignment]
    fs: float = arma22_noise_setup["fs"]  # type: ignore[assignment]
    p: int = arma22_noise_setup["p"]  # type: ignore[assignment]

    # 3-component linear model matrix M of shape (p, 3)
    t = np.arange(p) / fs
    dt = np.maximum(0.0, t - 200 / fs)
    active = (t >= 200 / fs).astype(np.float64)
    comp1 = active * (np.exp(-dt / 200e-6) - np.exp(-dt / 12e-6))
    comp2 = active * (np.exp(-dt / 75e-6) - np.exp(-dt / 12e-6))
    comp3 = np.sin(2.0 * np.pi * 6_000.0 * t)
    M = np.column_stack([comp1, comp2, comp3])
    n_comp = M.shape[1]

    F = model.optimal_fitter(M)
    assert F.shape == (n_comp, p)

    # 1. Exact agreement with explicit p x p matrix product pinv(W @ M) @ W
    W_explicit = model.whiten(np.eye(p))
    F_explicit = np.linalg.pinv(W_explicit @ M) @ W_explicit
    assert np.allclose(F, F_explicit, atol=1e-12)

    # 2. Left-inverse property on noiseless signals and annihilation of constant offsets
    assert np.allclose(F @ M, np.eye(n_comp), atol=1e-11)
    assert np.allclose(F @ np.ones(p), np.zeros(n_comp), atol=1e-11)

    # 3. Exact agreement with fit_linear_model on 1D pulse and 2D batch of column pulses
    rng = np.random.default_rng(707)
    noise_batch, _, _ = simulate_arma22_lowpass_noise(rng, n_seqs=25, seq_len=p, fs=fs, sigma_w=sigma_w, add_dc_offsets=True)
    beta_true = np.array([3.2, -1.1, 0.45])
    Y_cols = noise_batch.T + (M @ beta_true)[:, None]  # shape (p, 25)

    betas_batch = F @ Y_cols
    assert betas_batch.shape == (n_comp, 25)
    assert np.allclose(F @ Y_cols[:, 0], betas_batch[:, 0])
    assert np.allclose(F @ (Y_cols + 15_000.0), betas_batch, atol=1e-9)

    for i in range(5):
        beta_gls, _ = model.fit_linear_model(Y_cols[:, i], M)
        assert np.allclose(betas_batch[:, i], beta_gls, atol=1e-11)

    # 4. Covariance propagation F @ Gamma_cal @ F.T == predict_parameter_cov(M)
    Gamma_cal = linalg.toeplitz(model.digest["autocov"])
    cov_propagated = F @ Gamma_cal @ F.T
    cov_predicted = model.predict_parameter_cov(M)
    assert np.allclose(cov_propagated, cov_predicted, atol=1e-11)


def test_optimal_fitter_rank_deficient_and_no_offset_projection() -> None:
    rng = np.random.default_rng(808)
    p = 512
    fs = 250_000.0
    seqs, _, _ = simulate_arma22_lowpass_noise(rng, n_seqs=12, seq_len=4 * p, fs=fs)

    # Without offset projection: verify exact match vs explicit pinv(W @ M) @ W
    model_no_proj = ARNoiseWhitener(sequences=seqs, pulse_length=p, max_order=64, sample_rate=fs, project_offset=False)
    t = np.arange(p) / fs
    M_with_dc = np.column_stack([np.exp(-t / 100e-6), np.ones(p)])
    F_no_proj = model_no_proj.optimal_fitter(M_with_dc)
    W_no_proj = model_no_proj.whiten(np.eye(p))
    assert np.allclose(F_no_proj, np.linalg.pinv(W_no_proj @ M_with_dc) @ W_no_proj, atol=1e-12)
    assert np.allclose(F_no_proj @ M_with_dc, np.eye(2), atol=1e-11)

    # With offset projection: constant column is in nullspace of W, pseudoinverse handles it cleanly
    model_proj = ARNoiseWhitener(sequences=seqs, pulse_length=p, max_order=64, sample_rate=fs, project_offset=True)
    F_proj = model_proj.optimal_fitter(M_with_dc)
    assert F_proj.shape == (2, p)
    assert np.allclose(F_proj[1], np.zeros(p), atol=1e-11)
    assert np.isclose(F_proj[0] @ M_with_dc[:, 0], 1.0, atol=1e-11)

    # Input shape validation
    with pytest.raises(ValueError, match="Model matrix M"):
        model_proj.optimal_fitter(np.ones(p))
    with pytest.raises(ValueError, match="Model matrix M"):
        model_proj.optimal_fitter(np.ones((p + 1, 2)))
    with pytest.raises(ValueError, match="Model matrix M"):
        model_proj.optimal_fitter(np.empty((p, 0)))


# ----------------------------------------------------------------------
# 4. Long-vector whitening (length > p, Toeplitz & exact AR modes)
# ----------------------------------------------------------------------


def test_whiten_long_toeplitz_and_exact_ar(arma22_noise_setup: dict[str, object]) -> None:
    model: ARNoiseWhitener = arma22_noise_setup["model"]  # type: ignore[assignment]
    sigma_w: float = arma22_noise_setup["sigma_w"]  # type: ignore[assignment]
    fs: float = arma22_noise_setup["fs"]  # type: ignore[assignment]
    p: int = arma22_noise_setup["p"]  # type: ignore[assignment]
    q = model.selected_order

    rng = np.random.default_rng(505)
    long_len = 3 * p
    long_batch, _, _ = simulate_arma22_lowpass_noise(rng, n_seqs=60, seq_len=long_len, fs=fs, sigma_w=sigma_w, add_dc_offsets=True)

    # Toeplitz FIR mode (1D and 2D)
    z_toep_1d = model.whiten_long(long_batch[0], mode="toeplitz", anchor_first=True)
    z_toep_2d = model.whiten_long(long_batch, mode="toeplitz", anchor_first=True)
    assert z_toep_1d.shape == (long_len,)
    assert z_toep_2d.shape == (60, long_len)
    assert np.allclose(z_toep_1d, z_toep_2d[0])

    # Verify steady-state variance (t >= q) is ~1 and matches direct lfilter with toeplitz_fir_taps
    assert np.isclose(np.var(z_toep_2d[:, q:]), 1.0, rtol=0.06)
    manual_lfilter = signal.lfilter(model.toeplitz_fir_taps, [1.0], long_batch[0] - long_batch[0, 0])
    assert isinstance(manual_lfilter, np.ndarray)
    assert np.allclose(z_toep_1d, manual_lfilter)

    # Constant offset invariance with anchor_first=True
    z_toep_shifted = model.whiten_long(long_batch + 50_000.0, mode="toeplitz", anchor_first=True)
    assert np.allclose(z_toep_2d, z_toep_shifted, atol=1e-9)

    # Exact AR mode on length > p
    z_exact = model.whiten_long(long_batch, mode="exact_ar")
    assert z_exact.shape == (60, long_len)
    assert np.isclose(np.mean(np.sum(z_exact**2, axis=1)), long_len - 1, rtol=0.05)

    with pytest.raises(ValueError, match="Unknown mode"):
        model.whiten_long(long_batch[0], mode="invalid")  # type: ignore[arg-type]


# ----------------------------------------------------------------------
# 5. Digest & Matplotlib plotting
# ----------------------------------------------------------------------


def test_digest_and_plot_digest(arma22_noise_setup: dict[str, object]) -> None:
    model: ARNoiseWhitener = arma22_noise_setup["model"]  # type: ignore[assignment]
    p: int = arma22_noise_setup["p"]  # type: ignore[assignment]
    d = model.digest

    expected_keys = {
        "selected_order",
        "max_order",
        "pulse_length",
        "sample_rate",
        "calibration_alpha",
        "rms_noise",
        "innovation_std",
        "spectral_dynamic_range_db",
        "noise_floor_kappa",
        "lags",
        "autocov",
        "autocorr",
        "freqs_ar",
        "psd_ar",
        "freqs_welch",
        "psd_welch",
        "whitened_acf",
        "whitened_acf_bound",
    }
    assert expected_keys.issubset(d.keys())
    assert len(d["autocorr"]) == p
    assert np.all(np.isfinite(d["psd_ar"]))
    assert np.all(np.isfinite(d["psd_welch"]))

    fig, axes = model.plot_digest(max_acf_lag=512)
    try:
        assert isinstance(fig, matplotlib.figure.Figure)
        assert axes.shape == (2, 2)
        for ax in axes.ravel():
            assert len(ax.lines) >= 1
    finally:
        plt.close(fig)


# ----------------------------------------------------------------------
# 6. Parquet & Arrow IPC multi-row serialization
# ----------------------------------------------------------------------


@pytest.mark.parametrize("fmt,ext", [("parquet", ".parquet"), ("ipc", ".arrow")])
def test_multi_row_persistence_roundtrip(
    tmp_path,
    white_noise_setup: dict[str, object],
    arma22_noise_setup: dict[str, object],
    fmt: str,
    ext: str,
) -> None:
    m_white: ARNoiseWhitener = white_noise_setup["model"]  # type: ignore[assignment]
    m_arma: ARNoiseWhitener = arma22_noise_setup["model"]  # type: ignore[assignment]
    p: int = arma22_noise_setup["p"]  # type: ignore[assignment]

    archive_path = tmp_path / f"noise_models{ext}"

    # Append rows sequentially
    m_white.save(archive_path, append=True)
    m_arma.save(archive_path, append=True)

    # Load all rows and verify count & ordering
    loaded_all = ARNoiseWhitener.load_all(archive_path)
    assert len(loaded_all) == 2
    assert loaded_all[0].model_id == m_white.model_id
    assert loaded_all[1].model_id == m_arma.model_id

    # Load by label, model_id, and negative row index
    rec_by_label = ARNoiseWhitener.load(archive_path, label="white_250kHz")
    rec_by_id = ARNoiseWhitener.load(archive_path, model_id=m_arma.model_id)
    rec_last = ARNoiseWhitener.load(archive_path, row=-1)

    assert rec_by_label.metadata == m_white.metadata
    assert rec_by_id.model_id == m_arma.model_id
    assert rec_last.model_id == m_arma.model_id

    # Verify exact numerical equivalence after round-trip
    rng = np.random.default_rng(606)
    test_pulse = rng.standard_normal(p) + 123.0
    test_cols = rng.standard_normal((p, 5)) + 123.0
    test_long = rng.standard_normal(2 * p) + 123.0
    X = np.ones((p, 1))
    X[:, 0] = np.sin(2.0 * np.pi * np.arange(p) / p)

    assert np.allclose(m_arma.whiten(test_pulse), rec_by_id.whiten(test_pulse))
    assert np.allclose(m_arma.whiten(test_cols), rec_by_id.whiten(test_cols))
    assert np.allclose(m_arma.whiten_long(test_long), rec_by_id.whiten_long(test_long))
    assert np.allclose(m_arma.predict_parameter_cov(X), rec_by_id.predict_parameter_cov(X))
    assert np.allclose(m_arma.digest["psd_ar"], rec_by_id.digest["psd_ar"])
    assert np.allclose(m_arma.digest["autocorr"], rec_by_id.digest["autocorr"])

    # Test batch save_many overwrite + missing key/index errors
    batch_path = tmp_path / f"batch_models{ext}"
    ARNoiseWhitener.save_many([m_white, m_arma], batch_path, append=False)
    assert len(ARNoiseWhitener.load_all(batch_path)) == 2

    with pytest.raises(KeyError):
        ARNoiseWhitener.load(batch_path, model_id="nonexistent-id")
    with pytest.raises(KeyError):
        ARNoiseWhitener.load(batch_path, label="nonexistent-label")
    with pytest.raises(IndexError):
        ARNoiseWhitener.load(batch_path, row=99)
