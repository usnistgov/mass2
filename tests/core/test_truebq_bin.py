from mass2.core import truebq_bin
import numpy as np
from pathlib import Path


def test_basic(tmpdir):
    data_true = np.zeros(11_000)
    pulse_shape = np.exp(-np.arange(500) / 50)
    pulse_start_inds = np.array([50, 200, 300, 5000, 6000, 6233, 9950])
    for i in pulse_start_inds:
        data_true[i : i + len(pulse_shape)] += pulse_shape
    data_true = data_true[:10000]  # truncate to 10k samples
    data_true = (data_true * (2**13)).astype(np.int16)

    tmpfile = Path(tmpdir) / "2A" / "test.bin"
    tmpfile.parent.mkdir(parents=True, exist_ok=True)
    truebq_bin.write_truebq_bin_file(tmpfile, data_true, sample_rate_hz=1e5)
    assert (tmpfile).exists()
    assert (tmpfile).stat().st_size > 0
    read_truebq_bin = truebq_bin.TrueBqBin.load(tmpfile)
    assert read_truebq_bin.data.shape == (10000,)
    trigger_result = read_truebq_bin.trigger(filter_in=np.array([-1, 1]), threshold=1e-3, verbose=False)
    ch_mem = trigger_result.to_channel_copy_to_memory(noise_n_dead_samples_after_pulse_trigger=100, npre=100, npost=100, invert=False)
    ch_mmap = trigger_result.to_channel_mmap(noise_n_dead_samples_after_pulse_trigger=100, npre=100, npost=100, invert=False)
    # ch_summ = trigger_result.to_summarized_channel(noise_n_dead_samples_after_pulse_trigger=100,
    #                                                   npre=100, npost=100, invert=False)
    assert all(ch_mem.df["framecount"].to_numpy() == pulse_start_inds[1:-1])
    assert all(ch_mmap.df["framecount"].to_numpy() == pulse_start_inds[1:-1])
    # assert all(ch_summ.df["framecount"].to_numpy() == pulse_start_inds[1:-1])


def test_trigger_cache_key_separates_threshold_and_length(tmp_path):
    data = np.random.default_rng(3).normal(0, 4, 200_000).astype(np.int16)
    filt = np.r_[-np.ones(5), np.ones(5)] / 5
    bin_path = tmp_path / "x.bin"
    truebq_bin._fasttrig_filter_trigger_with_cache(data, filt, 2.0, 123456, bin_path, verbose=False)
    cached = truebq_bin._fasttrig_filter_trigger_with_cache(data, filt, 2.01, 23456, bin_path, verbose=False)
    direct = truebq_bin.fasttrig_filter_trigger(data[:23456].copy(), filt, 2.01, False)
    assert np.array_equal(cached, direct)


def test_fasttrig_filter_trigger_short_data():
    data = np.zeros(80, dtype=np.int16)
    data[10] = -50
    data[20] = 100
    inds = truebq_bin.fasttrig_filter_trigger(data, np.array([-1.0, 1.0]), 10.0, False)
    assert inds.tolist() == [11]


def test_truebq_bin_channel_number(tmp_path):
    expected = {"dev2_ai2": 2, "dev2_ai12": 12, "2A": 4, "3d": 11}
    for folder, channel_number in expected.items():
        bin_path = tmp_path / folder / "x.bin"
        truebq_bin.write_truebq_bin_file(bin_path, np.zeros(100, np.int16), sample_rate_hz=1e5)
        assert truebq_bin.TrueBqBin.load(bin_path).channel_number == channel_number
