from mass2.core import truebq_bin
import numpy as np
from tempfile import TemporaryDirectory
from pathlib import Path


def test_basic():
    data_true = np.zeros(11_000)
    pulse_shape = np.exp(-np.arange(500) / 50)
    pulse_start_inds = np.array([50, 200, 300, 5000, 6000, 6233, 9950])
    for i in pulse_start_inds:
        data_true[i : i + len(pulse_shape)] += pulse_shape
    data_true = data_true[:10000]  # truncate to 10k samples
    data_true = (data_true * (2**13)).astype(np.int16)

    with TemporaryDirectory(delete=False) as tmpdir:
        tmpfile = Path(tmpdir) / "2A" / "test.bin"
        tmpdir.mkdir(parents=True, exist_ok=False)
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
