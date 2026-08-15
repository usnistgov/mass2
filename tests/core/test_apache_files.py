import numpy as np
import polars as pl
import argparse

import mass2
import pulsedata


def test_conversion_to_arrow_and_loading(tmp_path):
    """Test that we can translate LJH files to IPC files and then read them as a Channels object."""
    # Load LJH example data, and convert to a set of arrow IPC files.
    pairs = pulsedata.pulse_noise_ljh_pairs["bessy_20240727"]
    shorter_pulse_files = pairs.pulse_folder / ".." / "0001"
    args = argparse.Namespace()
    args.dry_run = False
    args.base_dir = shorter_pulse_files.resolve()
    args.output = tmp_path
    args.mix = False
    mass2.core.apache_files.translate_ljh_files(args)

    # Load the IPC files as a data set, and make sure it can be summarized, and
    # that at least one pulse fails the good expression.
    data = mass2.Channels.from_ipc(tmp_path)

    def quick_analysis(ch: mass2.Channel) -> mass2.Channel:
        return ch.summarize_pulses().with_good_expr_pretrig_rms_and_postpeak_deriv()

    data = data.map(quick_analysis)
    assert "pretrig_rms" in data.ch0.df.columns
    s = data.ch0.good_series("pretrig_rms")
    assert np.all(s.to_numpy() > 1)
    assert len(s) < data.ch0.npulses


def test_conversion_arrow_parquet(tmp_path):
    """Test scripts that convert Arrow to Parquet and vice versa."""
    Npulse = 30
    Nsamp = 50
    rng = np.random.default_rng()
    pulse = rng.integers(0, 65535, size=(Npulse, Nsamp), dtype=np.uint16)
    timestamp = rng.integers(0, 1000000, size=Npulse).cumsum() + 1734567890
    subframecount = rng.integers(0, 100000, size=Npulse).cumsum()
    df = pl.DataFrame().with_columns(subframecount=subframecount, timestamp=timestamp, pulse=pulse)

    df.write_parquet(tmp_path / "data_parquet_chan1.parquet")
    df.write_ipc(tmp_path / "data_arrow_chan1.arrow")

    base = out = tmp_path
    mass2.core.apache_files.arrow_parquet(base, out, arrow2parquet=False)
    mass2.core.apache_files.arrow_parquet(base, out, arrow2parquet=True)

    for init_format in ("arrow", "parquet"):
        fname = tmp_path / f"data_{init_format}_chan1.arrow"
        print(fname)
        df2 = pl.read_ipc(fname)
        assert df.equals(df2)

        fname = tmp_path / f"data_{init_format}_chan1.parquet"
        df2 = pl.read_parquet(fname)
        assert df.equals(df2)
