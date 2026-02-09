import mass2
import numpy as np
import pytest
from pathlib import Path
import polars as pl

current_dir = Path(__file__).parent.resolve()


def channel_from_df(df, ch_num: int = 0, n_presamples: int = 1, n_samples: int = 2):
    header_df = pl.DataFrame()
    frametime_s = 1e-5
    header = mass2.ChannelHeader(
        description="from npy arrays",
        data_source=None,
        ch_num=ch_num,
        frametime_s=frametime_s,
        n_presamples=n_presamples,
        n_samples=n_samples,
        df=header_df,
    )
    ch = mass2.Channel(df, header, npulses=len(df), noise=None)
    return ch


def test_drift_correct(N=2e5, a0=100, b0=100, slope=0.01):
    N = int(N)
    rng = np.random.default_rng(42)
    a = rng.standard_normal(N) + a0
    b = rng.standard_normal(N) + b0
    dc_perfect = mass2.core.drift_correction.DriftCorrection(slope=-slope, offset=a0)
    b_tilted = dc_perfect(a, b)
    dc = mass2.core.drift_correct(indicator=a, uncorrected=b_tilted)
    assert dc.slope == pytest.approx(slope, abs=1e-3)
    assert dc.offset == pytest.approx(b0, abs=1e-2)


def test_bens_data():
    path = current_dir / "parquets" / "dc_test_from_bens_data_20260209.parquet"
    df = pl.read_parquet(path)
    df = df.with_columns(df.select(peak_value_f=pl.col("peak_value") * 1.0))
    dc = mass2.core.drift_correct(indicator=df["rel_sec"].to_numpy(), uncorrected=df["peak_value"].to_numpy())
    ch = channel_from_df(df)
    ch = ch.driftcorrect(indicator_col="rel_sec", uncorrected_col="peak_value", corrected_col="peak_value_dc")

    print(ch.df.limit(10))
    assert ch.df.std()["peak_value_dc"][0] < ch.df.std()["peak_value"][0]
