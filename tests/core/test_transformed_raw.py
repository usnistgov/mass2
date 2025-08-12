import numpy as np
import polars as pl
from polars.testing import assert_frame_equal
import pulsedata
import mass2


def test_inverted_data():
    """Read the same file normally and inverted. Be sure that all means of
    accssing the raw data give bitwise inverses of each other."""
    src_name = pulsedata.pulse_noise_ljh_pairs["bessy_20240727"].noise_folder / "20240727_run0000_chan4219.ljh"

    def invert(raw):
        "Perform bitwise inversion of the `raw` array"
        return ~raw

    ds1 = mass2.Channel.from_ljh(src_name)
    ds2 = mass2.Channel.from_ljh(src_name, transform_raw=invert)
    raw1 = ds1.df["pulse"].to_numpy()
    raw2 = ds2.df["pulse"].to_numpy()
    tr2 = ds2.transform_raw(raw2)

    assert np.all(raw1 == raw2)
    assert np.all(raw1 == ~tr2)

    src_name = pulsedata.pulse_noise_ljh_pairs["bessy_20240727"].pulse_folder / "20240727_run0002_chan4219.ljh"

    ds1 = mass2.Channel.from_ljh(src_name)
    ds2 = mass2.Channel.from_ljh(src_name, transform_raw=invert)

    # Use only a limited # of rows for this test
    nmax = 400
    ds1 = ds1.with_replacement_df(ds1.df.limit(nmax))
    ds2 = ds2.with_replacement_df(ds2.df.limit(nmax))

    # Now replace ds2's raw data with a bitwise inverse of it
    inverted = invert(ds2.df["pulse"].to_numpy())
    df2 = ds2.df.drop("pulse").with_columns(pl.Series(inverted).alias("pulse"))
    ds2 = ds2.with_replacement_df(df2)

    # Make sure the pulse data are indeed inverted
    raw1 = ds1.df["pulse"].to_numpy()
    raw2 = ds2.df["pulse"].to_numpy()
    tr2 = ds2.transform_raw(raw2)
    assert np.all(raw1 == ~raw2)
    assert np.all(raw1 == tr2)

    # Now remove the raw pulse column and compare all other columns in the data frame
    ds1 = ds1.summarize_pulses()
    ds2 = ds2.summarize_pulses()
    df1 = ds1.df.drop("pulse")
    df2 = ds2.df.drop("pulse")
    print(df1.limit(5))
    print(df2.limit(5))
    assert_frame_equal(df1, df2, check_exact=False)
