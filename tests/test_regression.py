import numpy as np
import polars as pl
import mass2
import pulsedata


def load_data():
    pairs = pulsedata.pulse_noise_ljh_pairs["bessy_20240727"]
    shorter_pulse_files = pairs.pulse_folder / ".." / "0001"
    data1 = mass2.Channels.from_ljh_folder(shorter_pulse_files, pairs.noise_folder)

    pairs = pulsedata.pulse_noise_ljh_pairs["regression"]
    data2 = mass2.Channels.from_ljh_folder(pairs.pulse_folder, pairs.noise_folder)
    return data1.with_more_channels(data2)


def all_steps(ch: mass2.Channel) -> mass2.Channel:
    use_dc = pl.lit(True)
    return (
        ch
        .summarize_pulses()
        .with_good_expr_pretrig_rms_and_postpeak_deriv(8, 8)
        .filter5lag(f_3db=10000)
        .driftcorrect(indicator_col="pretrig_mean", uncorrected_col="5lagy", use_expr=use_dc)
    )


def test_analysis_regression():
    """Make sure that analysis results don't change without us noticing"""
    data = load_data()
    data = data.map(all_steps)
    expected_df = pl.read_parquet("tests/regression_test_data.parquet")
    for ch_num, ch in data.channels.items():
        expect = expected_df.filter(pl.col("ch_num") == ch_num).drop("ch_num")
        found = ch.df.drop("pulse", "timestamp", "subframecount", strict=False)
        assert np.allclose(expect, found)


###########################################################################
# To update the regression_test_data.parquet file,
# cd into this directory, tests/ under the main mass2 directory. Then:
# python test_regression.py
###########################################################################


def store_analysis_regression():
    """Use this to replace the test data file"""
    data = load_data()
    data = data.map(all_steps)
    frames = []
    for ch_num, ch in data.channels.items():
        df = ch.df.drop("pulse", "timestamp", "subframecount", strict=False).with_columns(ch_num=pl.lit(ch_num))
        frames.append(df)
    df = pl.concat(frames)
    df.write_parquet("regression_test_data.parquet")


if __name__ == "__main__":
    store_analysis_regression()
