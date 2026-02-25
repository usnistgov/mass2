import os

import mass2
import pulsedata


def test_external_trigger_experiment_state():
    off_paths = mass2.core.ljhutil.find_ljh_files(pulsedata.off["ebit_20240723_0000"], ext=".off")
    assert len(off_paths) == 2

    data = mass2.Channels.from_off_paths(off_paths, "ebit_20240723_0000").with_experiment_state_by_path()

    # Check that the experiment states are in the order and number we expect
    series = data.channels[3].df["state_label"]
    labels = series.unique()
    counts = series.unique_counts()
    expect_labels = ("START", "IGNORE", "B", "C", "D", "E", "F", "G")
    expect_counts = (54669, 22957, 192, 398, 3790, 3947, 2284, 3853)
    for L, C, eL, eC in zip(labels, counts, expect_labels, expect_counts):
        assert L == eL
        assert C == eC

    # Now load and check external trigger file
    dir = os.path.dirname(off_paths[0])
    trigfile = os.path.join(dir, "20240723_run0000_external_trigger.bin")
    data = data.with_external_trigger_by_path(trigfile)
    ch = data.channels[3]

    sprev = ch.df["subframecount_prev_ext_trig"]
    sthis = ch.df["subframecount"]
    snext = ch.df["subframecount_next_ext_trig"]

    assert (sprev <= sthis).all()
    assert (sthis <= snext).all()
    assert sprev.unique().count() == 48750
    assert sthis.unique().count() == 92090
    assert snext.unique().count() == 48751


def test_external_trigger_LJH_issue101():
    heates_paths = pulsedata.pulse_noise_ljh_pairs["heates20240212"]
    data1 = mass2.Channels.from_ljh_folder(heates_paths.pulse_folder, heates_paths.noise_folder)
    trigpath = heates_paths.pulse_folder / "20240212_run0019_external_trigger.bin"
    data2 = data1.with_external_trigger_by_path(trigpath)
    for cnum, bc in data2.bad_channels.items():
        print(f"Chan {cnum:4d} bad: {bc.error_msg}")
    assert len(data2.bad_channels) == 0
