import pytest
import numpy as np

import pulsedata
from mass2 import LJHFile
from mass2.core.ljhutil import ljh_truncate, ljh_merge, ljh_append_traces, find_ljh_files, extract_channel_number

# ruff: noqa: PLR0914


def test_ljh_truncate_wrong_format(tmp_path):
    # First a file using LJH format 2.1.0 - should raise an exception
    src_name = str(pulsedata.pulse_noise_ljh_pairs["regression"].pulse_folder / "20150813_regression_pulse_chan1.ljh")
    dest_name = str(tmp_path / "xyz_chan1.ljh")

    with pytest.raises(Exception, match="Don't know how to truncate"):
        ljh_truncate(src_name, dest_name, n_pulses=100)


def test_ljh_merge_pre_2_2(tmp_path):
    src_name = str(pulsedata.pulse_noise_ljh_pairs["regression"].noise_folder / "20150813_regression_noise_chan1.ljh")
    dest_name = str(tmp_path / "merged_chan1.ljh")
    ljh_merge(dest_name, [src_name, src_name], overwrite=True)
    src = LJHFile.open(src_name)
    dest = LJHFile.open(dest_name)
    assert dest.npulses == 2 * src.npulses
    assert np.array_equal(dest._mmap[src.npulses :], src._mmap)


@pytest.mark.parametrize("src_offset", [b"1565016958.937494", b"1565009758.937494", b"1564998958.937494"])
def test_ljh_append_traces_pre_2_2_other_offset(tmp_path, src_offset):
    orig = pulsedata.pulse_noise_ljh_pairs["regression"].pulse_folder / "20150813_regression_pulse_chan1.ljh"
    raw = orig.read_bytes()
    dest_name, src_name = str(tmp_path / "dest_chan1.ljh"), str(tmp_path / "src_chan1.ljh")
    (tmp_path / "dest_chan1.ljh").write_bytes(raw)
    (tmp_path / "src_chan1.ljh").write_bytes(raw.replace(b"1565013358.937494", src_offset))
    src = LJHFile.open(src_name)
    if src_offset == b"1564998958.937494":
        with pytest.raises(ValueError, match="fall outside"):
            ljh_append_traces(src_name, dest_name, range(5))
        assert LJHFile.open(dest_name).npulses == src.npulses
        return
    ljh_append_traces(src_name, dest_name, range(5))
    dest = LJHFile.open(dest_name)
    assert np.array_equal(dest.datatimes_raw[src.npulses :], src.datatimes_raw[:5])
    assert np.array_equal(dest._mmap["pulse"][src.npulses :], src._mmap["pulse"][:5])


def run_test_ljh_truncate_timestamp(tpf, src_name, n_pulses_expected, timestamp):
    dest_name = str(tpf.mktemp("truncated_ljh", numbered=True) / "xyz_chan1.ljh")
    ljh_truncate(src_name, dest_name, timestamp=timestamp)

    src = LJHFile.open(src_name)
    dest = LJHFile.open(dest_name)
    assert n_pulses_expected == dest.npulses
    for k in range(n_pulses_expected):
        assert np.all(src.read_trace(k) == dest.read_trace(k))
        assert src.subframecount[k] == dest.subframecount[k]
        assert src.datatimes_float[k] == pytest.approx(dest.datatimes_float[k], abs=1e-5)


def run_test_ljh_truncate_n_pulses(tpf, src_name, n_pulses):
    # Tests with a file with 1230 pulses, each 1016 bytes long
    dest_name = str(tpf.mktemp("truncated_ljh", numbered=True) / "xyz_chan1.ljh")
    ljh_truncate(src_name, dest_name, n_pulses=n_pulses)

    src = LJHFile.open(src_name)
    dest = LJHFile.open(dest_name)
    assert n_pulses == dest.npulses
    for k in range(n_pulses):
        assert np.all(src.read_trace(k) == dest.read_trace(k))
        assert src.subframecount[k] == dest.subframecount[k]
        assert src.datatimes_float[k] == pytest.approx(dest.datatimes_float[k], abs=1e-5)


def test_ljh_truncate_n_pulses(tmp_path_factory):
    # Want to make sure that we didn't screw something up with the
    # segmentation, so try various lengths
    # Tests with a file with 3352 pulses, each 1016 bytes long
    src_name = pulsedata.pulse_noise_ljh_pairs["bessy_20240727"].noise_folder / "20240727_run0000_chan4219.ljh"
    run_test_ljh_truncate_n_pulses(tmp_path_factory, src_name, 1000)
    run_test_ljh_truncate_n_pulses(tmp_path_factory, src_name, 0)
    run_test_ljh_truncate_n_pulses(tmp_path_factory, src_name, 1)
    run_test_ljh_truncate_n_pulses(tmp_path_factory, src_name, 100)
    run_test_ljh_truncate_n_pulses(tmp_path_factory, src_name, 49)
    run_test_ljh_truncate_n_pulses(tmp_path_factory, src_name, 50)
    run_test_ljh_truncate_n_pulses(tmp_path_factory, src_name, 51)
    run_test_ljh_truncate_n_pulses(tmp_path_factory, src_name, 75)
    run_test_ljh_truncate_n_pulses(tmp_path_factory, src_name, 334)


def test_ljh_truncate_timestamp(tmp_path_factory):
    # Want to make sure that we didn't screw something up with the
    # segmentation, so try various lengths
    # Tests with a file with 3352 pulses, each 1016 bytes long
    src_name = pulsedata.pulse_noise_ljh_pairs["bessy_20240727"].noise_folder / "20240727_run0000_chan4219.ljh"
    run_test_ljh_truncate_timestamp(tmp_path_factory, src_name, 49, 1722086440433920 / 1e6)
    run_test_ljh_truncate_timestamp(tmp_path_factory, src_name, 50, 1722086440435920 / 1e6)
    run_test_ljh_truncate_timestamp(tmp_path_factory, src_name, 51, 1722086440437920 / 1e6)
    run_test_ljh_truncate_timestamp(tmp_path_factory, src_name, 75, 1722086440485930 / 1e6)
    run_test_ljh_truncate_timestamp(tmp_path_factory, src_name, 100, 1722086440535870 / 1e6)
    run_test_ljh_truncate_timestamp(tmp_path_factory, src_name, 334, 1722086441003870 / 1e6)
    run_test_ljh_truncate_timestamp(tmp_path_factory, src_name, 1000, 1722086442335960 / 1e6)


def test_find_ljh_files(tmp_path):
    strpath = str(tmp_path)
    all_chnum = set(range(8))
    for i in all_chnum:
        fpath = tmp_path / f"test_chan{i}.ljh"
        fpath.touch()

    # exclude, include, expect
    testset = (
        ([], None, all_chnum),
        ([], all_chnum, all_chnum),
        ([], [], []),
        ([], [2, 3, 4], [2, 3, 4]),
        ([], [2, 3, 10, -2], [2, 3]),
        ([1, 3, 5], None, [0, 2, 4, 6, 7]),
        ([1, 3, 5, 6, 7, 10, 200], None, [0, 2, 4]),
        ([1, 3, 5], [2, 4, 6], [2, 4, 6]),
        ([1, 3, 5, 99, 100, 101], [2, 4, 6, 6, 6, 6, 4], [2, 4, 6]),
        ([1, 3, 5], [2, 4, 5], "this should error"),
        ([2, 1], [1, 2], "this should error"),
    )
    for excl, incl, expect in testset:
        # If excl and incl have any overlap, raise an error
        if incl is not None and len(set(excl).intersection(incl)) > 0:

            def func():
                return find_ljh_files(strpath, exclude_ch_nums=excl, include_ch_nums=incl)

            pytest.raises(ValueError, func)
            continue

        # Test that the expected filenames are found, both by human- and computer-based expectations
        filenames = find_ljh_files(strpath, exclude_ch_nums=excl, include_ch_nums=incl)
        result = set([extract_channel_number(f) for f in filenames])
        assert set(expect) == result
        compute_expected = all_chnum.copy()
        if excl is not None and len(excl) > 0:
            compute_expected.difference_update(excl)
        if incl is not None:
            compute_expected.intersection_update(incl)
        assert compute_expected == result
