import pytest
import os
import numpy as np

import pulsedata
from mass2 import LJHFile
from mass2.core.ljhutil import ljh_truncate

# ruff: noqa: PLR0914


def test_ljh_truncate_wrong_format(tmp_path):
    # First a file using LJH format 2.1.0 - should raise an exception
    src_name = os.path.join("tests", "regression_test", "regress_chan1.ljh")
    destfile = tmp_path / "xyz_chan1.ljh"
    dest_name = str(destfile.name)

    def func():
        ljh_truncate(src_name, dest_name, n_pulses=100)

    pytest.raises(Exception, func)


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
