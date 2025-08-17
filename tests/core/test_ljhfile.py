import glob
import numpy as np
import os
import struct
from packaging.version import Version

from mass2 import LJHFile


def test_read_header():
    """Make sure the files in ljh_example_headers/ have headers that get parsed."""
    for file in glob.glob("tests/core/ljh_example_headers/*.ljh"):
        header_d, _, _ = LJHFile.read_header(file)
        assert 200 == header_d["Total Samples"]
        assert 100 == header_d["Presamples"]
        version = Version(header_d["Save File Format Version"])
        assert version >= Version("2.0.0")
        assert header_d["Timestamp offset (s)"] > 1.0e9


def test_ljh_all_versions(tmp_path):
    """Generate fake LJH files using the Version 2.0, 2.1, and 2.2 headers found in ljh_example_headers/
    Fill them with random pulse data and equally-spaced timestamps and rowcounts. Check that they are read
    back correctly."""
    rng = np.random.default_rng(100)
    npulse = 101
    nsamples = 200
    times_microsec = np.linspace(0, 8000000, npulse, dtype=np.uint64)
    rowcount = np.linspace(0, 250000, npulse, dtype=np.uint64)
    pulses = rng.integers(1000, 29000, size=(npulse, nsamples), dtype=np.uint16)
    v200 = Version("2.0.0")
    v210 = Version("2.1.0")
    v220 = Version("2.2.0")
    for filename in glob.glob("tests/core/ljh_example_headers/*.ljh"):
        template_file = LJHFile.open(filename)
        assert nsamples == template_file.nsamples
        npre = 100
        time_offset = round(template_file.header["Timestamp offset (s)"] * 1e6)
        version = Version(template_file.header["Save File Format Version"])
        if "Dastard" not in template_file.header["Software Version"]:
            npre += 3
        assert npre == template_file.npresamples

        # Create a temporary LJH file using the header from tests/core/ljh_example_headers, with
        # npulse records of length nsamples,
        _, basename = os.path.split(filename)
        created_path = str(tmp_path / basename)
        with open(created_path, "wb") as fp:
            fp.write(template_file.header_string.encode("utf-8"))
            for i in range(npulse):
                t_ms = times_microsec[i] // 1000
                if version == v200:
                    b = struct.pack("<hI", 0, t_ms)
                elif version == v210:
                    b = struct.pack("<BBI", 0, 0, t_ms)
                elif version == v220:
                    b = struct.pack("<QQ", rowcount[i], time_offset + times_microsec[i])
                fp.write(b)
                # print(len(b))
                fp.write(pulses[i].tobytes())

        file = LJHFile.open(created_path)
        assert nsamples == file.nsamples
        npre = 100
        if "Dastard" not in file.header["Software Version"]:
            npre += 3
        assert npre == file.npresamples

        assert np.all(file.datatimes_raw == times_microsec + time_offset)
        if version == v220:
            assert np.all(file.subframecount == rowcount)
        for i in range(npulse):
            assert np.all(file.read_trace(i) == pulses[i, :])
