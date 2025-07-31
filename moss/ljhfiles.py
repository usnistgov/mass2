from dataclasses import dataclass, replace
from typing import Optional, ClassVar, Self, Tuple
import numpy.typing as npt
import os
import numpy as np
import polars as pl


@dataclass(frozen=True)
class LJHFile:
    filename: str
    dtype: np.dtype
    npulses: int
    timebase: float
    nsamples: int
    npresamples: int
    client: str
    header: dict
    header_string: str
    header_size: int
    binary_size: int
    _mmap: np.memmap
    first_pulse: int = 0
    max_pulses: Optional[int] = None
    OVERLONG_HEADER: ClassVar[int] = 100

    @classmethod
    def open(cls, filename: str, first_pulse: int = 0, max_pulses: Optional[int] = None) -> "LJHFile":
        header_dict, header_string, header_size = cls.read_header(filename)
        nsamples = header_dict["Total Samples"]
        timebase = header_dict["Timebase"]
        nsamples = header_dict["Total Samples"]
        npresamples = header_dict["Presamples"]
        client = header_dict.get("Software Version", "UNKNOWN")
        dtype = np.dtype([('rowcount', np.int64),
                          ('posix_usec', np.int64),
                          ('data', np.uint16, nsamples)])
        pulse_size_bytes = dtype.itemsize
        binary_size = os.path.getsize(filename) - header_size

        # Fix long-standing bug in LJH files made by MATTER or XCALDAQ_client:
        # It adds 3 to the "true value" of nPresamples. For now, assume that only
        # DASTARD clients have this figure correct.
        if "DASTARD" not in client:
            npresamples += 3

        full_npulses = binary_size // pulse_size_bytes
        npulses = full_npulses - first_pulse
        if max_pulses is not None:
            npulses = min(max_pulses, npulses)
        offset = header_size + first_pulse * pulse_size_bytes
        mmap = np.memmap(filename, dtype, mode="r",
                         offset=offset, shape=(npulses,))

        return LJHFile(filename, dtype, npulses, timebase, nsamples, npresamples, client,
                       header_dict, header_string, header_size, binary_size,
                       mmap, first_pulse, max_pulses)

    @classmethod
    def read_header(cls, filename: str) -> Tuple[dict, str, int]:
        """Read in the text header of an LJH file.

        On success, several attributes will be set: self.timebase, .nSamples,
        and .nPresamples

        Args:
            filename: path to the file to be opened.
        """
        # parse header into a dictionary
        header_dict = {}
        with open(filename, "rb") as fp:
            i = 0
            lines = []
            while True:
                line = fp.readline().decode()
                lines.append(line)
                i += 1
                if line.startswith("#End of Header"):
                    break
                elif line == "":
                    raise Exception("reached EOF before #End of Header")
                elif i > cls.OVERLONG_HEADER:
                    raise Exception(f"header is too long--seems not to contain '#End of Header'\nin file {filename}")
                # ignore lines without ":"
                elif ":" in line:
                    a, b = line.split(":", maxsplit=1)
                    a = a.strip()
                    b = b.strip()
                    header_dict[a] = b
            header_size = fp.tell()
        header_string = "".join(lines)

        # Convert values from header_dict into numeric types, when appropriate
        header_dict["Filename"] = filename
        for (name, datatype) in {
            ("Channel", int),
            ("Timebase", float),
            ("Total Samples", int),
            ("Presamples", int),
            ("Number of columns", int),
            ("Number of rows", int),
        }:
            header_dict[name] = datatype(header_dict.get(name, -1))
        return header_dict, header_string, header_size

    @property
    def pulse_size_bytes(self):
        return self.dtype.itemsize

    def reopen_binary(self, first_pulse: int = 0, max_pulses: Optional[int] = None) -> Self:
        current_binary_size = os.path.getsize(self.filename) - self.header_size
        full_npulses = current_binary_size // self.pulse_size_bytes
        npulses = full_npulses - first_pulse
        if max_pulses is not None:
            npulses = min(max_pulses, npulses)
        offset = self.header_size + first_pulse * self.pulse_size_bytes
        mmap = np.memmap(self.filename, self.dtype, mode="r", offset=offset, shape=(npulses,))
        return replace(self, _mmap=mmap, first_pulse=first_pulse, max_pulses=max_pulses, binary_size=current_binary_size)

    def read_trace(self, i) -> npt.ArrayLike:
        return self._mmap[i]["data"]

    def to_polars(self, keep_posix_usec=False) -> Tuple[pl.DataFrame, pl.DataFrame]:
        df = pl.DataFrame({"pulse": self._mmap["data"],
                           "posix_usec": self._mmap["posix_usec"],
                           "rowcount": self._mmap["rowcount"]},
                          schema={"pulse": pl.Array(pl.UInt16, self.nsamples),
                                  "posix_usec": pl.UInt64,
                                  "rowcount": pl.UInt64})
        df = df.select(pl.from_epoch("posix_usec", time_unit="us").alias("timestamp")).with_columns(df)
        if not keep_posix_usec:
            df = df.select(pl.exclude("posix_usec"))
        header_df = pl.DataFrame(self.header)
        return df, header_df

    def write_truncated_ljh(self, filename: str, npulses: int) -> None:
        with open(filename, "wb") as f:
            f.write(self.header_string)
            for i in range(npulses):
                f.write(self._mmap[i].tobytes())
