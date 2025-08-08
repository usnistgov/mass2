from dataclasses import dataclass, replace
from typing import Optional, ClassVar, Self, Tuple
import numpy.typing as npt
import os
import numpy as np
import polars as pl
from packaging.version import Version


@dataclass(frozen=True)
class LJHFile:
    """Represents the header and binary information of a single LJH file.

    Includes the complete ASCII header stored both as a dictionary and a string, and
    key attributes including the number of pulses, number of samples (and presamples)
    in each pulse record, client information stored by the LJH writer, and the filename.

    Also includes a `np.memmap` to the raw binary data. This memmap always starts with
    pulse zero and extends to the last full pulse given the file size at the time of object
    creation. To extend the memmap for files that are growing, use `LJHFile.reopen_binary()`
    to return a new object with a possibly longer memmap.
    """

    filename: str
    dtype: np.dtype
    npulses: int
    timebase: float
    nsamples: int
    npresamples: int
    subframediv: int
    client: str
    header: dict
    header_string: str
    header_size: int
    binary_size: int
    _mmap: np.memmap
    ljh_version: Version
    max_pulses: Optional[int] = None

    OVERLONG_HEADER: ClassVar[int] = 100

    @classmethod
    def open(cls, filename: str, max_pulses: Optional[int] = None) -> "LJHFile":
        header_dict, header_string, header_size = cls.read_header(filename)
        nsamples = header_dict["Total Samples"]
        timebase = header_dict["Timebase"]
        nsamples = header_dict["Total Samples"]
        npresamples = header_dict["Presamples"]
        client = header_dict.get("Software Version", "UNKNOWN")
        if "Subframe divisions" in header_dict:
            subframediv = header_dict["Subframe divisions"]
        elif "Number of rows" in header_dict:
            subframediv = header_dict["Number of rows"]
        else:
            subframediv = 0

        ljh_version = Version(header_dict["Save File Format Version"])
        if ljh_version < Version("2.2.0"):
            dtype = np.dtype(
                [
                    ("internal_us", np.uint8),
                    ("internal_unused", np.uint8),
                    ("internal_ms", np.uint32),
                    ("data", np.uint16, nsamples),
                ]
            )
        else:
            dtype = np.dtype(
                [
                    ("subframecount", np.int64),
                    ("posix_usec", np.int64),
                    ("data", np.uint16, nsamples),
                ]
            )
        pulse_size_bytes = dtype.itemsize
        binary_size = os.path.getsize(filename) - header_size

        # Fix long-standing bug in LJH files made by MATTER or XCALDAQ_client:
        # It adds 3 to the "true value" of nPresamples. For now, assume that only
        # DASTARD clients have this figure correct.
        if "DASTARD" not in client:
            npresamples += 3

        npulses = binary_size // pulse_size_bytes
        if max_pulses is not None:
            npulses = min(max_pulses, npulses)
        mmap = np.memmap(
            filename, dtype, mode="r", offset=header_size, shape=(npulses,)
        )

        return LJHFile(
            filename,
            dtype,
            npulses,
            timebase,
            nsamples,
            npresamples,
            subframediv,
            client,
            header_dict,
            header_string,
            header_size,
            binary_size,
            mmap,
            ljh_version,
            max_pulses,
        )

    @classmethod
    def read_header(cls, filename: str) -> Tuple[dict, str, int]:
        """Read in the text header of an LJH file. Return the header parsed into a dictionary,
        the complete header string (in case you want to generate a new LJH file from this one),
        and the size of the header in bytes. The file does not remain open after this method.

        Returns:
            (header_dict, header_string, header_size)

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
                    raise Exception(
                        f"header is too long--seems not to contain '#End of Header'\nin file {filename}"
                    )
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
        for name, datatype in {
            ("Channel", int),
            ("Timebase", float),
            ("Total Samples", int),
            ("Presamples", int),
            ("Number of columns", int),
            ("Number of rows", int),
            ("Subframe divisions", int),
        }:
            header_dict[name] = datatype(header_dict.get(name, -1))
        return header_dict, header_string, header_size

    @property
    def pulse_size_bytes(self) -> int:
        """The size in bytes of each binary pulse record (including the timestamps)"""
        return self.dtype.itemsize

    def reopen_binary(self, max_pulses: Optional[int] = None) -> Self:
        """Reopen the underlying binary section of the LJH file, in case its size has changed,
        without re-reading the LJH header section.

        Parameters
        ----------
        max_pulses : Optional[int], optional
            A limit to the number of pulses to memory map or None for no limit, by default None

        Returns
        -------
        Self
            A new `LJHFile` object with the same header but a new memmap and number of pulses.
        """
        current_binary_size = os.path.getsize(self.filename) - self.header_size
        npulses = current_binary_size // self.pulse_size_bytes
        if max_pulses is not None:
            npulses = min(max_pulses, npulses)
        mmap = np.memmap(
            self.filename,
            self.dtype,
            mode="r",
            offset=self.header_size,
            shape=(npulses,),
        )
        return replace(
            self,
            npulses=npulses,
            _mmap=mmap,
            max_pulses=max_pulses,
            binary_size=current_binary_size,
        )

    def read_trace(self, i: int) -> npt.ArrayLike:
        """Return a single pulse record from an LJH file.

        Parameters
        ----------
        i : int
            Pulse record number (0-indexed)

        Returns
        -------
        npt.ArrayLike
            A view into the pulse record.
        """
        return self._mmap["data"][i]

    def to_polars(
        self, first_pulse: int = 0, keep_posix_usec: bool = False
    ) -> Tuple[pl.DataFrame, pl.DataFrame]:
        """Convert this LJH file to two Polars dataframes: one for the binary data, one for the header.

        Parameters
        ----------
        first_pulse : int, optional
            The pulse dataframe starts with this pulse record number, by default 0
        keep_posix_usec : bool, optional
            Whether to keep the raw `posix_usec` field in the pulse dataframe, by default False

        Returns
        -------
        Tuple[pl.DataFrame, pl.DataFrame]
            (df, header_df)
            df: the dataframe containing raw pulse information, one row per pulse
            header_df: a one-row dataframe containing the information from the LJH file header
        """
        if self.ljh_version < Version("2.2.0"):
            raise NotImplementedError(
                "cannot convert LJH pre-2.2 files to Polars dataframes yet"
            )

        data = {
            "pulse": self._mmap["data"][first_pulse:],
            "posix_usec": self._mmap["posix_usec"][first_pulse:],
            "subframecount": self._mmap["subframecount"][first_pulse:],
        }
        schema = {
            "pulse": pl.Array(pl.UInt16, self.nsamples),
            "posix_usec": pl.UInt64,
            "subframecount": pl.UInt64,
        }
        df = pl.DataFrame(data, schema=schema)
        df = df.select(
            pl.from_epoch("posix_usec", time_unit="us").alias("timestamp")
        ).with_columns(df)
        if not keep_posix_usec:
            df = df.select(pl.exclude("posix_usec"))
        header_df = pl.DataFrame(self.header)
        return df, header_df

    def write_truncated_ljh(self, filename: str, npulses: int) -> None:
        """Write an LJH copy of this file, with a limited number of pulses.

        Parameters
        ----------
        filename : str
            The path where a new LJH file will be created (or replaced).
        npulses : int
            Number of pulse records to write
        """
        npulses = max(npulses, self.npulses)
        with open(filename, "wb") as f:
            f.write(self.header_string)
            f.write(self._mmap[:npulses].tobytes())

    @property
    def is_continuous(self) -> bool:
        """Is this LJH file made of a perfectly continuous data stream?

        We generally do take noise data in this mode, and it's useful to analyze the noise
        data by gluing many records together. This property says whether such gluing is valid.

        Returns
        -------
        bool
            Whether every record is strictly continuous with the ones before and after
        """
        expected_subframe_diff = self.nsamples * self.subframediv
        subframe = self._mmap["subframecount"]
        return np.max(np.diff(subframe)) <= expected_subframe_diff
