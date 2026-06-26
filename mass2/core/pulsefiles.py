"""
Classes and functions for reading and handling any files that allow random access
to pulse records.
"""

from collections.abc import Iterable
from dataclasses import dataclass
from numpy.typing import NDArray
from typing import BinaryIO, Any
import numpy as np
import mmap
from abc import ABC, abstractmethod


class PulseMaker(ABC):
    """An abstract base class that knows how to return one or many pulse records.

    Real data will use implementation PulseReader. Simulated data for testing may use MemReader."""

    @abstractmethod
    def pulse(self, id: int, fieldname: str = "data") -> NDArray:
        raise NotImplementedError("illegal: this is an abstract base class")

    @abstractmethod
    def pulses(self, ids: slice | range | Iterable, fieldname: str = "data") -> NDArray:
        raise NotImplementedError("illegal: this is an abstract base class")


@dataclass(frozen=True)
class PulseReader(PulseMaker):
    """A class to implement PulseMaker and read raw pulses from a memory map, for efficiency."""

    file_path: str
    fd: BinaryIO
    mm: mmap.mmap
    mv: memoryview
    dtype: np.dtype
    itemsize: int
    offset: int

    @property
    def length(self) -> int:
        return (len(self.mv) - self.offset) // self.itemsize

    @classmethod
    def open_by_path(cls, file_path: str, dtype: np.dtype, offset: int) -> "PulseReader":
        fd = open(file_path, "rb")
        mm = mmap.mmap(fd.fileno(), 0, access=mmap.ACCESS_READ)
        mv = memoryview(mm)
        return cls(file_path, fd, mm, mv, dtype, dtype.itemsize, offset)

    @classmethod
    def from_open_file(cls, fd: BinaryIO, dtype: np.dtype, offset: int) -> "PulseReader":
        file_path = fd.name
        mm = mmap.mmap(fd.fileno(), 0, access=mmap.ACCESS_READ)
        mv = memoryview(mm)
        return cls(file_path, fd, mm, mv, dtype, dtype.itemsize, offset)

    @classmethod
    def from_array_in_memory(cls, data: NDArray) -> "PulseReader":
        mv = memoryview(data)
        nsamples = data.shape[1]
        dtype = np.dtype([("data", np.uint16, nsamples)])
        return cls("", None, None, mv, dtype, dtype.itemsize, offset=0)

    def record(self, id: int) -> NDArray:
        """Return a single raw pulse record with timing data, selected by pulse id number.

        Parameters
        ----------
        id : int
            The id number of the pulse to be retrieved.

        Returns
        -------
        NDArray
            A 1-dimensional array of raw data, view into the underlying mmap.
        """
        start = self.offset + id * self.itemsize
        chunk = self.mv[start : start + self.itemsize]
        return np.frombuffer(chunk, self.dtype).ravel()

    def records(self, ids: slice | range | Iterable) -> NDArray:
        """Return a 2d array of raw pulse records, selected by pulse id numbers.

        Parameters
        ----------
        ids : slice | range | Iterable
            The id numbers of the pulse recordss to be retrieved. If this is a slice or range with the
            default (unit) step size, then this will return a mem-mapped array, which is very efficient.
            If the step size exceeds one, or the `ids` is an arbitrary iterable, the result will be a
            copy of data from the underlying memmap.

        Returns
        -------
        NDArray
            A 2-dimensional array of raw data, of size `(npulses, pulse_length)`.
        """
        # If asked for a range or slice with step=1, use the more efficient self._records_by_range
        # to return a memory-mapped array. But if step > 1 or `ids` is a generate Iterable,
        # we have to build a new array and copy in the pulse data.
        if type(ids) is slice:
            if ids.step is None or ids.step == 1:
                start = ids.start
                if start is None:
                    start = 0
                stop = ids.stop
                if stop is None:
                    stop = self.length
                range_ids = range(start, stop)
                return self._records_by_range(range_ids)
            ids = list(range(*ids.indices(self.length)))
        elif type(ids) is range:
            if ids.step == 1:
                return self._records_by_range(ids)
            ids = list(ids)
        return np.vstack([self.record(id) for id in ids])

    def _records_by_range(self, prange: range) -> NDArray:
        start = self.offset + prange.start * self.itemsize
        stop = self.offset + prange.stop * self.itemsize
        assert prange.step == 1
        byteslice = slice(start, stop)
        chunk = self.mv[byteslice]
        return np.frombuffer(chunk, self.dtype)

    def pulse(self, id: int, fieldname: str = "data") -> NDArray:
        """Return a single raw pulse, selected by pulse id number.

        Parameters
        ----------
        id : int
            The id number of the pulse to be retrieved.
        fieldname : str, optional
            The field name from the `self.dtype` composite type, by default "data".

        Returns
        -------
        NDArray
            A 1-dimensional array of raw data, view into the underlying mmap.
        """
        return self.record(id)[fieldname].ravel()

    def pulses(self, ids: slice | range | Iterable, fieldname: str = "data") -> NDArray:
        """Return a 2d array of pulse data, selected by pulse id numbers.

        Parameters
        ----------
        ids : slice | range | Iterable
            The id numbers of the pulses to be retrieved. If this is a slice or range with the default
            (unit) step size, then this will return a mem-mapped array, which is very efficient. If
            the step size exceeds one, or the `ids` is an arbitrary iterable, the result will be a
            copy of data from the underlying memmap.
        fieldname : str, optional
            The field name from the `self.dtype` composite type, by default "data".

        Returns
        -------
        NDArray
            A 2-dimensional array of raw data, of size `(npulses, pulse_length)`.
        """
        return self.records(ids)[fieldname].squeeze()

    def close(self) -> None:
        self.mv.release()
        self.mm.close()
        self.fd.close()

    def __getstate__(self) -> dict[str, Any]:
        """Define what gets pickled (ignore the live mmap and file handle)."""
        state = self.__dict__.copy()
        del state["mv"]
        del state["mm"]
        del state["fd"]
        return state


@dataclass(frozen=True)
class MemReader(PulseMaker):
    """A class to implement PulseMaker and store raw pulses in memory, for simplifying tests."""

    array: NDArray

    @classmethod
    def from_array(cls, data: NDArray) -> "MemReader":
        return cls(data)

    def pulse(self, id: int, fieldname: str = "data") -> NDArray:
        return self.array[id]

    def pulses(self, ids: slice | range | Iterable, fieldname: str = "data") -> NDArray:
        return self.array[ids]
