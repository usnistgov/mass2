"""
Tools for working with continuous data, as taken by the True Bequerel project.
"""

from numpy.typing import NDArray, ArrayLike
import numpy as np
import polars as pl
import pylab as plt
import hashlib
from dataclasses import dataclass
from pathlib import Path
from numba import njit

# import pyarrow as pa
from .channel import Channel, ChannelHeader
from .noise_channel import NoiseChannel
from . import misc


header_dtype = np.dtype([
    ("format", np.uint32),
    ("schema", np.uint32),
    ("sample_rate_hz", np.float64),
    ("data_reduction_factor", np.int16),
    ("voltage_scale", np.float64),
    ("aquisition_flags", np.uint16),
    ("start_time", np.uint64, 2),
    ("stop_time", np.uint64, 2),  # often wrong, written at end of run
    ("number_of_samples", np.uint64),  # often wrong, written at end of run
])


@dataclass(frozen=True)
class TriggerResult:
    """A trigger result from applying a triggering filter and threshold to a TrueBqBin data source."""

    data_source: "TrueBqBin"
    filter_in: np.ndarray
    threshold: float
    trig_inds: np.ndarray
    limit_samples: int

    def plot(
        self, decimate: int = 10, n_limit: int = 100000, offset_raw: int = 0, x_axis_time_s: bool = False, ax: plt.Axes | None = None
    ) -> None:
        """Make a diagnostic plot of the trigger result."""
        if ax is None:
            plt.figure()
            ax = plt.gca()
        plt.sca(ax)

        # raw (full-resolution) index ranges
        raw_start = offset_raw
        raw_stop = raw_start + n_limit * decimate

        data = self.data_source.data

        # scaling for x-axis (applied after decimation)
        x_scale = self.data_source.frametime_s * decimate if x_axis_time_s else 1

        # raw filter output
        filt_raw = fast_apply_filter(data[raw_start:raw_stop], self.filter_in)

        # decimated data and filter
        data_dec = data[raw_start:raw_stop:decimate]
        filt_dec = filt_raw[::decimate]

        # truncate to the same length
        n = min(len(data_dec), len(filt_dec))
        data_dec = data_dec[:n]
        filt_dec = filt_dec[:n]

        # shared x-axis
        x_dec = np.arange(n) * x_scale

        # plot data + filter
        plt.plot(x_dec, data_dec, ".", label="data")
        plt.plot(x_dec, filt_dec, label="filter_out")

        plt.axhline(self.threshold, label="threshold")

        # trigger indices (raw) → restrict to plotted window → convert to decimated indices
        trig_inds_raw = (
            pl.DataFrame({"trig_inds": self.trig_inds})
            .filter(pl.col("trig_inds").is_between(raw_start, raw_stop))
            .to_series()
            .to_numpy()
        )
        trig_inds_dec = (trig_inds_raw - raw_start) // decimate
        # clip to avoid indexing past n
        trig_inds_dec = trig_inds_dec[trig_inds_dec < n]
        plt.plot(x_dec[trig_inds_dec], filt_dec[trig_inds_dec], "o", label="trig_inds filt")
        plt.plot(x_dec[trig_inds_dec], data_dec[trig_inds_dec], "o", label="trig_inds data")

        # labels
        plt.title(f"{self.data_source.description}, trigger result debug plot")
        plt.legend()
        plt.xlabel("time with arb offset / s" if x_axis_time_s else "sample number (decimated)")
        plt.ylabel("signal (arb)")

    def get_noise(
        self,
        n_dead_samples_after_pulse_trigger: int,
        n_record_samples: int,
        max_noise_triggers: int = 200,
    ) -> NoiseChannel:
        """Synthesize a NoiseChannel from the data source by finding time periods without pulse triggers."""
        noise_trigger_inds = get_noise_trigger_inds(
            self.trig_inds,
            n_dead_samples_after_pulse_trigger,
            n_record_samples,
            max_noise_triggers,
        )
        inds = noise_trigger_inds[noise_trigger_inds > 0]  # ensure all inds are greater than 0
        inds = inds[inds < (len(self.data_source.data) - n_record_samples)]  # ensure all inds inbounds
        pulses = gather_pulses_from_inds_numpy_contiguous(
            self.data_source.data,
            npre=0,
            nsamples=n_record_samples,
            inds=inds,
        )
        df = pl.DataFrame({"pulse": pulses, "framecount": inds})
        noise = NoiseChannel(
            df,
            header_df=self.data_source.header_df,
            frametime_s=self.data_source.frametime_s,
        )
        return noise

    def to_channel_copy_to_memory(
        self, noise_n_dead_samples_after_pulse_trigger: int, npre: int, npost: int, invert: bool = False
    ) -> Channel:
        """Create a Channel object by copying pulse data into memory."""
        noise = self.get_noise(
            noise_n_dead_samples_after_pulse_trigger,
            npre + npost,
            max_noise_triggers=1000,
        )
        inds = self.trig_inds[self.trig_inds > npre]
        inds = inds[inds < (len(self.data_source.data) - npre - npost)]  # ensure all inds inbounds
        pulses = gather_pulses_from_inds_numpy_contiguous(self.data_source.data, npre=npre, nsamples=npre + npost, inds=inds)
        assert pulses.shape[0] == len(inds), "pulses and trig_inds must have the same length"
        if invert:
            df = pl.DataFrame({"pulse": pulses * -1, "framecount": inds})
        else:
            df = pl.DataFrame({"pulse": pulses, "framecount": inds})
        ch_header = ChannelHeader(
            self.data_source.description,
            None,
            self.data_source.channel_number,
            self.data_source.frametime_s,
            npre,
            npre + npost,
            self.data_source.header_df,
        )
        ch = Channel(df, ch_header, npulses=len(pulses), noise=noise)
        return ch

    def to_channel_mmap(
        self,
        noise_n_dead_samples_after_pulse_trigger: int,
        npre: int,
        npost: int,
        invert: bool = False,
        verbose: bool = True,
    ) -> Channel:
        """Create a Channel object by memory-mapping pulse data from disk."""
        noise = self.get_noise(
            noise_n_dead_samples_after_pulse_trigger,
            npre + npost,
            max_noise_triggers=1000,
        )
        inds = self.trig_inds[self.trig_inds > npre]  # ensure all inds inbounds
        inds = inds[inds < (len(self.data_source.data) - npre - npost)]  # ensure all inds inbounds
        pulses = gather_pulses_from_inds_numpy_contiguous_mmap_with_cache(
            self.data_source.data,
            npre=npre,
            nsamples=npre + npost,
            inds=inds,
            bin_path=self.data_source.bin_path,
            verbose=verbose,
        )
        if invert:
            df = pl.DataFrame({"pulse": pulses * -1, "framecount": inds})
        else:
            df = pl.DataFrame({"pulse": pulses, "framecount": inds})
        ch_header = ChannelHeader(
            self.data_source.description,
            None,
            self.data_source.channel_number,
            self.data_source.frametime_s,
            npre,
            npre + npost,
            self.data_source.header_df,
        )
        ch = Channel(df, ch_header, npulses=len(pulses), noise=noise)
        return ch

    # def to_summarized_channel(
    #     self,
    #     noise_n_dead_samples_after_pulse_trigger,
    #     npre,
    #     npost,
    #     peak_index=None,
    #     pretrigger_ignore=0,
    #     invert=False,
    # ):
    #     batch_size = 10000
    #     n = len(self.trig_inds)
    #     n_batches = np.ceil(n / batch_size).astype(int)
    #     dfs = []
    #     for i_batch in range(n_batches):
    #         # i0 = i_batch*batch_size
    #         # i1 = min((i_batch+1)*batch_size, n)
    #         # inds = self.trig_inds[i0:i1]
    #         inds = self.trig_inds
    #         inds = inds[inds > npre]  # ensure all inds inbounds
    #         inds = inds[inds < (len(self.data_source.data) - npre - npost)]  # ensure all inds inbounds
    #         pulses = gather_pulses_from_inds_numpy_contiguous(
    #             self.data_source.data,
    #             npre=npre,
    #             nsamples=npre + npost,
    #             inds=inds,
    #         )
    #         if invert:
    #             pulses *= -1
    #         if i_batch == 0 and peak_index is None:  # learn peak index
    #             peak_index = int(np.median(np.amax(pulses, axis=1)))
    #         assert isinstance(peak_index, int), "peak_index must be an integer"
    #         print(f"summarizing batch {i_batch=}/{n_batches=}")
    #         print(f"{self.data_source.frametime_s=}, {peak_index=}, {pretrigger_ignore=}, {npre=}")
    #         summary_np = pulse_algorithms.summarize_data_numba(
    #             pulses,
    #             self.data_source.frametime_s,
    #             peak_samplenumber=peak_index,
    #             pretrigger_ignore_samples=pretrigger_ignore,
    #             nPresamples=npre,
    #         )
    #         df_batch = pl.from_numpy(summary_np)
    #         df_batch = df_batch.with_columns(pl.DataFrame({"framecount": np.array(inds) + npre}))
    #         dfs.append(df_batch)
    #     df = pl.concat(dfs)
    #     ch_header = ChannelHeader(
    #         self.data_source.description,
    #         None,
    #         self.data_source.channel_number,
    #         self.data_source.frametime_s,
    #         npre,
    #         npre + npost,
    #         self.data_source.header_df,
    #     )
    #     noise = self.get_noise(
    #         noise_n_dead_samples_after_pulse_trigger,
    #         npre + npost,
    #         max_noise_triggers=1000,
    #     )
    #     # The following lines are broken August 8, 2025. Replace them with something non-broken.
    #     # pulse_storage = PulseStorageInArray(self.data_source.data, self.trig_inds, npre, npre + npost)
    #     # ch = Channel(df, ch_header, noise, pulse_storage=pulse_storage)
    #     return Channel(df, ch_header, npulses=len(df), noise=noise)


@dataclass(frozen=True)
class TrueBqBin:
    """Represents a binary data file from the True Bequerel project."""

    bin_path: Path
    description: str
    channel_number: int
    header_df: pl.DataFrame
    frametime_s: float
    voltage_scale: float
    data: np.ndarray
    # the bin file is a continuous data aqusition, untriggered

    @classmethod
    def load(cls, bin_path: str | Path) -> "TrueBqBin":
        """Create a TrueBqBin object by memory-mapping the given binary file."""
        bin_path = Path(bin_path)
        try:
            # for when it's named like dev2_ai6
            channel_number = int(str(bin_path.parent)[-1])
        except ValueError:
            # for when it's named like 2A
            def bay2int(bay: str) -> int:
                """Convert a bay name like '2A' to a channel number like 4."""
                return (int(bay[0]) - 1) * 4 + "ABCD".index(bay[1].upper())

            channel_number = bay2int(str(bin_path.parent.stem))
        desc = str(bin_path.parent.parent.stem) + "_" + str(bin_path.parent.stem)
        header_np = np.memmap(bin_path, dtype=header_dtype, mode="r", offset=0, shape=1)
        sample_rate_hz = header_np["sample_rate_hz"][0]
        header_df = pl.from_numpy(header_np)
        data = np.memmap(bin_path, dtype=np.int16, mode="r", offset=68)
        return cls(
            bin_path,
            desc,
            channel_number,
            header_df,
            1 / sample_rate_hz,
            header_np["voltage_scale"][0],
            data,
        )

    def trigger(self, filter_in: NDArray, threshold: float, limit_hours: float | None = None, verbose: bool = True) -> TriggerResult:
        """Compute trigger indices by applying the given filter and threshold to the data."""
        if limit_hours is None:
            limit_samples = len(self.data)
        else:
            limit_samples = int(limit_hours * 3600 / self.frametime_s)
        trig_inds = _fasttrig_filter_trigger_with_cache(self.data, filter_in, threshold, limit_samples, self.bin_path, verbose=verbose)
        return TriggerResult(self, filter_in, threshold, trig_inds, limit_samples)


def write_truebq_bin_file(
    path: str | Path,
    data: np.ndarray,
    sample_rate_hz: float,
    *,  # force keyword only
    voltage_scale: float = 1.0,
    format_version: int = 1,
    schema_version: int = 1,
    data_reduction_factor: int = 1,
    acquisition_flags: int = 0,
    start_time: np.ndarray | None = None,
    stop_time: np.ndarray | None = None,
) -> None:
    """
    Write a binary file that can be opened by TrueBqBin.load().

    This function writes data efficiently without copying by using memory mapping
    and direct file operations.

    Args:
        path: Output file path
        data: Data array to write (will be converted to int16 if not already)
        sample_rate_hz: Sample rate in Hz
        voltage_scale: Voltage scaling factor
        format_version: File format version (default: 1)
        schema_version: Schema version (default: 1)
        data_reduction_factor: Data reduction factor (default: 1)
        acquisition_flags: Acquisition flags (default: 0)
        start_time: Start time as uint64 array of length 2 (optional)
        stop_time: Stop time as uint64 array of length 2 (optional)
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Ensure data is int16 (convert if necessary, but avoid unnecessary copying)
    if data.dtype != np.int16:
        if not np.can_cast(data.dtype, np.int16, casting="safe"):
            print(f"Warning: Converting data from {data.dtype} to int16 may cause data loss")
        data = data.astype(np.int16)

    # Prepare header
    num_samples = len(data)

    # Default time values if not provided
    if start_time is None:
        start_time = np.array([0, 0], dtype=np.uint64)
    if stop_time is None:
        stop_time = np.array([0, 0], dtype=np.uint64)

    # Create header array
    header = np.array(
        [
            (
                format_version,
                schema_version,
                sample_rate_hz,
                data_reduction_factor,
                voltage_scale,
                acquisition_flags,
                start_time,
                stop_time,
                num_samples,
            )
        ],
        dtype=header_dtype,
    )

    # Create the file with the correct size
    with open(path, "wb") as f:
        # Write header
        f.write(header.tobytes())

        # For large data arrays, write in chunks to avoid memory issues
        chunk_size = 1024 * 1024  # 1MB chunks

        if data.nbytes <= chunk_size:
            # Small data, write directly
            f.write(data.tobytes())
        else:
            # Large data, write in chunks
            data_flat = data.ravel()  # Flatten without copying if possible
            for i in range(0, len(data_flat), chunk_size // data.itemsize):
                chunk = data_flat[i : i + chunk_size // data.itemsize]
                f.write(chunk.tobytes())


@njit
def fasttrig_filter_trigger(data: NDArray, filter_in: NDArray, threshold: float, verbose: bool) -> NDArray:
    """Apply a filter to the data and return trigger indices where the filter output crosses the threshold."""
    assert threshold > 0, "algorithm assumes we trigger with positive threshold, change sign of filter_in to accomodate"
    filter_len = len(filter_in)
    inds = []
    jmax = len(data) - filter_len - 1
    # njit only likes float64s, so I'm trying to force float64 use without allocating a ton of memory
    cache = np.zeros(len(filter_in))
    filter = np.zeros(len(filter_in))
    filter[:] = filter_in
    # intitalize a,b,c
    j = 0
    cache[:] = data[j : (j + filter_len)]
    b = np.dot(cache, filter)
    a = b  # won't be used, just need same type
    j = 1
    cache[:] = data[j : (j + filter_len)]
    c = np.dot(cache, filter)
    j = 2
    ready = False
    prog_step = jmax // 100
    prog_ticks = 0
    while j <= jmax:
        if j % prog_step == 0:
            prog_ticks += 1
            if verbose:
                print(f"fasttrig_filter_trigger {prog_ticks}/{100}")
        a, b = b, c
        cache[:] = data[j : (j + filter_len)]
        c = np.dot(cache, filter)
        if b > threshold and b >= c and b > a and ready:
            inds.append(j)
            ready = False
        if b < 0:  # hold off on retriggering until we see opposite sign slope
            ready = True
        j += 1
    return np.array(inds)


def gather_pulses_from_inds_numpy_contiguous(data: NDArray, npre: int, nsamples: int, inds: NDArray) -> NDArray:
    """Gather pulses from data at the given trigger indices, returning a contiguous numpy array."""
    assert all(inds > npre), "all inds must be greater than npre"
    assert all(inds < (len(data) - nsamples)), "all inds must be less than len(data) - nsamples"
    offsets = inds - npre  # shift by npre to start at correct offset
    pulses = np.zeros((len(offsets), nsamples), dtype=np.int16)
    for i, offset in enumerate(offsets):
        pulses[i, :] = data[offset : offset + nsamples]
    return pulses


def gather_pulses_from_inds_numpy_contiguous_mmap(
    data: NDArray, npre: int, nsamples: int, inds: NDArray, filename: str | Path = ".mmapped_pulses.npy"
) -> NDArray:
    """Gather pulses from data at the given trigger indices, returning a memory-mapped numpy array."""
    assert all(inds > npre), "all inds must be greater than npre"
    assert all(inds < (len(data) - nsamples)), "all inds must be less than len(data) - nsamples"
    offsets = inds - npre  # shift by npre to start at correct offset
    pulses = np.memmap(filename, dtype=np.int16, mode="w+", shape=(len(offsets), nsamples))
    for i, offset in enumerate(offsets):
        pulses[i, :] = data[offset : offset + nsamples]
    pulses.flush()
    # re-open the mmap to ensure it is read-only
    del pulses
    pulses = np.memmap(filename, dtype=np.int16, mode="r", shape=(len(offsets), nsamples))
    return pulses


"""
def gather_pulses_from_inds_pyarrow_share_memory(data, npre, nsamples, inds):
    # pyarrow supports the +vL datatype, which is defined by three arrays
    # one is the data source, the 2nd is the offsets, and the 3rd is the lengths
    # however polars does not support this datatype, so this is exploratory
    # code looking at the feature
    # https://arrow.apache.org/docs/python/data.html#listview-arrays
    # this would allow us to keep our records in the dataframe at little/no cost

    inds = inds[inds > nsamples]  # ensure all inds inbounds
    inds = inds[inds < (len(data) - nsamples)]  # ensure all inds inbounds
    offsets = inds - npre  # shift by npre to start at correct offset
    pool = pa.default_memory_pool()
    allocated_before = pool.bytes_allocated()
    # LargeListViewArray uses 64bit offets
    pulses = pa.LargeListViewArray(
        offsets=inds - inds,
        sizes=[nsamples]
        * len(inds),  # i can't find a constructor that takes offsets and a fixed size
        values=data,
        pool=pool,
    )
    allocation_increase = pool.bytes_allocated() - allocated_before
    # ensure memory is shared
    # 1. address to pyarrow buffer is the same as address to the numpy array
    assert pulses.buffers()[4].address == data.ctypes.data
    # 2. offsets in the pyarrow ListArray match our offsets
    assert all(pulses.offsets == inds)
    # 3. not many bytes allocated
    # could be made more precise by knowing size of data type in bytes
    assert allocation_increase < (len(data) / 2)
    return pulses, offsets
"""


def filter_and_residual_rms(
    data: NDArray, chosen_filter: NDArray, avg_pulse: NDArray, trig_inds: NDArray, npre: int, nsamples: int, polarity: int
) -> tuple[NDArray, NDArray, NDArray]:
    """Apply a filter to pulses extracted from data at the given trigger indices, returning filter values and residual RMS."""
    filt_value = np.zeros(len(trig_inds))
    residual_rms = np.zeros(len(trig_inds))
    filt_value_template = np.zeros(len(trig_inds))
    template = avg_pulse - np.mean(avg_pulse)
    template /= np.sqrt(np.dot(template, template))
    for i in range(len(trig_inds)):
        j = trig_inds[i]
        pulse = data[j - npre : j + nsamples - npre] * polarity
        pulse -= pulse.mean()
        filt_value[i] = np.dot(chosen_filter, pulse)
        filt_value_template[i] = np.dot(template, pulse)
        residual = pulse - template * filt_value_template[i]
        residual_rms_val = misc.root_mean_squared(residual)
        residual_rms[i] = residual_rms_val
    return filt_value, residual_rms, filt_value_template


@njit
def fast_apply_filter(data: NDArray, filter_in: NDArray) -> NDArray:
    """Apply a filter to the data, returning the filter output."""
    cache = np.zeros(len(filter_in))
    filter = np.zeros(len(filter_in))
    filter[:] = filter_in
    filter_len = len(filter)
    filter_out = np.zeros(len(data) - len(filter))
    j = 0
    jmax = len(data) - filter_len - 1
    while j <= jmax:
        cache[:] = data[j : (j + filter_len)]
        filter_out[j] = np.dot(cache, filter)
        j += 1
    return filter_out


def get_noise_trigger_inds(
    pulse_trigger_inds: ArrayLike,
    n_dead_samples_after_previous_pulse: int,
    n_record_samples: int,
    max_noise_triggers: int,
) -> NDArray:
    """Get trigger indices for noise periods, avoiding pulses."""
    pulse_trigger_inds = np.asarray(pulse_trigger_inds)
    diffs = np.diff(pulse_trigger_inds)
    inds = []
    for i in range(len(diffs)):
        if diffs[i] > n_dead_samples_after_previous_pulse:
            n_make = (diffs[i] - n_dead_samples_after_previous_pulse) // n_record_samples
            ind0 = pulse_trigger_inds[i] + n_dead_samples_after_previous_pulse
            for j in range(n_make):
                inds.append(ind0 + n_record_samples * j)
                if len(inds) == max_noise_triggers:
                    return np.array(inds)
    return np.array(inds)


def _fasttrig_filter_trigger_with_cache(
    data: NDArray, filter_in: NDArray, threshold: float, limit_samples: int, bin_path: str | Path, verbose: bool = True
) -> NDArray:
    """Apply a filter to the data and return trigger indices where the filter output crosses the threshold, using a cache."""
    bin_full_path = Path(bin_path).absolute()
    actual_n_samples = min(len(data), limit_samples)
    to_hash_str = str(filter_in) + str(threshold) + str(actual_n_samples) + str(bin_full_path)
    key = hashlib.sha256(to_hash_str.encode()).hexdigest()
    fname = f".{key}.truebq_trigger_cache.npy"
    cache_dir_path = bin_full_path.parent / "_truebq_bin_cache"
    cache_dir_path.mkdir(exist_ok=True)
    file_path = cache_dir_path / fname
    try:
        trig_inds = np.load(file_path)
        if verbose:
            print(f"trigger cache hit for {file_path}")
    except FileNotFoundError:
        if verbose:
            print(f"trigger cache miss for {file_path}")
        data_trunc = data[:actual_n_samples]
        trig_inds = fasttrig_filter_trigger(data_trunc, filter_in, threshold, verbose=verbose)
        np.save(file_path, trig_inds)
    return trig_inds


def gather_pulses_from_inds_numpy_contiguous_mmap_with_cache(
    data: NDArray, npre: int, nsamples: int, inds: NDArray, bin_path: Path | str, verbose: bool = True
) -> NDArray | np.memmap:
    """Gather pulses from data at the given trigger indices, returning a memory-mapped numpy array, using a cache."""
    bin_full_path = Path(bin_path).absolute()
    inds = inds[inds > npre]  # ensure all inds inbounds
    inds = inds[inds < (len(data) - nsamples)]  # ensure all inds inbounds
    inds_hash = hashlib.sha256(inds.tobytes()).hexdigest()
    to_hash_str = str(npre) + str(nsamples) + str(bin_full_path) + inds_hash
    key = hashlib.sha256(to_hash_str.encode()).hexdigest()
    fname = f".{key}.truebq_pulse_cache.npy"
    cache_dir_path = bin_full_path.parent / "_truebq_bin_cache"
    cache_dir_path.mkdir(exist_ok=True)
    file_path = cache_dir_path / fname
    inds = np.array(inds)
    if file_path.is_file():
        # check if the file is the right size
        Nbytes = len(inds) * nsamples * 2  # 2 bytes per int16
        if file_path.stat().st_size != Nbytes:
            # on windows the error if the file is the wrong size makes it sound like you don't have enough memory
            # and python doesn't seem to catch the exception, so we check the size here
            if verbose:
                print(f"pulse cache is corrupted, re-gathering pulses for {file_path}")
            file_path.unlink()
            cache_hit = False
        else:
            cache_hit = True
    else:
        cache_hit = False

    if cache_hit:
        if verbose:
            print(f"pulse cache hit for {file_path}")
        return np.memmap(file_path, dtype=np.int16, mode="r", shape=(len(inds), nsamples))
    if verbose:
        print(f"pulse cache miss for {file_path}")
    return gather_pulses_from_inds_numpy_contiguous_mmap(data, npre, nsamples, inds, filename=file_path)
