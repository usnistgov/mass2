import argparse
import functools
import glob
import numpy as np
import os
import polars as pl
import pyarrow as pa
from pyarrow import ipc
import pyarrow.parquet as pq
import re
import shutil
import time
from pathlib import Path
from dataclasses import dataclass
from collections.abc import Iterable, Generator

from .ljhutil import filename_glob_expand
from .ljhfiles import LJHFile
from .misc import PulseDataFramer

import tzlocal

_local_timezone_name = tzlocal.get_localzone_name()

"""
Functions to translate NIST LJH file format into a new format based on:
- Apache Parquet (for compression and long-term archiving)
- Apache Arrow (for hot analysis of new pulse data)
- Apache Avro (for online streaming of low-rate data)

See especially the script `ljh2apache`
"""


@dataclass(frozen=True)
class PulseDataFromArrow(PulseDataFramer):
    lf: pl.LazyFrame
    path: Path

    @property
    def df(self) -> pl.DataFrame:
        return pl.read_ipc(self.path, memory_map=True)

    @functools.cached_property
    def npulses(self) -> int:
        return self.df.select(pl.len()).item()

    def iterate_raw_pulses(self, chunksize: int, extra_fields: Iterable[str] = []) -> Generator[pl.DataFrame]:
        """Yield successive chunks of `chunksize` raw pulses each, covering the whole source in order.
        These chunks are memory mapped to the underlying Arrows file."""
        df = self.df
        try:
            for i in range(0, self.npulses, chunksize):
                yield df[i : i + chunksize]
        finally:
            del df

    def load_raw_chunk(self, start: int, stop: int, step: int = 1, extra_fields: Iterable[str] = []) -> pl.DataFrame:
        stop = min(stop, self.npulses)
        s = slice(start, stop, step)
        return self.df[s].select("pulse")

    def load_raw_pulse(self, id: int, extra_fields: Iterable[str] = []) -> pl.DataFrame:
        return self.df.select("pulse").slice(id, 1)

    def load_raw_pulses(self, ids: Iterable[int], extra_fields: Iterable[str] = []) -> pl.DataFrame:
        return self.df.select("pulse").with_row_index("idx").filter(pl.col("idx").is_in(list(ids)))

    def load_timing(self) -> pl.DataFrame:
        """Load the timing information from a raw data Arrows file.

        Specifically, drop the pulse record column, and return *a copy* of the rest of
        the information. Does not generate a memory map or leave a file open.

        Returns
        -------
        pl.DataFrame
            A copy of the raw timing information, but no pulse records.
        """
        return self.lf.drop("pulse").collect()

    @classmethod
    def open(cls, path: str | Path) -> "PulseDataFromArrow":
        lf = pl.scan_ipc(path)
        return cls(lf, Path(path))


def translate_external_trigger(args: argparse.Namespace) -> None:
    base = Path(args.base_dir)
    output = Path(args.output)
    pattern = str(base / "*_external_trigger.bin")
    trig_files = glob.glob(pattern)
    nf = len(trig_files)
    assert nf < 2, f"Expect no more than one *_external_trigger.bin file, found {nf}"
    if nf == 0:
        if args.verbose:
            print("No external trigger file found")
        return
    binary_path = trig_files[0]
    input_basename = os.path.basename(binary_path)
    output_basename = input_basename.replace(".bin", ".parquet")
    parquet_path = output / output_basename
    if not args.force and os.path.exists(parquet_path):
        raise OSError(f"Cannot overwrite {parquet_path} without --force argument.")
    print(f"Converting {binary_path}\n-> {parquet_path}")

    with open(binary_path, "rb") as _f:
        _header_line = _f.readline()  # read the one header line before opening the binary data
        external_trigger_subframe_count = np.fromfile(_f, "int64")
        df = pl.DataFrame({"ext_trig_subframecount": external_trigger_subframe_count})
    net = len(df)
    print(f"We found {net} external triggers in {binary_path}")

    if args.dry_run:
        return
    df.write_parquet(parquet_path)


def generate_ljh_metadata_df(ljh: dict[int, LJHFile]) -> pl.DataFrame:
    files = list(ljh.values())
    return pl.DataFrame({
        "channel_number": [f.channum for f in files],
        "timebase": [f.timebase for f in files],
        "nsamples": [f.nsamples for f in files],
        "npresamples": [f.npresamples for f in files],
        "subframediv": [f.subframediv for f in files],
        "filename": [f.filename for f in files],
        "ljh_version": [str(f.ljh_version) for f in files],
    })


base32_crockford_symbols = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"


def base32_crockford_encode(number: int, length: int = 1) -> str:
    chars = []
    while number > 0:
        i = number % 32
        chars.append(base32_crockford_symbols[i])
        number //= 32
    nextra = length - len(chars)
    if nextra > 0:
        chars += nextra * ["0"]
    return "".join(chars[::-1])


def translate_ljh_files(args: argparse.Namespace) -> None:
    base = Path(args.base_dir)
    output = Path(args.output)
    pattern = base / "*_chan*.ljh"
    ljh_filenames = filename_glob_expand(str(pattern))
    if len(ljh_filenames) == 0:
        print("No LJH files to translate")
        return

    ljhfiles: dict[int, LJHFile] = {}
    for in_fname in ljh_filenames:
        matches = re.search(r"chan(\d+)\.ljh", in_fname)
        if matches:
            ch = matches.groups()[0]
            ch_num = int(ch)
            ljhfiles[ch_num] = LJHFile.open(in_fname)
    print(f"There are {len(ljhfiles)} LJH files to read:")
    print(f"Channels: {ljhfiles.keys()}")
    out_path = str(output / "channel_metadata.parquet")
    print(f"Writing {out_path}")
    if not args.dry_run:
        df = generate_ljh_metadata_df(ljhfiles)
        df.write_parquet(out_path)

    if args.mix:
        mix_ljh_arrow(ljhfiles, args)
    else:
        convert_ljh_arrow(ljhfiles, args)


def mix_ljh_arrow(ljhfiles: dict[int, LJHFile], args: argparse.Namespace) -> None:  # noqa: PLR0914
    """Write a numbered collection of LJHFiles into a series of Arrow files.

    The LJH files are numbered by channel number, but the output Arrow files are
    sequential and mix all channels into the same files.

    Parameters
    ----------
    ljhfiles : dict[int, LJHFile]
        A map from channel number to a single-channel LJHFile object.
    args : argparse.Namespace
        Command-line arguments that control conversion behavior.
    """
    # Extract the timing information (subframe count and posix timestamps) as dictionaries
    # of numpy arrays, indexed by channel number.
    ljh0 = next(iter(ljhfiles.values()))
    frames_per_sec = 1 / ljh0.timebase
    if ljh0.subframediv is None:
        subframes_per_sec = int(64 * frames_per_sec)
    else:
        subframes_per_sec = int(ljh0.subframediv * frames_per_sec)
    subframes_per_batch = int(args.batch * subframes_per_sec)
    subframes_per_file = int(args.period * subframes_per_sec)

    first_subframe = np.min([ljh._mmap[0]["subframecount"] for ljh in ljhfiles.values()])
    final_subframe = np.max([ljh._mmap[-1]["subframecount"] for ljh in ljhfiles.values()])
    next_idx = {k: 0 for k in ljhfiles.keys()}

    # Construct the Arrow files to contain `args.period` seconds of data apiece.
    output = Path(args.output)
    print(f"Writing arrow files to directory {output}/")

    nsamples = ljh0.nsamples
    raw_schema = pl.Schema([
        ("pulse", pl.Array(pl.UInt16, shape=(nsamples,))),
        ("subframecount", pl.Int64),
        ("timestamp", pl.Datetime(time_unit="us", time_zone="America/Denver")),
    ])

    # Loop over files. Every args.period, close file and start a new one.
    output_number = 0
    while first_subframe < final_subframe:
        last_subframe = first_subframe + subframes_per_file
        out_name = f"all_pulses_{base32_crockford_encode(output_number, length=3)}.arrows_WAL"
        out_path = str(output / out_name)

        duration = (last_subframe - first_subframe) / subframes_per_sec
        print(f"Analyzing subframes {first_subframe}-{last_subframe}. Duration: {duration:.4f} s")
        if args.dry_run:
            print(f"Writing {out_name}")
            first_subframe = last_subframe
            output_number += 1
            continue

        all_batches: list[pl.DataFrame] = []
        batch_first_subframe = first_subframe
        size_MB = 0.0
        while batch_first_subframe < last_subframe:
            batch_last_subframe = batch_first_subframe + subframes_per_batch
            all_df: list[pl.DataFrame] = []

            # For each LJH file, find the contiguous group of pulses that match this Arrow file batch
            # range of subframes: [batch_first_subframe, batch_last_subframe].
            for k, v in ljhfiles.items():
                start = next_idx[k]
                last_allowable_idx = v.npulses
                if start >= last_allowable_idx:
                    continue

                # Guess that 64 records is enough; if not, keep doubling the set until we have enough.
                step = 64
                while True:
                    stop = start + step
                    if stop > last_allowable_idx:
                        stop = last_allowable_idx
                        break
                    if v._mmap[stop - 1]["subframecount"] >= batch_last_subframe:
                        break
                    step *= 2
                sfc = v._mmap[start:stop]["subframecount"]
                number_in_range = (sfc < batch_last_subframe).sum()
                end = start + number_in_range
                next_idx[k] = end
                if end <= start:
                    if args.verbose:
                        print(f"     chan {k:2d} has no records")
                    continue

                mmap = v._mmap[start:end]
                df = (
                    pl.DataFrame(
                        {
                            "pulse": mmap["pulse"],
                            "subframecount": mmap["subframecount"],
                            "timestamp": mmap["posix_usec"],
                        },
                        schema=raw_schema,
                    )
                    .with_columns(
                        channel_number=pl.lit(k),
                    )
                    .with_columns(pl.from_epoch("timestamp", time_unit="us"))
                )
                all_df.append(df)
            batch_first_subframe = batch_last_subframe
            if len(all_df) == 0:
                continue
            complete_df = pl.concat(all_df, rechunk=True)
            df_size_mb = float(complete_df.estimated_size("megabytes"))
            all_batches.append(complete_df)
            size_MB += df_size_mb
            print(f"  Created batch {len(all_batches):4d} of size {df_size_mb:6.3f} MB, {len(complete_df)} rows")
            if size_MB >= args.max_mb:
                break

        # Write the Arrow IPC file, one batch at a time, and prepare for next iteration.
        first_table = all_batches[0].to_arrow()
        ipc_schema = first_table.schema
        print(f"Writing {out_name}", end="\n\r")
        with pa.OSFile(out_path, "wb") as f:
            with ipc.new_stream(f, ipc_schema) as writer:
                for i, df in enumerate(all_batches):
                    table = df.to_arrow()
                    writer.write_table(table)
                    # df_size_mb = float(complete_df.estimated_size("megabytes"))
                    print(f"\bWriting batch number: {i}", end="\r")
                    time.sleep(args.sleep)
                print()

        final_ouput_name = Path(out_path).with_suffix(".arrows_timeorder")
        os.rename(out_path, final_ouput_name)

        # Ready for the next IPC file
        first_subframe = batch_first_subframe
        output_number += 1


def main_ljh2apache() -> None:
    parser = argparse.ArgumentParser(description="Convert a set of LJH files to new apache file formats")
    parser.add_argument("base_dir", type=str, help="directory of files to process, with *_chan*.ljh as the LJH files")
    parser.add_argument("-o", "--output", type=str, help="Write output to this directory (default: same as base_dir)")
    parser.add_argument("-f", "--force", action="store_true", help="Overwrite existing data")
    parser.add_argument("-n", "--dry-run", action="store_true", help="Dry run: say what would be done, but don't do it")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose mode; print extra info to terminal")
    parser.add_argument("-m", "--max-mb", type=float, default=500, help="Maximum arrow file size in MB (default=500)")
    parser.add_argument(
        "-p", "--period", type=float, default=300, help="Period for starting new Arrow IPC stream files, in seconds (default=300)"
    )
    parser.add_argument(
        "-b", "--batch", type=float, default=5, help="Period for starting a new record batch within an arrow file (default=5)"
    )
    parser.add_argument("-s", "--sleep", type=float, default=0, help="Sleep this many second between writing each batch (default=0)")
    args = parser.parse_args()
    args.mix = True
    if not args.output:
        args.output = args.base_dir
    Path(args.output).mkdir(parents=True, exist_ok=True)

    translate_external_trigger(args)
    translate_ljh_files(args)


def convert_ljh_arrow(ljhfiles: dict[int, LJHFile], args: argparse.Namespace) -> None:
    """Write a numbered collection of LJHFiles into a series of single-channel Arrow IPC files.

    The LJH files are numbered by channel number.

    Parameters
    ----------
    ljhfiles : dict[int, LJHFile]
        A map from channel number to a single-channel LJHFile object.
    args : argparse.Namespace
        Command-line arguments that control conversion behavior.
    """
    for chnum, ljh in ljhfiles.items():
        ljhpath = Path(ljh.filename)
        ljhstem = ljhpath.stem
        outpath = Path(args.output) / (ljhstem + ".arrow")
        print(f"Converting chan {chnum:4d}: {ljhpath} -> {outpath}")
        if args.dry_run:
            continue

        df, _dfheader = ljh.to_polars(keep_raw_pulses=True)
        df.write_ipc(outpath)


def main_ljh2arrow() -> None:
    parser = argparse.ArgumentParser(description="Convert a set of LJH files to new single-channel arrow IPC files")
    parser.add_argument("base_dir", type=str, help="directory of files to process, with *_chan*.ljh as the LJH files")
    parser.add_argument("-o", "--output", type=str, help="write output to this directory (default: same as base_dir)")
    parser.add_argument("-f", "--force", action="store_true", help="overwrite existing data")
    parser.add_argument("-n", "--dry-run", action="store_true", help="dry run: say what would be done, but don't do it")
    parser.add_argument("-v", "--verbose", action="store_true", help="verbose mode; print extra info to terminal")
    args = parser.parse_args()
    args.mix = False
    if not args.output:
        args.output = args.base_dir
    Path(args.output).mkdir(parents=True, exist_ok=True)

    translate_external_trigger(args)
    translate_ljh_files(args)


def _get_parser(description: str, base_dir_help: str) -> argparse.ArgumentParser:
    """Helper to keep CLI arguments DRY."""
    parser = argparse.ArgumentParser(description=description)
    # Using type=Path directly parses the string into a Path object
    parser.add_argument("base_dir", type=Path, help=base_dir_help)
    parser.add_argument("output_dir", type=Path, help="write output to this directory")
    parser.add_argument("-f", "--force", action="store_true", help="overwrite existing data")
    parser.add_argument("-n", "--dry-run", action="store_true", help="dry run: say what would be done, but don't do it")
    return parser


def main_arrow2parquet() -> None:
    parser = _get_parser(
        description="Convert a set of single-channel arrow IPC files to equivalent parquet files",
        base_dir_help="directory of files to process, with *_chan*.arrow as the arrow files",
    )
    args = parser.parse_args()
    base = Path(args.base_dir)
    out = Path(args.output_dir)
    arrow_parquet(base, out, arrow2parquet=True, dry_run=args.dry_run, force=args.force)


def main_parquet2arrow() -> None:
    parser = _get_parser(
        description="Convert a set of single-channel parquet files to equivalent arrow IPC files",
        base_dir_help="directory of files to process, with *_chan*.parquet as the parquet files",
    )
    args = parser.parse_args()
    base = Path(args.base_dir)
    out = Path(args.output_dir)
    arrow_parquet(base, out, arrow2parquet=False, dry_run=args.dry_run, force=args.force)


def arrow_parquet(base: Path, out: Path, arrow2parquet: bool, dry_run: bool = False, force: bool = False) -> None:
    """Convert a directory of arrow files to another of equivalent parquet files, or vice versa

    Parameters
    ----------
    base : Path
        Where to find the input files (names will be *_chan.suffix)
    out : Path
        Where to place the output files (names will be *_chan.suffix)
    arrow2parquet : bool
        Whether to convert arrow to parquet, or the reverse. The input and output suffixes will be "arrow", "parquet"
        respectively if True, or the reverse if False.
    dry_run : bool
        Whether to print intentions only and not perform the conversions
    verbose : bool
        Whether to print extra information
    """
    if not dry_run:
        out.mkdir(parents=True, exist_ok=True)

    # Copy auxiliary files
    files_to_copy = {"channel_metadata.parquet", "*experiment_state.txt", "*external_trigger*.bin"}
    for pattern in files_to_copy:
        for f in base.glob(pattern):
            newpath = out / f.name
            if newpath.exists() and not force:
                print(f"Skipping {f.name} (already exists). Use --force to overwrite.")
                continue
            print(f"Copying {f} -> {newpath}")
            if not dry_run:
                shutil.copy2(f, newpath)

    if arrow2parquet:
        oldsuffix, newsuffix = "arrow", "parquet"
    else:
        oldsuffix, newsuffix = "parquet", "arrow"

    for f in base.glob(f"*_chan*.{oldsuffix}"):
        newpath = out / f.with_suffix(f".{newsuffix}").name
        if newpath.exists() and not force:
            print(f"Skipping {f.name} (already exists). Use --force to overwrite.")
            continue
        print(f"Converting {f} -> {newpath}")
        if not dry_run:
            if arrow2parquet:
                pl.scan_ipc(f).sink_parquet(newpath)
            else:
                pl.scan_parquet(f).sink_ipc(newpath)


def analyze_compression(file_path: str) -> None:
    # Read only the metadata footer (does not load the massive data arrays into RAM)
    # Run it on your archive file
    # analyze_compression("/telemetry_data/archive/experiments/experiment_id=Run_042/data.parquet")
    metadata = pq.read_metadata(file_path)

    total_compressed = 0
    total_uncompressed = 0

    print(f"--- Compression Report for: {file_path} ---")
    print(f"Total Rows: {metadata.num_rows}")
    print(f"Total Row Groups: {metadata.num_row_groups}\n")

    # Iterate through every row group and every column chunk inside it
    for rg_idx in range(metadata.num_row_groups):
        row_group = metadata.row_group(rg_idx)

        for col_idx in range(row_group.num_columns):
            col_chunk = row_group.column(col_idx)

            # Tally the byte sizes
            total_compressed += col_chunk.total_compressed_size
            total_uncompressed += col_chunk.total_uncompressed_size

            # Optional: Print per-column stats for the first row group
            if rg_idx == 0:
                col_name = col_chunk.path_in_schema
                comp_size = col_chunk.total_compressed_size
                uncomp_size = col_chunk.total_uncompressed_size
                ratio = uncomp_size / comp_size if comp_size > 0 else 0
                print(f"Column '{col_name}': {ratio:.2f}x compression ({comp_size} bytes)")

    # Calculate total efficiency
    overall_ratio = total_uncompressed / total_compressed if total_compressed > 0 else 0
    space_saved = (1 - (total_compressed / total_uncompressed)) * 100 if total_uncompressed > 0 else 0

    print("\n--- Overall File Totals ---")
    print(f"Uncompressed Size: {total_uncompressed / (1024**2):.2f} MB")
    print(f"Compressed Size:   {total_compressed / (1024**2):.2f} MB")
    print(f"Compression Ratio: {overall_ratio:.2f}x")
    print(f"Space Saved:       {space_saved:.1f}%")


if __name__ == "__main__":
    main_ljh2apache()
