import argparse
import glob
import numpy as np
import os
import polars as pl
import pyarrow.parquet as pq
import re
from pathlib import Path

from .ljhutil import filename_glob_expand
from .ljhfiles import LJHFile

"""
Functions to translate NIST LJH file format into a new format based on:
- Apache Parquet (for compression and long-term archiving)
- Apache Arrow (for hot analysis of new pulse data)
- Apache Avro (for online streaming of low-rate data)

See especially the script `ljh2apache`
"""


def translate_external_trigger(args: argparse.Namespace) -> None:
    base = Path(args.base_dir)
    pattern = str(base / "*_external_trigger.bin")
    trig_files = glob.glob(pattern)
    nf = len(trig_files)
    assert nf < 2, f"Expect no more than one *_external_trigger.bin file, found {nf}"
    if nf == 0:
        if args.verbose:
            print("No external trigger file found")
        return
    path = trig_files[0]
    out_path = path.replace(".bin", ".parquet")
    if not args.force and os.path.exists(out_path):
        raise OSError(f"Cannot overwrite {out_path} without --force argument.")

    print(f"Writing {path}\n-> {out_path}")
    if args.dry_run:
        return
    with open(path, "rb") as _f:
        _header_line = _f.readline()  # read the one header line before opening the binary data
        external_trigger_subframe_count = np.fromfile(_f, "int64")
        df = pl.DataFrame({"ext_trig_subframecount": external_trigger_subframe_count})
    df.write_parquet(out_path)
    net = len(df)
    print(f"We found {net} external triggers in {path}")


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
    pattern = base / "*_chan*.ljh"
    ljh_filenames = filename_glob_expand(str(pattern))
    if len(ljh_filenames) == 0:
        print("No LJH files to translate")
        return

    ljh: dict[int, LJHFile] = {}
    for in_fname in ljh_filenames:
        matches = re.search(r"chan(\d+)\.ljh", in_fname)
        if matches:
            ch = matches.groups()[0]
            ch_num = int(ch)
            ljh[ch_num] = LJHFile.open(in_fname)
    print(f"There are {len(ljh)} LJH files to read:")
    print(f"Channels: {ljh.keys()}")
    out_path = str(base / "channel_metadata.parquet")
    print(f"Writing {out_path}")
    if not args.dry_run:
        df = generate_ljh_metadata_df(ljh)
        df.write_parquet(out_path)

    write_ljh_arrow(ljh, args)


def write_ljh_arrow(ljh: dict[int, LJHFile], args: argparse.Namespace) -> None:
    base = Path(args.base_dir)
    ljh0 = next(iter(ljh.values()))
    frames_per_sec = 1 / ljh0.timebase
    if ljh0.subframediv is None:
        subframes_per_sec = int(64 * frames_per_sec)
    else:
        subframes_per_sec = int(ljh0.subframediv * frames_per_sec)
    subframes = {k: v.subframecount for (k, v) in ljh.items()}
    posix_usec = {k: v.datatimes_raw for (k, v) in ljh.items()}
    first_subframe = np.min([sfc[0] for sfc in subframes.values()])
    final_subframe = np.max([sfc[-1] for sfc in subframes.values()])

    output_number = 0
    while first_subframe < final_subframe:
        last_subframe = first_subframe + args.period * subframes_per_sec
        out_name = f"pulse_data_{base32_crockford_encode(output_number, length=4)}.arrow"
        duration = (last_subframe - first_subframe) / subframes_per_sec
        print(f"Writing {out_name} with subframes {first_subframe}-{last_subframe} over {duration:.4f} s")
        if args.dry_run:
            first_subframe = last_subframe
            continue

        all_df: list[pl.DataFrame] = []
        for k, v in ljh.items():
            start_idx = np.searchsorted(subframes[k], first_subframe, side="left")
            stop_idx = np.searchsorted(subframes[k], last_subframe, side="right")
            df = v.load_raw_chunk(start_idx, stop_idx)
            df = df.with_columns(
                subframecount=subframes[k][start_idx:stop_idx],
                timestamp=posix_usec[k][start_idx:stop_idx],
                channel_number=pl.lit(k),
            ).with_columns(pl.from_epoch("timestamp", time_unit="us"))
            all_df.append(df)
        complete_df = pl.concat(all_df, rechunk=True)
        out_path = str(base / out_name)

        # Generate the index map that *would* sort these columns
        # If the data is already sorted, the index array is identical to its own row count
        indices = df.select(pl.arg_sort_by(["channel_number", "subframecount"])).to_series()
        if not indices.is_sorted():
            complete_df = complete_df.sort("channel_number", "subframecount")
        complete_df.write_ipc(out_path)

        first_subframe = last_subframe
        output_number += 1


def main_ljh2apache() -> None:
    parser = argparse.ArgumentParser(description="Convert a set of LJH files to new apache file formats")
    parser.add_argument("base_dir", type=str, help="directory of files to process, with *_chan*.ljh as the LJH files")
    parser.add_argument("-f", "--force", action="store_true", help="Overwrite existing data")
    parser.add_argument("-n", "--dry-run", action="store_true", help="Dry run: say what would be done, but don't do it")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose mode; print extra info to terminal")
    parser.add_argument(
        "-p", "--period", type=float, default=10, help="Period for starting new Arrow output files, in seconds (default=10)"
    )
    args = parser.parse_args()

    translate_external_trigger(args)
    translate_ljh_files(args)


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
