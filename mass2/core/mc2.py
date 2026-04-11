import argparse
import glob
import os
from pathlib import Path
import polars as pl
import shutil
import time
import mass2


def make_output(output_dir: Path, search_path: Path, force: bool):
    if output_dir.exists() and len(glob.glob(str(search_path))) > 0:
        if force:
            newname = output_dir / ".." / "mc2_prev"
            if newname.exists():
                shutil.rmtree(newname)
            os.rename(output_dir, newname)
            output_dir.mkdir()
        else:
            raise OSError(
                f"Cannot use existing output directory {output_dir} with parquet files, unless you choose the --force argument"
            )
    else:
        output_dir.mkdir()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Perform online analysis on a set of LJH files while they are being written",
    )
    parser.add_argument("-F", "--force", action="store_true", help="force overwrite of existing analysis? (default: False)")
    parser.add_argument("-v", "--verbose", action="store_true", help="print more facts (default: False)")
    parser.add_argument("-a", "--archival", action="store_true", help="data is archival, not live (default: False)")
    parser.add_argument(
        "-o", "--output-dir", nargs="?", const=None, default=None, help="store output to this directory (default: $ljhpath/mc2)"
    )
    parser.add_argument(
        "-p", "--analysis-period", default=10.0, type=float,
        help="repeat basic analysis at this repetition period in seconds (default: 10)"
    )
    parser.add_argument(
        "-P", "--pulses_per_parquet", default=1000, type=int,
        help="analyze no more that this many pulses per channel in each iteration (default: 1000)"
    )
    parser.add_argument(
        "-m", "--minpulses", default=10, type=int,
        help="don't analyze new data from a channel unless there are at least this many pulses (default: 10)"
    )
    parser.add_argument(
        "-d", "--dry-run", action="store_true",
        help="don't perform analysis, just say what would be done (default: False)"
    )
    # parser.add_argument("-f", "--fast-analysis-period", default=1.0, type=float,
    #                     help="repeat basic analysis at this repetition period for any fast channels (seconds)")
    # parser.add_argument("-c", "--fast-channels", type=int, nargs="*", default=None,
    #                     help="analyze this subset of channel numbers at a faster rate")
    parser.add_argument("recipefile", type=str, nargs=1, help="pickle file with dictionary of recipes")
    parser.add_argument("ljhpath", type=str, nargs=1, help="")

    args = parser.parse_args()
    if args.archival:
        args.analysis_period = 0.0
    ljhpath = Path(args.ljhpath[0])

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = ljhpath / "mc2"
    if not args.dry_run:
        search_path = output_dir / "*_analyzed_*.parquet"
        make_output(output_dir, search_path, force=args.force)

    paths = mass2.core.ljhutil.find_ljh_files(ljhpath)
    paths = mass2.core.ljhutil.ljh_sort_filenames_numerically(paths)
    filefullpath = paths[0]
    _, filename = os.path.split(filefullpath)
    prefix = filename.split("_chan")[0]
    parquet_file_prefix = str(output_dir / f"{prefix}_analyzed_")

    ljhfiles = {}
    for p in paths:
        ljh = mass2.core.LJHFile.open(p, last_pulse=args.pulses_per_parquet)
        ljhfiles[ljh.channum] = ljh
    del ljhfiles[0]

    run_recipe_loop(ljhfiles, parquet_file_prefix, args)
    # try:
    # except KeyboardInterrupt:
    #     return


ENCODING = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"


def encode_counter(j: int, nchar: int = 4) -> str:
    """Encode integer `j` as a base-32 string, using the 32 characters in ENCODING
    (Crockford's base-32 encoding)."""
    c = []
    for i in range(nchar):
        c.append(ENCODING[j % 32])
        j //= 32
    c.reverse()
    return "".join(c)


def last_good_expr(ch: mass2.Channel) -> pl.Expr:
    "Use the `good_expr` from the last step that has one, or literal True if none do."
    for step in reversed(ch.steps):
        try:
            return step.good_expr
        except AttributeError:
            continue
    return pl.lit(True)


def run_recipe_loop(ljhfiles: dict[int, mass2.LJHFile], parquet_file_prefix: str, args : argparse.Namespace) -> None:

    recipefile = args.recipefile[0]
    analysis_period = args.analysis_period
    # verbose = args.verbose
    parquet_counter = 0
    processed_counter = {ch_num: 0 for ch_num in ljhfiles.keys()}

    while True:
        channels = {ch_num: mass2.Channel.from_open_ljh(ljh) for (ch_num, ljh) in ljhfiles.items()}
        for ch_num in ljhfiles.keys():
            if channels[ch_num].npulses <= 0:
                del channels[ch_num]
        data = mass2.Channels(channels, "mc2 channels")
        if not args.dry_run:
            data = data.load_recipes(recipefile)
        dframes = []
        # processed = set({})
        for ch_num, ch in data.channels.items():
            if len(ch.df) < args.minpulses:
                continue
            # processed.add(ch_num)
            good_expr = last_good_expr(ch)
            chdf = ch.df.drop("pulse").with_columns(good_expr.alias("good"), pl.lit(ch_num).alias("channel"))
            processed_counter[ch_num] += len(chdf)
            print(f"Channel {ch_num:3d} processed {len(chdf):5d} pulses.")
            dframes.append(chdf)

        # Write parquet file, if there are any data to write.
        if len(dframes) > 0:
            df = pl.concat(dframes)
            code = encode_counter(parquet_counter)
            parquet_file = str(parquet_file_prefix) + f"{code}.parquet"
            if not args.dry_run:
                df.write_parquet(parquet_file)
            parquet_counter += 1
            print(f"...written to {parquet_file}\n")
        else:
            if args.archival:
                print("No remaining archival data")
                return
            print("...waiting for new data")
            time.sleep(5.0)

        # Now wait a time to load and analyze new data.
        time.sleep(analysis_period)
        for ch_num in data.channels.keys():
            first = processed_counter[ch_num]
            last = first + args.pulses_per_parquet
            ljhfiles[ch_num] = ljhfiles[ch_num].reopen_binary(first, last)
