import argparse
import glob
import os
from pathlib import Path
import polars as pl
import mass2


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Perform online analysis on a set of LJH files while they are being written",
    )
    parser.add_argument("-F", "--force", action="store_true", help="force overwrite of existing analysis? (default: False)")
    parser.add_argument("-v", "--verbose", action="store_true", help="print more facts (default: False)")
    parser.add_argument(
        "-o", "--output-dir", nargs="?", const=None, default=None, help="store output to this directory (default: $ljhpath/mc2)"
    )
    parser.add_argument(
        "-p", "--analysis-period", default=10.0, type=float, help="repeat basic analysis at this repetition period (seconds)"
    )
    # parser.add_argument("-f", "--fast-analysis-period", default=1.0, type=float,
    #                     help="repeat basic analysis at this repetition period for any fast channels (seconds)")
    # parser.add_argument("-c", "--fast-channels", type=int, nargs="*", default=None,
    #                     help="analyze this subset of channel numbers at a faster rate")
    parser.add_argument("recipefile", type=str, nargs=1, help="pickle file with dictionary of recipes")
    parser.add_argument("ljhpath", type=str, nargs=1, help="")

    args = parser.parse_args()

    ljhpath = Path(args.ljhpath)
    data = mass2.core.Channels.from_ljh_folder(ljhpath)

    output_dir = ljhpath / "mc2"
    filefullpath = data.ch0.header.df["Filename"][0]
    _, filename = os.path.split(filefullpath)
    prefix = filename.split("_chan")[0]
    parquet_model = str(output_dir / f"{prefix}_analyzed_")
    if output_dir.exists():
        search_path = output_dir / "*_analyzed_*.parquet"
        if args.force:
            os.rename(output_dir, ljhpath / "mc2_prev")
        elif len(glob.glob(str(search_path))) > 0:
            raise OSError(
                f"Cannot use existing output directory {output_dir} with parquet files, unless you choose the --force argument"
            )

    else:
        output_dir.mkdir()

    run_recipe_loop(data, args.recipefile, parquet_model, args.verbose)


def run_recipe_loop(data: mass2.Channels, recipefile: str, parquet_model: str, verbose: bool = False) -> None:
    parquet_counter = 0
    while True:
        parquet_file = str(parquet_model) + f"{parquet_counter:04d}.parquet"
        data = data.load_recipes(recipefile)
        df = pl.DataFrame()
        for ch_num, ch in data.channels.items():
            chdf = ch.df.drop("pulse")
            df = pl.concat([df, chdf])
        df.write_parquet(parquet_file)
        parquet_counter += 1

        break
        # TODO: wipe old info from each channel in data, add any incremental data newly added to the LJH file.
        # For a quick test, we could maybe just read LJH files 1000 pulses at a time.
