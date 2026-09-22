import argparse
import glob
import polars as pl
from pathlib import Path

import mass2


def noise_analysis(directory: Path, excursion_nsigma: float = 5, verbose: bool = False) -> dict[int, mass2.NoiseResult]:
    results: dict[int, mass2.NoiseResult] = {}
    arrow_files = glob.glob(str(directory / "*_chan*.arrow"))
    if len(arrow_files) > 0:
        arrow_files.sort()
        if verbose:
            print(f"Found {len(arrow_files)} Arrow files")
        data1 = mass2.Channels.from_ipc(directory)
        for cnum, ch in data1.channels.items():
            if verbose:
                print(f"Analyzing {ch.header.data_source}")
            nch = ch.to_noisechannel()
            results[cnum] = nch.spectrum()

    ljh_files = glob.glob(str(directory / "*_chan*.ljh"))
    if len(ljh_files) > 0:
        exclude = list(results.keys())
        data1 = mass2.Channels.from_ljh_folder(directory, exclude_ch_nums=exclude)
        if verbose:
            if len(arrow_files) == 0:
                print(f"Found {len(ljh_files)} LJH files")
            else:
                print(f"Found {len(ljh_files)} LJH files, with {len(data1.channels)} not duplicating Arrow")
        for cnum, ch in data1.channels.items():
            if verbose:
                print(f"Analyzing {ch.header.data_source}")
            nch = ch.to_noisechannel()
            results[cnum] = nch.spectrum()

    return results


def save_noise(results: dict[int, mass2.NoiseResult], filename: Path | str) -> None:
    keys = list(results.keys())
    keys.sort()
    rows = []
    for cnum in keys:
        result = results[cnum]
        row = {
            "channel_number": cnum,
            "dfreq": result.frequencies[1],
            "PSD": result.psd,
            "autocorr": result.autocorr_vec,
        }
        rows.append(row)
        npsd = len(result.psd)
        if result.autocorr_vec is not None:
            nacorr = len(result.autocorr_vec)

    if len(rows) == 0:
        print("No noise files found!")
        return

    df = pl.DataFrame(
        rows,
        schema={
            "channel_number": pl.Int32,
            "dfreq": pl.Float64,
            "PSD": pl.Array(pl.Float64, npsd),
            "autocorr": pl.Array(pl.Float64, nacorr),
        },
    )
    df.write_parquet(filename)


def main():
    parser = argparse.ArgumentParser(
        description="Run a noise analysis on a set of LJH or Arrow IPC files",
    )
    parser.add_argument("dir", type=str, nargs="?", default=".", help="directory to find LJH/Arrow files (default: current directory)")
    parser.add_argument("output", type=str, nargs="?", default="", help="path to store result (default: dir/noise_analysis.parquet)")
    parser.add_argument("-v", "--verbose", action="store_true", help="print arguments to terminal (default: False)")
    parser.add_argument(
        "-x",
        "--excursion",
        type=float,
        default=5.0,
        help="exclude noise records with excursions more than this many sigma from median (default: 5)",
    )

    args = parser.parse_args()
    dir = Path(args.dir)
    if not args.output:
        args.output = dir / "noise_analysis.parquet"

    results = noise_analysis(dir, excursion_nsigma=args.excursion, verbose=args.verbose)
    save_noise(results, args.output)


if __name__ == "__main__":
    main()
