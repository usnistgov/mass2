import argparse
import glob
import polars as pl
from pathlib import Path

from .channels import Channels
from .noise_algorithms import NoiseResult


def analyze_noise_directory(
    directory: str | Path, excursion_nsigma: float = 5, verbose: bool = False, savefile: Path | str | None = None
) -> dict[int, NoiseResult]:
    """Analyze all the raw pulse data files in a given directory. Return the results, and optionally
    store them as parquet file.

    Parameters
    ----------
    directory : str | Path
        Where the noise LJH or single-channel Arrow files are to be found
    excursion_nsigma : float, optional
        Exclude noise records with excursions more than this many sigma from median, by default 5
    verbose : bool, optional
        Print extra facts to the terminal, by default False
    savefile : Path | str | None, optional
        Save results as an Apache Parquet file to this path, by default None

    Returns
    -------
    dict[int, NoiseResult]
        _description_
    """
    # Strategy will be to analyze all existing single-channel arrow files, then any LJH files (omitting any LJH
    # that copy the channel numbers of an arrow file in the same directory).
    results: dict[int, NoiseResult] = {}
    directory = Path(directory)
    prefix = ""
    arrow_files = glob.glob(str(directory / "*_chan*.arrow"))
    if len(arrow_files) > 0:
        arrow_files.sort()
        if verbose:
            print(f"Found {len(arrow_files)} Arrow files")
        data1 = Channels.from_ipc(directory)
        prefix = data1.file_prefix
        results = data1.analyze_noise(excursion_nsigma=excursion_nsigma)

    ljh_files = glob.glob(str(directory / "*_chan*.ljh"))
    if len(ljh_files) > 0:
        exclude = list(results.keys())
        data2 = Channels.from_ljh_folder(directory, exclude_ch_nums=exclude)
        if not prefix:
            prefix = data2.file_prefix
        if verbose:
            if len(arrow_files) == 0:
                print(f"Found {len(ljh_files)} LJH files")
            else:
                print(f"Found {len(ljh_files)} LJH files, excluding any also found as Arrow")
        results2 = data2.analyze_noise(excursion_nsigma=excursion_nsigma, verbose=verbose)
        results = results2 | results

    if not savefile:
        savefile = f"{directory}/{prefix}_noise_analysis.parquet"

    print(f"Writing to {savefile}")
    save_noise(results, savefile)
    return results


def save_noise(results: dict[int, NoiseResult], filename: Path | str) -> None:
    """Save noise analysis to a parquet file

    Parameters
    ----------
    results : dict[int, NoiseResult]
        A dictionary of `NoiseResult` objects, indexed by channel number.
    filename : Path | str
        Where to store the result
    """
    keys = list(results.keys())
    keys.sort()
    rows = []
    for cnum in keys:
        result = results[cnum]
        row = {
            "channel_number": cnum,
            "dfreq": result.frequencies[1],
            "dt": result.dt,
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
            "dt": pl.Float64,
            "PSD": pl.Array(pl.Float64, npsd),
            "autocorr": pl.Array(pl.Float64, nacorr),
        },
    )
    df.write_parquet(filename)


def main() -> None:
    """A main script to generate a noise analysis for 1 or more directories"""
    parser = argparse.ArgumentParser(
        description="Run a noise analysis on a set of LJH or Arrow IPC files",
    )
    parser.add_argument("dir", type=str, nargs="?", default=".", help="directory to find LJH/Arrow files (default: current directory)")
    parser.add_argument(
        "-o", "--output", type=str, default="", help="path to store result (default: dir/{prefix}_noise_analysis.parquet)"
    )
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

    analyze_noise_directory(dir, excursion_nsigma=args.excursion, verbose=args.verbose, savefile=args.output)


if __name__ == "__main__":
    main()
