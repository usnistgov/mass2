from IPython import start_ipython
import argparse
import os
import numpy as np
import scipy as sp
import pylab as plt
import polars as pl
from typing import Optional
import mass2

"""mass2main

    A script to start an iPython session with all LJH files in the current directory pre-loaded
"""


def load_ljh(directory: str, limit: Optional[int], exclude_ch_nums: Optional[int]) -> mass2.core.Channels:
    data = mass2.core.Channels.from_ljh_folder(directory, limit=limit, exclude_ch_nums=exclude_ch_nums)
    return data


def main():
    parser = argparse.ArgumentParser(
        description="Start a mass2 ipython session by loading a set of LJH files",
        # epilog="Channels will be excluded if in both the excluded and included lists."
    )
    parser.add_argument("dir", type=str, nargs="?", default=".", help="directory to find LJH files (default: current directory)")

    # TODO: have some way of handling an include-only-these list.
    # parser.add_argument("--include", metavar="ch", type=int, nargs="*", help="An optional list of channel numbers to include")
    parser.add_argument("-l", "--limit", type=int, default=None, help="load only LIMIT files (default: load all)")
    parser.add_argument("--exclude", metavar="ch", type=int, nargs="*", help="An optional list of channel numbers to exclude")
    parser.add_argument("-v", "--verbose", action="store_true", help="print arguments to terminal (default: False)")
    parser.add_argument(
        "-n",
        "--no-ipython",
        action="store_true",
        help="list files that could be loaded, then quit without running ipython (default: False)",
    )

    args = parser.parse_args()

    if args.verbose:
        print(f"Loading LJH files from directory: {args.dir}")
        if args.dir == ".":
            print(f" = {os.path.abspath(args.dir)}")
        print(f"Excluding channels in {args.exclude}")
    data = load_ljh(args.dir, limit=args.limit, exclude_ch_nums=args.exclude)

    print(f"Object 'data' is loaded with {len(data.channels)} channels available")
    if args.no_ipython:
        print("No ipython session started because of the no-ipython command-line argument.")
        return

    print("Launching mass2 IPython environment.")
    print("Importing mass2, numpy as np, scipy as sp, pylab as plt, polars as pl...")
    imports = {
        "mass2": mass2,
        "pl": pl,
        "sp": sp,
        "np": np,
        "plt": plt,
        "data": data,
    }
    start_ipython(argv=[], user_ns=imports)  # Launch IPython without parsing command-line arguments


if __name__ == "__main__":
    main()
