"""The command lines of the core: mass2-live-apply (run_live_apply) and mass2-live-hist (run_live_hist), both in loop.py."""

import argparse
from collections.abc import Sequence
from pathlib import Path

from .fit import RoiFit
from .histogram import HistogramSpec
from .loop import run_live_apply, run_live_hist


def apply_main(argv: Sequence[str] | None = None) -> None:
    """Entry point for `mass2-live-apply`."""
    p = argparse.ArgumentParser(description="Apply saved mass2 recipes to raw pulse records as they are written.")
    p.add_argument("recipes", type=Path, help="recipes pickle from Channels.save_recipes")
    p.add_argument("raw", type=Path, help="Arrow IPC stream of raw records (may still be growing, or not exist yet)")
    p.add_argument("analyzed", type=Path, help="Arrow IPC stream to write (must not exist yet)")
    p.add_argument("--poll", type=float, default=0.5, help="seconds between polls (default 0.5)")
    args = p.parse_args(argv)
    if args.analyzed.exists():
        p.error(f"{args.analyzed} already exists; remove it or name a new file")
    try:
        run_live_apply(args.recipes, args.raw, args.analyzed, args.poll)
    except KeyboardInterrupt:
        pass


def hist_main(argv: Sequence[str] | None = None) -> None:
    """Entry point for `mass2-live-hist`."""
    p = argparse.ArgumentParser(description="Histogram analyzed records by channel and state in time slices, and refit one line.")
    p.add_argument("analyzed", type=Path, help="Arrow IPC stream written by mass2-live-apply")
    p.add_argument("hist_dir", type=Path, help="directory for the histogram files")
    p.add_argument("--experiment-state", type=Path, default=None, help="the experiment_state.txt file to follow (default: none)")
    p.add_argument("--energy-col", default="energy_5lagy_best", help="column to histogram (default energy_5lagy_best)")
    p.add_argument("--slice", type=float, default=10.0, help="time slice, seconds of data time (default 10)")
    p.add_argument("--e-lo", type=float, default=0.0)
    p.add_argument("--e-hi", type=float, default=1200.0)
    p.add_argument("--bin", type=float, default=1.0, help="energy bin width, eV (default 1)")
    p.add_argument(
        "--line", type=parse_line, default=None, help="line to refit, a name (e.g. MnKAlpha) or energy in eV (default: none)"
    )
    p.add_argument("--dlo", type=float, default=50, help="fit from this far below the peak, eV (default 50)")
    p.add_argument("--dhi", type=float, default=50, help="fit to this far above the peak, eV (default 50)")
    p.add_argument("--every", type=float, default=10, help="refit at most this often, seconds (default 10)")
    p.add_argument("--source", default="", help="where this fit comes from, shown in the viewer")
    p.add_argument("--poll", type=float, default=0.5, help="seconds between polls (default 0.5)")
    args = p.parse_args(argv)
    for name in ["histograms.arrows", "fits.arrows"]:
        if (args.hist_dir / name).exists():
            p.error(f"{args.hist_dir / name} already exists; remove it or name a new directory")
    spec = HistogramSpec(e_lo=args.e_lo, e_hi=args.e_hi, bin_width=args.bin, slice_s=args.slice)
    roi = None if args.line is None else RoiFit(args.line, args.dlo, args.dhi, args.source)
    try:
        run_live_hist(
            args.analyzed,
            args.hist_dir,
            spec,
            args.energy_col,
            roi=roi,
            experiment_state_path=args.experiment_state,
            every_s=args.every,
            poll_s=args.poll,
        )
    except KeyboardInterrupt:
        pass


def parse_line(text: str) -> str | float:
    """A line name like MnKAlpha, or an energy in eV."""
    try:
        return float(text)
    except ValueError:
        return text
