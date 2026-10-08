"""Record the real pipeline on each demo dataset and write a standalone page that replays it.

For every dataset this runs the real simulator and the real `run_live_apply` and `run_live_hist` loops (with
the saved recipe) over one pass of the data, as fast as possible, keeping the ORIGINAL timestamps. Every time in
the recording is a real time of the original run. It records exactly what the live viewer reads: the histogram
slices `run_live_hist` finished, with the records behind each, the state changes, and a refit of the
dataset's line every 30 slices. The page plays the recording at a chosen multiple of real time, with no server.

Command line:  mass2-live-export OUT.html [--datasets bessy_20240727 20230626 ...] [--workdir DIR]
"""

import argparse
import base64
import json
import tempfile
from collections.abc import Sequence
from importlib import resources
from pathlib import Path

import numpy as np
import polars as pl
from numpy.typing import NDArray

from ..core.arrow_stream import ArrowStreamTailer
from ..core.fit import LiveFitter
from ..core.states import STATE_SCHEMA
from ..core.histogram import df_to_slices
from ..core.loop import run_live_apply, run_live_hist
from .datasets import DATASETS, DemoDataset
from .launcher import viewer_meta
from .simulate import load_experiment_states, load_ljh_sources, simulate, experiment_state_path

FIT_EVERY_SLICES = 30


def record(dataset: DemoDataset, workdir: Path) -> dict:
    """Run one pass of `dataset` through the real pipeline and return the recording for the page.

    The raw pulse stream is deleted once processed (it is by far the largest file); its size is kept.
    """
    stream, analyzed, hist = workdir / "pulses.arrows", workdir / "analyzed.arrows", workdir / "hist"
    states = load_experiment_states(dataset.pulse_folder)
    simulate(load_ljh_sources(dataset.pulse_folder), stream, pace=False, scaled=dataset.scales, states=states)
    run_live_apply(dataset.recipe_path, stream, analyzed, poll_s=0)
    run_live_hist(
        analyzed, hist, dataset.spec, dataset.energy_col, experiment_state_path=experiment_state_path(stream), poll_s=0, grace_s=0
    )
    input_bytes, output_bytes = stream.stat().st_size, analyzed.stat().st_size
    stream.unlink()

    slices, fits, t0_us = _read_slices_and_refit(dataset, hist / "histograms.arrows")
    records = sum(n for _, n, _ in slices)
    states = pl.DataFrame(schema=STATE_SCHEMA) if states is None else states
    changes = states.select(pl.col("timestamp").dt.epoch("us"), "state_label").rows()
    return {
        "key": dataset.key,
        "title": dataset.title,
        "meta": dataset.spec.to_dict() | {"energy_col": dataset.energy_col} | viewer_meta(dataset.key) | {
            "t0": t0_us / 1e6,  # absolute time (s) of offset 0 in this recording, a slice_s boundary
            "bytes_per_record": {"input": input_bytes / records, "output": output_bytes / records},
        },
        "states": [[round((t - t0_us) / 1e6, 6), st] for t, st in changes],
        "slices": slices,
        "fits": fits,
    }  # fmt: skip


def _read_slices_and_refit(dataset: DemoDataset, path: Path) -> tuple[list, dict, int]:
    """Read the finished slices batch by batch (a full-resolution copy of them all would not fit in memory),
    keeping each one sparse as [seconds after t0, records, sparse counts], and refit the line as mass2-live-hist
    does, after every FIT_EVERY_SLICES slices."""
    spec, slices, fitter, out, t0_us = dataset.spec, [], LiveFitter(dataset.spec, dataset.roi), [], None
    reader = ArrowStreamTailer(path)
    while not reader.ended:
        reader, frames, _ = reader.poll(64_000_000)
        for df in frames:
            for s in df_to_slices(df):
                t0_us = (s.start_us // spec.slice_us) * spec.slice_us if t0_us is None else t0_us
                slices.append([round((s.start_us - t0_us) / 1e6, 6), sum(s.records.values()), sparse_counts(s.counts)])
                fitter = fitter.add([s])
                if fitter.slices % FIT_EVERY_SLICES == 0:
                    fitter, fit = fitter.fit()
                    if fit is not None:
                        entry = fit.row(0, named=True)
                        entry["t"] = round(entry["t"] - t0_us / 1e6, 3)  # seconds into the recording
                        entry["png"] = "data:image/png;base64," + base64.b64encode(entry["png"]).decode()
                        out.append(entry)
    roi = dataset.roi
    fits = {"roi": {"label": roi.label, "source": roi.source, "dlo": roi.dlo, "dhi": roi.dhi},
            "every_s": FIT_EVERY_SLICES * spec.slice_s, "fits": out}  # fmt: skip
    assert t0_us is not None, "the recording has no histogram slices"
    return slices, fits, t0_us


def write_replay_page(recordings: list[dict], out: Path) -> None:
    """The viewer page with `recordings` embedded; it plays them instead of asking a server."""
    page = resources.files("mass2.live.viewer").joinpath("viewer.html").read_text(encoding="utf-8")
    data = json.dumps({"datasets": recordings}, separators=(",", ":")).replace("</", "<\\/")
    marker = "<!-- replay data -->"
    assert marker in page
    out.write_text(page.replace(marker, f'<script id="replay">window.MASS2_REPLAY = {data};</script>'), encoding="utf-8")


def sparse_counts(counts: dict[tuple[int, str], NDArray]) -> dict[str, dict[str, list[int]]]:
    """{state: {channel: [bin gap, count, bin gap, count, ...]}}: nonzero bins only, bin numbers delta-coded.

    This is how slices travel to the viewer: with bins as fine as the resolution fits use (up to ~20,000),
    nearly all bins of a short slice are empty.
    """
    out: dict[str, dict[str, list[int]]] = {}
    for (ch, state), c in counts.items():
        nz = np.flatnonzero(c)
        if len(nz) > 0:
            gaps = np.diff(nz, prepend=0)
            out.setdefault(state, {})[str(ch)] = np.column_stack([gaps, c[nz]]).ravel().tolist()
    return out


def main(argv: Sequence[str] | None = None) -> None:
    """Entry point for `mass2-live-export`."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("out", type=Path, help="HTML file to write")
    p.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=list(DATASETS), help="datasets to include (default: all)")
    p.add_argument("--workdir", type=Path, default=None, help="keep the recorded results here (default: a temporary directory)")
    args = p.parse_args(argv)

    with tempfile.TemporaryDirectory() as tmp:
        workdir = Path(tmp) if args.workdir is None else args.workdir
        recordings = []
        for key in args.datasets:
            print(f"mass2-live-export: recording {key}", flush=True)
            recordings.append(record(DATASETS[key], workdir / key))
    write_replay_page(recordings, args.out)
    print(f"mass2-live-export: wrote {args.out} ({args.out.stat().st_size / 1e6:.1f} MB)", flush=True)


if __name__ == "__main__":
    main()
