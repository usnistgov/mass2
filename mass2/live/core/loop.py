"""THE CORE LOOPS. Each follows a growing Arrow IPC stream until it ends: read the new records, process them,
write the results. Everything else is imported.

`run_live_apply`: read the new raw pulse records, have mass2 apply each channel's saved recipe, write the
analyzed records.

    mass2-live-apply RECIPES.pkl RAW.arrows ANALYZED.arrows

`run_live_hist`: read the new analyzed records, histogram the good ones by channel and experiment state in time
slices, write the finished histograms; and refit one line every few seconds. It writes, all as Arrow IPC (HIST is
the histogram directory):

    HIST/histograms.arrows          one finished histogram per row: slice_start, ch_num, state_label, n_records,
                                    counts, and its binning (e_lo, bin_width, slice_s, energy_col); slices end
                                    every slice_s and at each state change, so each is in one state
    HIST/histograms_current.arrows  the same for the slice still filling, rewritten every poll
    HIST/fits.arrows                one row per fit, with its plotm image (png)

    mass2-live-hist ANALYZED.arrows HIST --experiment-state pulses_experiment_state.txt --line MnKAlpha
"""

import time
from pathlib import Path

import polars as pl

import mass2

from .apply import apply_recipes
from .arrow_stream import ArrowStreamTailer, ArrowStreamWriter, write_stream_atomically
from .fit import LiveFitter, RoiFit
from .histogram import HistogramSpec, SlicedHistogrammer, slices_to_df
from .states import ExperimentStateFollower


MAX_RAW_BYTES = 20_000_000  # raw records read per pass of the loop; their analysis takes a few times this in memory
MAX_ANALYZED_BYTES = 20_000_000  # analyzed records read per pass of the loop


def run_live_apply(recipe_path: str | Path, raw_path: str | Path, analyzed_path: str | Path, poll_s: float = 0.5) -> None:
    """Follow RAW.arrows until its stream ends, writing the analyzed records to ANALYZED.arrows, which must not
    exist yet (FileExistsError)."""
    recipes = mass2.misc.unpickle_object(recipe_path)  # {ch_num: mass2 Recipe}, as saved by Channels.save_recipes
    reader = ArrowStreamTailer(Path(raw_path))
    with ArrowStreamWriter(Path(analyzed_path)) as writer:
        while not reader.ended:
            reader, raw_batches, caught_up = reader.poll(MAX_RAW_BYTES)  # read the new raw records
            if len(raw_batches) > 0:
                analyzed = apply_recipes(recipes, pl.concat(raw_batches))  # mass2 applies each channel's recipe
                writer.write(analyzed)  # write the analyzed records
            if caught_up:  # wait for the DAQ; when the read stopped at MAX_RAW_BYTES, go straight on
                time.sleep(poll_s)


def run_live_hist(
    analyzed_path: str | Path,
    hist_dir: str | Path,
    spec: HistogramSpec,
    energy_col: str,
    *,
    roi: RoiFit | None = None,
    experiment_state_path: str | Path | None = None,
    every_s: float = 10.0,
    poll_s: float = 0.5,
    grace_s: float = 1.0,
) -> None:
    """Follow ANALYZED.arrows until its stream ends, writing the finished histograms to HIST/histograms.arrows, the
    open ones to HIST/histograms_current.arrows, and the fits to HIST/fits.arrows (empty
    when `roi` is None).
    HIST/histograms.arrows and HIST/fits.arrows must not exist yet (FileExistsError)."""
    hist_dir = Path(hist_dir)
    states = ExperimentStateFollower(None if experiment_state_path is None else Path(experiment_state_path))
    histogrammer = SlicedHistogrammer(spec, energy_col, grace_s=grace_s)
    fitter = None if roi is None else LiveFitter(spec, roi, every_s)
    reader = ArrowStreamTailer(Path(analyzed_path))
    with ArrowStreamWriter(hist_dir / "histograms.arrows") as writer, ArrowStreamWriter(hist_dir / "fits.arrows") as fit_writer:
        while not reader.ended:
            reader, analyzed_batches, caught_up = reader.poll(MAX_ANALYZED_BYTES)  # read the new analyzed records
            states = states.poll()
            if len(analyzed_batches) > 0:
                histogrammer = histogrammer.add(states.label(pl.concat(analyzed_batches)))  # histogram the good ones
            histogrammer, finished = histogrammer.finish(reader.ended)  # the slices the data have moved past; at the end, all
            if len(finished) > 0:
                writer.write(slices_to_df(finished, spec, energy_col))  # write the finished histograms
                fitter = None if fitter is None else fitter.add(finished)
            write_stream_atomically(slices_to_df(list(histogrammer.open), spec, energy_col), hist_dir / "histograms_current.arrows")
            if fitter is not None and fitter.due(reader.ended):  # refit every `every_s` with new slices
                fitter, fit = fitter.fit()
                if fit is not None:
                    fit_writer.write(fit)  # write the fit
            if caught_up:  # wait for mass2-live-apply; when the read stopped at MAX_ANALYZED_BYTES, go straight on
                time.sleep(poll_s)
