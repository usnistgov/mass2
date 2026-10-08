"""Tests for the core of mass2.live: what a real instrument would run.

1. Stream files are standard Arrow IPC, and can be followed while they are written.
2. Applying a saved recipe live gives exactly what mass2 gives offline.
3. Every good pulse lands in exactly one time slice, and output chunks match input chunks.
4. The live line fit finds a line the way Channel.linefit would.
5. The line fit sums every state, a state that begins later too, and refits only with new slices.
"""

import threading
from dataclasses import replace

import numpy as np
import polars as pl
import pulsedata
import pytest

import mass2
from mass2.live.core.fit import LiveFitter
from mass2.live.core.apply import apply_recipes
from mass2.live.core.arrow_stream import ArrowStreamTailer, ArrowStreamWriter
from mass2.live.core.histogram import HistogramSlice, HistogramSpec, df_to_slices, spec_of
from mass2.live.core.loop import run_live_apply, run_live_hist
from mass2.live.core.states import NO_STATE, ExperimentStateFollower
from mass2.live.demo import simulate
from mass2.live.demo.datasets import DATASETS

BESSY = DATASETS["bessy_20240727"]
PULSE_FOLDER = pulsedata.pulse_noise_ljh_pairs["bessy_20240727"].pulse_folder


@pytest.fixture(scope="module")
def bessy():
    """The first 2000 pulses per channel of the BESSY run, its saved recipes, and its experiment states."""
    sources = simulate.load_ljh_sources(PULSE_FOLDER, max_pulses=2000)
    return sources, mass2.misc.unpickle_object(BESSY.recipe_path), simulate.load_experiment_states(PULSE_FOLDER)


def test_1_stream_files_are_standard_and_can_be_read_while_growing(tmp_path):
    path = tmp_path / "s.arrows"
    batch = pl.DataFrame({"i": [1, 2, 3], "pulse": np.ones((3, 4), dtype=np.uint16)})
    reader = ArrowStreamTailer(path)
    reader, frames, caught_up = reader.poll(max_bytes=1_000_000)
    assert frames == []  # the file does not exist yet: nothing to read, no error

    writer = ArrowStreamWriter(path)
    for _ in range(3):
        writer.write(batch)
    complete = path.read_bytes()
    path.write_bytes(complete[:-5])  # pretend we caught the writer halfway through its third batch
    reader, frames, caught_up = reader.poll(max_bytes=1)  # a budget smaller than one batch still reads one batch...
    assert len(frames) == 1 and not caught_up  # ...and says more is waiting
    reader, frames, caught_up = reader.poll(max_bytes=1_000_000)
    assert len(frames) == 1 and caught_up  # only the complete batch: the third is still being written
    path.write_bytes(complete)
    reader, frames, caught_up = reader.poll(max_bytes=1_000_000)
    assert len(frames) == 1  # and the rest arrives on the next poll
    writer.close()
    reader, frames, caught_up = reader.poll(max_bytes=1_000_000)
    assert frames == [] and reader.ended

    assert len(pl.read_ipc_stream(path)) == 9  # any ordinary Arrow reader accepts the finished file
    with pytest.raises(FileExistsError):
        ArrowStreamWriter(path)  # an existing file is never overwritten


def test_2_live_recipe_matches_offline_mass2(tmp_path, bessy):
    """The core promise: chunk-by-chunk live results equal offline `Channels.load_recipes` results."""
    sources, recipes, state_log = bessy
    path = tmp_path / "pulses.arrows"
    simulate.simulate(sources, path, pace=False, states=state_log)
    labeled = ExperimentStateFollower(simulate.experiment_state_path(path)).poll().label(pl.read_ipc_stream(path))
    live = apply_recipes(recipes, labeled)

    offline = mass2.Channels.from_ljh_folder(PULSE_FOLDER).with_experiment_state_by_path().load_recipes(str(BESSY.recipe_path))
    for ch_num, ch in offline.channels.items():
        expect = ch.df.head(2000)
        found = live.filter(pl.col("ch_num") == ch_num).sort("subframecount")
        assert np.allclose(found["energy_5lagy_best"], expect["energy_5lagy_best"], equal_nan=True)
        assert (found["good"] == expect.select(ch.good_expr.alias("good"))["good"]).all()
        assert (found["state_label"] == expect["state_label"].cast(pl.String).fill_null(NO_STATE)).all()
        assert (found["timestamp"].dt.epoch("us") == expect["timestamp"].dt.epoch("us")).all()  # original timestamps


def test_3_every_good_pulse_is_histogrammed_once_and_chunks_match(tmp_path, bessy):
    sources, _, state_log = bessy
    stream, out, hist = tmp_path / "pulses.arrows", tmp_path / "out.arrows", tmp_path / "hist"
    spec = HistogramSpec(e_lo=0, e_hi=1200, bin_width=0.25, slice_s=10)
    writer = threading.Thread(target=simulate.simulate, args=(sources, stream), kwargs=dict(repeats=2, speed=400, states=state_log))
    writer.start()  # the loop follows the file while it is being written
    run_live_apply(BESSY.recipe_path, stream, out, poll_s=0.05)
    writer.join()
    run_live_hist(
        out,
        hist,
        spec,
        "energy_5lagy_best",
        roi=BESSY.roi,
        experiment_state_path=simulate.experiment_state_path(stream),
        poll_s=0.05,
        grace_s=0.1,
    )

    raw, results = pl.read_ipc_stream(stream), pl.read_ipc_stream(out)
    assert results["chunk"].to_list() == raw["chunk"].to_list()  # same rows, same order, same chunk numbers
    assert results["ch_num"].to_list() == raw["ch_num"].to_list()

    histograms = pl.read_ipc_stream(hist / "histograms.arrows")
    assert spec_of(histograms) == spec and (histograms["energy_col"] == "energy_5lagy_best").all()  # each row has its binning
    slices = df_to_slices(histograms)
    good = results.filter(pl.col("good"), pl.col("energy_5lagy_best").is_between(0, 1200, closed="left"))
    assert sum(s.total() for s in slices) == len(good)
    assert histograms["n_records"].sum() == len(raw)  # every record is in one histogram's n_records, good or not
    fits = pl.read_ipc_stream(hist / "fits.arrows")  # the histogram loop also refits the line
    assert len(fits) > 0 and fits["png"][-1].startswith(b"\x89PNG")


def test_4_line_fit_finds_the_line():
    """Given a histogram with a 6 eV wide line at 600 eV, the live fitter fits it like Channel.linefit would."""
    spec = HistogramSpec(e_lo=0, e_hi=1200, bin_width=0.25, slice_s=10)
    energies = np.random.default_rng(0).normal(600, 6 / 2.355, 20000)
    counts = np.histogram(energies, bins=spec.nbins, range=(0, 1200))[0]
    _, fit = LiveFitter(spec, BESSY.roi).add([HistogramSlice(0, spec.nbins, {(1, "CAL2"): counts})]).fit()
    entry = fit.row(0, named=True)
    assert entry["peak"] == pytest.approx(600, abs=0.1)
    assert entry["fwhm"] == pytest.approx(6, rel=0.05)
    assert entry["png"].startswith(b"\x89PNG")  # drawn by LineModelResult.plotm()


def test_5_the_fit_sums_every_state_and_refits_on_new_slices():
    spec = HistogramSpec(e_lo=0, e_hi=1200, bin_width=0.25, slice_s=10)
    rng = np.random.default_rng(0)

    def line(e0):  # a 6 eV wide line at e0
        return np.histogram(rng.normal(e0, 6 / 2.355, 20000), bins=spec.nbins, range=(0, 1200))[0]

    both = LiveFitter(spec, DATASETS["bessy_20240727"].roi).add([
        HistogramSlice(0, spec.nbins, {(1, "CAL2"): line(600), (1, "SCAN3"): line(603)})
    ])
    both, fit = both.fit()
    assert fit["counts_in_roi"][0] > 2 * 20000 * 0.9 and 600 < fit["peak"][0] < 603  # both states, summed
    assert not both.due(ended=False) and both.due(ended=True)  # just fitted, no new slices; but the end refits
    later = both.add([HistogramSlice(10_000_000, spec.nbins, {(1, "SCAN4"): line(600)})])  # a state that begins later
    assert not later.due(ended=False)  # new slices, but every_s has not passed
    _, fit = replace(later, last_fit_s=0.0).fit()
    assert fit["counts_in_roi"][0] > 3 * 20000 * 0.9
