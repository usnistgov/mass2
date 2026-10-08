"""Energy histograms of good pulses, per channel and experiment state, in time slices (mass2's `hist_of_series`).
Slices end at multiples of `slice_s` since the Unix epoch and wherever the experiment state changes, so each slice
is in one state, and the slice starts say when each state began. A slice is "open" until the data have advanced
`grace_s` past its `slice_s` boundary; then it is finalized. Records for an already-finalized slice (more than
`grace_s` late) are dropped. Like mass2's `Channel`, the histogrammer is a frozen dataclass whose `add`
and `finish` return a new one. The loop that uses it is run_live_hist in loop.py.
"""

from dataclasses import dataclass, field, replace

import numpy as np
import polars as pl
from numpy.typing import NDArray

from mass2.core.misc import hist_of_series


@dataclass(frozen=True)
class HistogramSpec:
    """Energy binning and slice length shared by every histogram."""

    e_lo: float = 0.0
    e_hi: float = 1200.0
    bin_width: float = 1.0
    slice_s: float = 10.0

    @property
    def nbins(self) -> int:
        return int(round((self.e_hi - self.e_lo) / self.bin_width))

    @property
    def bin_edges(self) -> NDArray:
        return self.e_lo + self.bin_width * np.arange(self.nbins + 1)

    @property
    def slice_us(self) -> int:
        return int(round(self.slice_s * 1e6))

    def boundary_after(self, t_us: int) -> int:
        """The first `slice_s` boundary after `t_us`."""
        return (t_us // self.slice_us + 1) * self.slice_us

    def to_dict(self) -> dict:
        return {"e_lo": self.e_lo, "e_hi": self.e_hi, "bin_width": self.bin_width, "slice_s": self.slice_s, "nbins": self.nbins}


@dataclass(frozen=True)
class HistogramSlice:
    """Per (channel, state), the counts of good, in-range records and the number of all records, for one time slice,
    in one state, starting at `start_us` (µs since the epoch, UTC)."""

    start_us: int
    nbins: int
    counts: dict[tuple[int, str], NDArray[np.int64]] = field(default_factory=dict)
    records: dict[tuple[int, str], int] = field(default_factory=dict)

    def total(self) -> int:
        return int(sum(c.sum() for c in self.counts.values()))

    def summed(self) -> NDArray[np.int64]:
        """The counts summed over channels and states."""
        return np.sum([np.zeros(self.nbins, np.int64), *self.counts.values()], axis=0)

    def by_state(self) -> dict[str, NDArray[np.int64]]:
        """The counts summed over channels, by state."""
        out: dict[str, NDArray[np.int64]] = {}
        for (_, state), c in self.counts.items():
            out[state] = out[state] + c if state in out else c
        return out

    def plus(self, other: "HistogramSlice") -> "HistogramSlice":
        """This slice with `other`'s counts and records added."""
        counts, records = dict(self.counts), dict(self.records)
        for key, c in other.counts.items():
            counts[key] = counts[key] + c if key in counts else c
            records[key] = records.get(key, 0) + other.records.get(key, 0)
        return replace(self, counts=counts, records=records)


@dataclass(frozen=True)
class SlicedHistogrammer:
    """Good-pulse energies accumulated into histograms per (channel, state), one per time slice. Like mass2's
    `Channel`, `add` and `finish` return a new histogrammer."""

    spec: HistogramSpec
    energy_col: str
    grace_s: float = 1.0
    open: tuple[HistogramSlice, ...] = ()  # the slices still accumulating, oldest first; normally just one
    finalized_before_us: int = -1  # every slice starting earlier than this is finalized
    latest_us: int | None = None

    def add(self, df: pl.DataFrame) -> "SlicedHistogrammer":
        """Count the good, in-range records of `df` (with `ch_num`, `timestamp`, `good`, the energy, and the
        `state_label` and `state_start` of `ExperimentStateFollower.label`). Every (channel, state) with a record in
        a slice gets a histogram, with zero counts when none of its records are good and in range."""
        if len(df) == 0:
            return self
        spec, t_us = self.spec, pl.col("timestamp").dt.epoch("us")
        latest = df.select(t_us.max()).item()
        boundary = (t_us // spec.slice_us) * spec.slice_us
        df = df.with_columns(
            slice_start=pl.max_horizontal(boundary, pl.col("state_start").dt.epoch("us").fill_null(boundary)),
            in_range=pl.col("good") & pl.col(self.energy_col).is_between(spec.e_lo, spec.e_hi, closed="left"),
        )
        late = df["slice_start"] < self.finalized_before_us
        slices = {s.start_us: s for s in self.open}
        for (start, ch_num, state), rows in df.filter(~late).group_by("slice_start", "ch_num", "state_label"):
            _, counts = hist_of_series(rows.filter("in_range")[self.energy_col], spec.bin_edges)
            new = HistogramSlice(start, spec.nbins, {(ch_num, state): counts.astype(np.int64)}, {(ch_num, state): len(rows)})
            slices[start] = slices.get(start, HistogramSlice(start, spec.nbins)).plus(new)
        latest_us = latest if self.latest_us is None else max(self.latest_us, latest)
        return replace(self, open=tuple(slices[k] for k in sorted(slices)), latest_us=latest_us)

    def finish(self, ended: bool) -> tuple["SlicedHistogrammer", list[HistogramSlice]]:
        """Finalize the open slices the data have advanced `grace_s` past, or, when the stream has `ended`, all of
        them. Returns the new histogrammer and the finalized slices, oldest first."""

        def is_done(s: HistogramSlice) -> bool:
            past_us = self.spec.boundary_after(s.start_us) + int(self.grace_s * 1e6)
            return ended or (self.latest_us is not None and past_us <= self.latest_us)

        done = [s for s in self.open if is_done(s)]
        if len(done) == 0:
            return self, []
        rest = tuple(s for s in self.open if not is_done(s))
        return replace(self, open=rest, finalized_before_us=self.spec.boundary_after(done[-1].start_us)), done


def slices_to_df(slices: list[HistogramSlice], spec: HistogramSpec, energy_col: str) -> pl.DataFrame:
    """One row per (slice, channel, state): slice_start, ch_num, state_label, n_records (all records, good or not),
    counts (of the good, in-range ones), and the binning, e_lo, bin_width, slice_s and energy_col (the same on
    every row, so the file explains itself)."""
    rows = [
        (s.start_us, ch, st, s.records.get((ch, st), 0), c.astype(np.uint32))
        for s in slices
        for (ch, st), c in sorted(s.counts.items())
    ]
    return pl.DataFrame({
        "slice_start": pl.Series([r[0] for r in rows], dtype=pl.Int64).cast(pl.Datetime("us", "UTC")),
        "ch_num": pl.Series([r[1] for r in rows], dtype=pl.Int64),
        "state_label": pl.Series([r[2] for r in rows], dtype=pl.String),
        "n_records": pl.Series([r[3] for r in rows], dtype=pl.Int64),
        "counts": pl.Series([r[4] for r in rows], dtype=pl.List(pl.UInt32)),
    }).with_columns(
        e_lo=pl.lit(spec.e_lo), bin_width=pl.lit(spec.bin_width), slice_s=pl.lit(spec.slice_s), energy_col=pl.lit(energy_col)
    )


def spec_of(df: pl.DataFrame) -> HistogramSpec:
    """The binning of histogram rows, from the first (`df` must not be empty)."""
    row = df.row(0, named=True)
    return HistogramSpec(row["e_lo"], row["e_lo"] + len(row["counts"]) * row["bin_width"], row["bin_width"], row["slice_s"])


def df_to_slices(df: pl.DataFrame) -> list[HistogramSlice]:
    """Inverse of `slices_to_df`, oldest slice first."""
    by_start: dict[int, HistogramSlice] = {}
    rows = df.select(pl.col("slice_start").dt.epoch("us"), "ch_num", "state_label", "n_records", "counts").iter_rows()
    for start, ch_num, state, n, counts in rows:
        new = HistogramSlice(start, len(counts), {(ch_num, state): np.asarray(counts, dtype=np.int64)}, {(ch_num, state): n})
        by_start[start] = by_start[start].plus(new) if start in by_start else new
    return [by_start[start] for start in sorted(by_start)]
