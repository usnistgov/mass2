"""Experiment states for live data, from the DASTARD `*_experiment_state.txt` file while it grows.

mass2 reads the file (`Channels.get_experiment_state_df`); a record's state is the last change at or before its
timestamp, the backward as-of join of `Channel.with_experiment_state_df`.
"""

from dataclasses import dataclass, field, replace
from pathlib import Path

import polars as pl

import mass2

NO_STATE = "(none)"  # the state of a record before the first state change, or with no experiment_state.txt
STATE_SCHEMA = pl.Schema({"timestamp": pl.Datetime("us"), "state_label": pl.String()})


@dataclass(frozen=True)
class ExperimentStateFollower:
    """The state changes read so far from a growing experiment_state.txt; `poll` returns a new follower when the file has
    changed. `path=None` means there are no states."""

    path: Path | None
    size: int = -1
    states: pl.DataFrame = field(default_factory=lambda: pl.DataFrame(schema=STATE_SCHEMA))

    def poll(self) -> "ExperimentStateFollower":
        if self.path is None or not self.path.exists() or (size := self.path.stat().st_size) == self.size:
            return self
        try:
            states = mass2.Channels({}, "").get_experiment_state_df(self.path)
        except pl.exceptions.PolarsError:
            return self  # caught the DAQ halfway through a line: read it on the next poll
        return replace(self, size=size, states=states.with_columns(pl.col("state_label").cast(pl.String)))

    def label(self, df: pl.DataFrame) -> pl.DataFrame:
        """`df` with `state_label` and `state_start` (when that state began; null for NO_STATE) columns, in its
        original row order."""
        states = self.states.with_columns(pl.col("timestamp").dt.replace_time_zone(df["timestamp"].dtype.time_zone))  # type: ignore[attr-defined]
        states = states.with_columns(state_start=pl.col("timestamp"))
        labeled = df.with_row_index("_row").sort("timestamp").join_asof(states.sort("timestamp"), on="timestamp", strategy="backward")
        return labeled.sort("_row").drop("_row").with_columns(pl.col("state_label").fill_null(NO_STATE))
