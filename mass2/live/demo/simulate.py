"""Simulate a live data acquisition by replaying LJH pulse records into a growing Arrow IPC stream file.

All channels go into one stream, merged in time order, `chunk_size` records per record batch. The columns are

    chunk          UInt64   sequence number of the record batch, starting at 0
    ch_num         Int64    channel number
    timestamp      Datetime(us, UTC)
    subframecount  UInt64
    pulse          Array(UInt16, n_samples)

Records keep their original timestamps, and are written at a playback speed (a multiple of real time) that
can be changed while running through --speed-file. With `repeats` the data are replayed again after the end,
continuing the timeline. If the data have an experiment-state
file, its state changes are replayed on the same timeline into OUT_experiment_state.txt, each line appended
just before the first chunk that follows it, as DASTARD would. A `ScaledChannel` adds a fake channel
whose pulses are a source channel's pulses with the signal multiplied by a gain, as a detector of a different
gain would record them; it needs a recipe of its own to come out at the right energies.

Command line:  mass2-live-sim OUT.arrows [--ljh-folder DIR] [--repeats N] [--speed X] [--scale 4219:14219:1.03]
               [--no-states]
"""

import argparse
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TextIO

import numpy as np
import polars as pl
import pulsedata
from numpy.typing import NDArray

import mass2
from mass2.core import ljhutil
from .parent import exit_with_parent
from ..core.arrow_stream import ArrowStreamWriter


@dataclass(frozen=True)
class SimSource:
    """The raw records of one channel, held in memory for replay.

    A gain-scaled copy shares its source's `pulses` array and is scaled chunk by chunk as it is written, so
    an array of many copies costs no extra memory.
    """

    ch_num: int
    pulses: NDArray[np.uint16]  # shape (npulses, nsamples)
    timestamp_us: NDArray[np.int64]
    subframecount: NDArray[np.uint64]
    n_presamples: int
    gain: float = 1.0

    def records(self, rows: NDArray) -> NDArray[np.uint16]:
        """The pulse records at `rows`, with this channel's gain applied."""
        return self.pulses[rows] if self.gain == 1.0 else scale_pulses(self.pulses[rows], self.gain, self.n_presamples)


@dataclass(frozen=True)
class ScaledChannel:
    """A fake channel `new_ch` made from channel `source_ch` with its signal multiplied by `gain`."""

    source_ch: int
    new_ch: int
    gain: float

    @classmethod
    def parse(cls, text: str) -> "ScaledChannel":
        """Parse 'SOURCE:NEW:GAIN', e.g. '4219:14219:1.03'."""
        source, new, gain = text.split(":")
        return cls(int(source), int(new), float(gain))


def load_ljh_sources(pulse_folder: str | Path, max_pulses: int | None = None) -> dict[int, SimSource]:
    """Load every channel's pulses, timestamps and subframecounts from the LJH files in `pulse_folder`."""
    data = mass2.Channels.from_ljh_folder(pulse_folder)
    sources = {}
    for ch_num, ch in data.channels.items():
        assert ch.pulseframer is not None, f"channel {ch_num} has no raw pulses"
        n = ch.npulses if max_pulses is None else min(max_pulses, ch.npulses)
        sources[ch_num] = SimSource(
            ch_num=ch_num,
            pulses=ch.pulseframer.load_raw_chunk(0, n)["pulse"].to_numpy(),
            timestamp_us=ch.df["timestamp"].head(n).dt.epoch("us").to_numpy(),
            subframecount=ch.df["subframecount"].head(n).to_numpy(),
            n_presamples=ch.header.n_presamples,
        )
    return sources


def load_experiment_states(pulse_folder: str | Path) -> pl.DataFrame | None:
    """The experiment-state changes recorded alongside the LJH files in `pulse_folder`, or None if there is no file."""
    ljh = next(Path(pulse_folder).glob("*_chan*.ljh"))
    path = Path(ljhutil.experiment_state_path_from_ljh_path(ljh))
    return mass2.Channels({}, "").get_experiment_state_df(path) if path.exists() else None  # mass2 reads the file


def experiment_state_path(stream_path: str | Path) -> Path:
    """Where the simulator writes the experiment_state.txt for the stream at `stream_path`: <stem>_experiment_state.txt."""
    stream_path = Path(stream_path)
    return stream_path.with_name(f"{stream_path.stem}_experiment_state.txt")


def scale_pulses(pulses: NDArray, gain: float, n_presamples: int) -> NDArray[np.uint16]:
    """Multiply the signal of pulse records by `gain`, rounding and clipping to the uint16 range. Each record's
    pretrigger baseline is held fixed and only its deviations from it are multiplied: multiplying the raw samples
    would also move the pretrigger mean, which a drift correction would read as a large drift."""
    x = pulses.astype(np.float32)
    baseline = x[:, :n_presamples].mean(axis=1, keepdims=True)
    return np.clip(np.rint(baseline + gain * (x - baseline)), 0, np.iinfo(np.uint16).max).astype(np.uint16)


def simulate(
    sources: dict[int, SimSource],
    out_path: str | Path,
    *,
    chunk_size: int = 100,
    repeats: int = 1,
    scaled: Sequence[ScaledChannel] = (),
    speed: float = 5.0,
    speed_file: str | Path | None = None,
    pace: bool = True,
    states: pl.DataFrame | None = None,
) -> int:
    """Write the sources (plus any scaled channels) to an Arrow IPC stream; return the number of chunks written.

    Records keep their ORIGINAL timestamps, so every time and rate downstream is a real one. With `pace`, chunks
    are written at `speed` times real time: a chunk goes out once the wall clock reaches its last record's time,
    scaled by the speed. `speed_file`, if given, is a text file holding the speed, re-read before each chunk, so
    the speed can change while running. Without `pace`, everything is written as fast as possible. Repeats
    continue the timeline after the end of the data. `states` (timestamp, state_label) are replayed into
    `experiment_state_path(out_path)`, each change just before the first chunk after it.
    """
    sources = dict(sources)
    for s in scaled:
        sources[s.new_ch] = replace(sources[s.source_ch], ch_num=s.new_ch, gain=s.gain)

    # One replay cycle, merged across channels in time order (stable, so ties keep channel order).
    ch_nums = np.concatenate([np.full(len(s.timestamp_us), ch) for ch, s in sources.items()])
    rows = np.concatenate([np.arange(len(s.timestamp_us)) for s in sources.values()])
    t_us = np.concatenate([s.timestamp_us for s in sources.values()])
    sfc = np.concatenate([s.subframecount for s in sources.values()])
    order = np.argsort(t_us, kind="stable")
    ch_nums, rows, t_us, sfc = ch_nums[order], rows[order], t_us[order], sfc[order]
    cycle_us = int(t_us[-1] - t_us[0]) + 1000  # one cycle's duration, plus 1 ms so cycles don't overlap
    cycle_sfc = int(sfc.max() - sfc.min()) + 1
    no_states = pl.DataFrame(schema={"timestamp": pl.Datetime("us"), "state_label": pl.String})
    states = no_states if states is None else states
    state_us = np.clip(states["timestamp"].dt.epoch("us").to_numpy(), t_us[0], t_us[0] + cycle_us - 1)  # within a cycle
    labels = states["state_label"].cast(pl.String).to_list()
    pacer = _Pacer(float(speed), None if speed_file is None else Path(speed_file))

    nchunks, cycle = 0, 0
    with ArrowStreamWriter(Path(out_path)) as writer, _state_file(out_path, len(labels) > 0) as state_file:
        while repeats <= 0 or cycle < repeats:
            shift, k = cycle * cycle_us, 0  # k: the next state change of this cycle
            for lo in range(0, len(t_us), chunk_size):
                hi = min(lo + chunk_size, len(t_us))
                end_us = int(t_us[hi - 1]) + shift
                if pace:
                    pacer = pacer.wait_for(end_us)
                while state_file is not None and k < len(labels) and state_us[k] + shift <= end_us:
                    state_file.write(f"{(int(state_us[k]) + shift) * 1000}, {labels[k]}\n")  # before the records after it
                    state_file.flush()
                    k += 1
                sim_sfc = sfc[lo:hi] + np.uint64(cycle * cycle_sfc)
                writer.write(_chunk_frame(nchunks, sources, ch_nums[lo:hi], rows[lo:hi], t_us[lo:hi] + shift, sim_sfc))
                nchunks += 1
            cycle += 1
    return nchunks


@contextmanager
def _state_file(out_path: str | Path, wanted: bool) -> Iterator[TextIO | None]:
    """The DASTARD experiment_state.txt the simulator appends to (None when there are no states), flushed line by
    line so a follower sees each change at once."""
    if not wanted:
        yield None
        return
    with open(experiment_state_path(out_path), "w", encoding="utf-8") as f:
        f.write("# unix time in nanoseconds, state label\n")
        f.flush()
        yield f


@dataclass(frozen=True)
class _Pacer:
    """Holds back each chunk until the wall clock catches up with its data time at the current speed;
    `wait_for` returns the pacer to use next."""

    speed: float
    speed_file: Path | None
    data_anchor_us: float | None = None
    wall_anchor: float = 0.0

    def _read_speed(self) -> float:
        if self.speed_file is not None and self.speed_file.exists():
            try:
                return float(self.speed_file.read_text())
            except ValueError:
                pass  # a half-written file; keep the current speed
        return self.speed

    def wait_for(self, data_us: float) -> "_Pacer":
        p = self if self.data_anchor_us is not None else replace(self, data_anchor_us=data_us, wall_anchor=time.time())
        new_speed = p._read_speed()
        if new_speed != p.speed and new_speed > 0:
            # Re-anchor at the present position, so changing speed never jumps or stalls the data.
            now = time.time()
            assert p.data_anchor_us is not None
            p = replace(p, data_anchor_us=p.data_anchor_us + (now - p.wall_anchor) * p.speed * 1e6, wall_anchor=now, speed=new_speed)
        assert p.data_anchor_us is not None
        time.sleep(max(0.0, p.wall_anchor + (data_us - p.data_anchor_us) / 1e6 / p.speed - time.time()))
        return p


def _chunk_frame(
    chunk: int, sources: dict[int, SimSource], ch_nums: NDArray, rows: NDArray, t_us: NDArray, sfc: NDArray
) -> pl.DataFrame:
    """Build one chunk's DataFrame, gathering each row's pulse from its channel's array."""
    nsamples = next(iter(sources.values())).pulses.shape[1]
    pulses = np.empty((len(rows), nsamples), dtype=np.uint16)
    for ch in np.unique(ch_nums):
        here = ch_nums == ch
        pulses[here] = sources[int(ch)].records(rows[here])
    return pl.DataFrame({
        "chunk": pl.Series(np.full(len(rows), chunk, dtype=np.uint64)),
        "ch_num": pl.Series(ch_nums.astype(np.int64)),
        "timestamp": pl.Series(t_us.astype(np.int64)).cast(pl.Datetime("us", "UTC")),
        "subframecount": pl.Series(sfc.astype(np.uint64)),
        "pulse": pulses,
    })


def default_pulse_folder() -> Path:
    """The BESSY 2024-07-27 pulse data from the `pulsedata` package, also used by the mass2 tests."""
    return pulsedata.pulse_noise_ljh_pairs["bessy_20240727"].pulse_folder


def main(argv: Sequence[str] | None = None) -> None:
    """Entry point for `mass2-live-sim`."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("out", type=Path, help="Arrow IPC stream file to write (conventionally *.arrows)")
    p.add_argument(
        "--ljh-folder", type=Path, default=default_pulse_folder(), help="folder of LJH pulse files (default: pulsedata bessy_20240727)"
    )
    p.add_argument("--max-pulses", type=int, default=None, help="use at most this many records per channel")
    p.add_argument("--chunk-size", type=int, default=100, help="records per record batch (default 100)")
    p.add_argument("--repeats", type=int, default=1, help="replay the data this many times; 0 = forever (default 1)")
    p.add_argument("--speed", type=float, default=5.0, help="playback speed, in multiples of real time (default 5)")
    p.add_argument(
        "--speed-file", type=Path, default=None, help="a file holding the speed, re-read before each chunk to change it live"
    )
    p.add_argument("--as-fast-as-possible", action="store_true", help="write without pacing (timestamps are original either way)")
    p.add_argument(
        "--scale", action="append", default=[], type=ScaledChannel.parse, metavar="SRC:NEW:GAIN",
        help="add channel NEW made from channel SRC with its signal times GAIN; may repeat",
    )  # fmt: skip
    p.add_argument("--no-states", action="store_true", help="do not replay the experiment_state.txt file")
    args = p.parse_args(argv)
    exit_with_parent()

    sources = load_ljh_sources(args.ljh_folder, args.max_pulses)
    states = None if args.no_states else load_experiment_states(args.ljh_folder)
    print(f"mass2-live-sim: channels {sorted(sources)} + scaled {[s.new_ch for s in args.scale]} -> {args.out}", flush=True)
    try:
        n = simulate(
            sources, args.out, chunk_size=args.chunk_size, repeats=args.repeats, scaled=args.scale,
            speed=args.speed, speed_file=args.speed_file, pace=not args.as_fast_as_possible, states=states,
        )  # fmt: skip
        print(f"mass2-live-sim: finished, wrote {n} chunks", flush=True)
    except KeyboardInterrupt:
        print("mass2-live-sim: interrupted; stream closed", flush=True)


if __name__ == "__main__":
    main()
