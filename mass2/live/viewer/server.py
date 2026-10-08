"""Serve a live web view of the histograms written by `mass2-live-hist`.

GET  /                     the viewer page
GET  /api/state?row_s=L    an Arrow IPC stream of what the page draws: the totals by channel and state, and the newest
                           time-plot rows, up to L s long, by state; everything else the page shows (meta, status,
                           states, fits, ...) as JSON in its schema metadata
GET  /arrow.js             Apache Arrow's JavaScript library, which the page reads the stream with
GET  /fits/latest.png      the plotm image of the latest line fit
GET  /api/peek?file=NAME   one of the run's files (RUN_FILES) as the page shows it: its columns with their types, and
                           a few rows of its first batch, long values cut short
POST /api/states           {"hidden": [LABEL, ...]} the states the spectrum and the line fit leave out, for every viewer
                           of the run (kept here only)
POST /api/dataset          {"key": NAME} switch dataset   } only when run by mass2-live-demo,
POST /api/speed            {"speed": X} playback speed  } which owns the simulator

Under mass2-live-demo each run's pages are these under /r/<id>/, and / sends a new visitor to a run of their own.

Every refresh gets everything the page draws, the same size however long the run has been going; the
slices themselves stay in this process and in histograms.arrows. `run` changes whenever the
histogram directory is reset (e.g. a dataset switch), telling the browser to drop what it holds.

Command line:  mass2-live-view HIST_DIR [--port 8765] [--lan]
"""

import argparse
import gzip
import json
import os
import socket
import subprocess
import threading
import time
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from pathlib import Path
from collections.abc import Callable
from typing import Any, Protocol, TypeVar
from urllib.parse import parse_qs, urlparse

import numpy as np
import polars as pl
import pyarrow as pa
from numpy.typing import NDArray

from ..core.cli import parse_line
from ..core.fit import RoiFit, fit_line
from ..core.arrow_stream import ArrowStreamTailer
from ..core.histogram import HistogramSlice, HistogramSpec, df_to_slices, spec_of
from ..core.states import NO_STATE


ARROW_JS = "apache-arrow-21.2.0.es2015.min.js"  # Apache Arrow's JavaScript build, from npm, served at /arrow.js
RUN_FILES = [
    "pulses.arrows", "pulses_experiment_state.txt", "analyzed.arrows", "hist/histograms.arrows", "hist/histograms_current.arrows",
    "hist/fits.arrows",
]  # fmt: skip
PEEK_ROWS = 3  # rows of a file the page's file view shows
ROWS = 120  # time-plot rows a page is sent: more than its canvas shows (at most ~105)
KEEP_SLICES = 20_000  # slices kept, summed over channels, for time-plot rows: a few hundred MB at most


class Controller(Protocol):
    """What the viewer needs from whoever runs the pipeline, to offer a dataset switcher."""

    def describe(self) -> dict: ...  # {"datasets": [{"key", "title"}], "active": key, "phase": text}

    def switch(self, key: str) -> None: ...

    def set_speed(self, speed: float) -> None: ...

    def pids(self) -> dict[str, int]: ...  # the pipeline processes it runs, by name

    def maintain(self, max_gb: float) -> None: ...  # keep the run going (called often)

    def stop(self) -> None: ...


Sparse = tuple[NDArray[np.int32], NDArray[np.int32]]  # (nonempty bins, their counts)


@dataclass
class HistogramStore:
    """Follows the histogram files in `hist_dir`, keeping what the page draws: the totals of every finished
    slice by channel and state, and for the time plot the newest `KEEP_SLICES` slices summed over channels,
    by state, nonempty bins only. The full record of every slice is histograms.arrows. Pages read it from
    several threads while it follows a growing file, so it is not frozen.

    The states the pages turned off (`hidden`) live here only. The line fit the page shows is this store's, of the
    states shown, of the line mass2-live-hist fits (from the first row of fits.arrows), with the same `fit_line`: at
    once when the hidden states change, else every `every_s` with new slices."""

    hist_dir: Path
    run: int = 0  # changes whenever the store starts over, telling pages to drop what they hold
    spec: HistogramSpec | None = None  # the binning, from the first histogram rows
    energy_col: str = ""
    n_slices: int = 0
    total: HistogramSlice | None = None  # every finished slice added up: counts and records by channel and state
    history: deque[tuple[int, dict[str, Sparse]]] = field(default_factory=lambda: deque(maxlen=KEEP_SLICES))  # (start µs, by state)
    hidden: list[str] = field(default_factory=list)  # the states the pages turned off; every other state shows
    roi: tuple[RoiFit, float] | None = None  # the line mass2-live-hist fits, and how often (s)
    fits: list[dict] = field(default_factory=list)  # this store's fits since `hidden` last changed, without their images
    latest_png: bytes | None = None
    _fitted: tuple[list[str] | None, float, int] = (None, 0.0, -1)  # the hidden states, time and slices of the last fit
    _tailer: ArrowStreamTailer = field(init=False, repr=False)
    _fit_tailer: ArrowStreamTailer = field(init=False, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def __post_init__(self) -> None:
        self.reset(self.hist_dir)

    def reset(self, hist_dir: str | Path) -> None:
        """Forget everything but the hidden states, and follow `hist_dir` from the start."""
        with self._lock:
            self.hist_dir = Path(hist_dir)
            self._tailer = ArrowStreamTailer(self.hist_dir / "histograms.arrows")
            self._fit_tailer = ArrowStreamTailer(self.hist_dir / "fits.arrows")
            self.spec, self.n_slices, self.total = None, 0, None
            self.roi, self.fits, self.latest_png, self._fitted = None, [], None, (None, 0.0, -1)
            self.history = deque(maxlen=KEEP_SLICES)
            self.run += 1

    def _learn_spec(self, df: pl.DataFrame) -> None:
        """The binning, from the first histogram rows seen."""
        if self.spec is None and len(df) > 0:
            self.spec, self.energy_col = spec_of(df), df["energy_col"][0]

    def _update(self) -> None:
        if self.roi is None:  # the line, from mass2-live-hist's first fit
            self._fit_tailer, fit_frames, _ = self._fit_tailer.poll(1)
            if len(fit_frames) > 0 and len(fit_frames[0]) > 0:
                row = fit_frames[0].row(0, named=True)
                self.roi = RoiFit(parse_line(row["line"]), row["dlo"], row["dhi"], row["source"]), row["every_s"]
        self._tailer, frames, _ = self._tailer.poll(64_000_000)  # the rest on the next request
        for df in frames:
            self._learn_spec(df)
            for s in df_to_slices(df):
                self.total = s if self.total is None else self.total.plus(s)
                summed = s.by_state()
                if len(self.history) > 0 and self.history[-1][0] == s.start_us:  # a slice can arrive in more than one batch
                    for st, (bins, counts) in self.history.pop()[1].items():
                        np.add.at(summed.setdefault(st, np.zeros(s.nbins, np.int64)), bins, counts)
                else:
                    self.n_slices += 1
                self.history.append((s.start_us, {st: _sparse(c) for st, c in summed.items()}))

    def state(self, row_s: float | None = None) -> tuple[dict, list["CountRow"]]:
        """Everything the page draws: a small dict (run, meta, status, states, fits, ...) and the counts, as
        `CountRow`s: the totals by channel and state, and the newest `ROWS` rows of the time plot, `row_s`
        long, by state, summed over channels. Both include the slice still filling. What a page is sent does
        not grow with the length of the run."""
        with self._lock:
            self._update()
            info: dict[str, Any] = {
                "run": self.run,
                "meta": None,
                "status": None,
                "states": [],
                "n_slices": self.n_slices,
                "hidden_states": self.hidden,
                "file_bytes": self.file_bytes(),
            }
            current = _read_if_exists(self.hist_dir / "histograms_current.arrows", pl.read_ipc_stream, pl.DataFrame())
            self._learn_spec(current)
            if self.spec is None:
                return info, []
            spec, total, open_slices = self.spec, self.total, df_to_slices(current)
            for s in open_slices:
                total = s if total is None else total.plus(s)
            if total is None:
                return info, []
            row_us = max(1, round((spec.slice_s if row_s is None else row_s) / spec.slice_s)) * spec.slice_us
            history = [*self.history, *((s.start_us, {st: _sparse(c) for st, c in s.by_state().items()}) for s in open_slices)]
            latest_us = spec.boundary_after(history[-1][0])  # the newest slice's end
            self._refit(spec, total, latest_us)
            return info | {
                "meta": spec.to_dict() | {"energy_col": self.energy_col},
                "row_s": row_us / 1e6,
                "open_good": sum(s.total() for s in open_slices),
                "status": {
                    "records": sum(total.records.values()),
                    "good_pulses": total.total(),
                    "first_data_us": total.start_us,
                    "latest_data_us": latest_us,
                    "stream_ended": self._tailer.ended,
                    "updated_unix_s": _read_if_exists(self.hist_dir / "histograms_current.arrows", lambda p: p.stat().st_mtime, None),
                },
                "states": _state_changes(history),
                "fits": self._fit_series(),
            }, [
                *(CountRow.dense("total", None, st, ch, c) for (ch, st), c in total.counts.items()),
                *(CountRow.dense("row", t / 1e6, st, 0, c) for t, st, c in _time_rows(history, row_us, spec.nbins)),
            ]

    def _refit(self, spec: HistogramSpec, total: HistogramSlice, latest_us: int) -> None:
        """Fit the line on the totals of the states shown, when the hidden states have changed (starting a new
        series) or `every_s` has passed with new slices."""
        if self.roi is None:
            return
        (roi, every_s), (hidden, at, slices) = self.roi, self._fitted
        if self.hidden == hidden and (time.time() - at < every_s or self.n_slices == slices):
            return
        if self.hidden != hidden:
            self.fits = []
        self._fitted = (list(self.hidden), time.time(), self.n_slices)
        counts = np.sum([np.zeros(spec.nbins), *(c for (_, st), c in total.counts.items() if st not in self.hidden)], axis=0)
        which = "all states" if len(self.hidden) == 0 else f"all states but {', '.join(self.hidden)}"
        row = fit_line(counts, spec, roi, f"{roi.label}, all channels, {which}, {self.n_slices} slices")
        if row is not None:
            self.latest_png = row.pop("png")
            self.fits.append(row | {"t": latest_us / 1e6, "slices": self.n_slices, "every_s": every_s})

    def _fit_series(self) -> dict | None:
        """The fits since the hidden states last changed, for the page."""
        if self.roi is None or len(self.fits) == 0:
            return None
        roi, every_s = self.roi
        return {
            "roi": {"label": roi.label, "source": roi.source},
            "every_s": every_s,
            "hidden_states": self._fitted[0],
            "fits": self.fits,
        }

    def file_bytes(self) -> dict[str, int]:
        """Sizes of the run's files that exist, by path relative to the run directory (the histogram directory's parent)."""
        run_dir, out = self.hist_dir.parent, {}
        for name in RUN_FILES:
            try:
                out[name] = (run_dir / name).stat().st_size
            except FileNotFoundError:
                pass
        return out

    def peek(self, name: str) -> dict:
        """The run's file `name` (one of RUN_FILES) for the page: {"columns": [{"name", "type"}], "rows": [[text]]}:
        PEEK_ROWS rows, from the start, middle and end of an Arrow IPC stream's first record batch (the stream is read
        no further), or the first lines of a text file; long lists and
        binary values are cut short. Raises KeyError for a name not in RUN_FILES, FileNotFoundError or
        StopIteration when there is nothing to show yet."""
        if name not in RUN_FILES:
            raise KeyError(name)
        path = self.hist_dir.parent / name
        if path.suffix == ".txt":  # the DASTARD experiment_state.txt: a "# a, b" header line, then "a, b" lines
            lines = path.read_text().splitlines()
            header = lines[0].lstrip("# ").split(", ")
            return {
                "columns": [{"name": h, "type": "text"} for h in header],
                "rows": [ln.split(", ") for ln in lines[1 : 1 + PEEK_ROWS]],
            }
        with pa.OSFile(str(path)) as f:
            batch = pa.ipc.open_stream(f).read_next_batch()
        columns = [{"name": fld.name, "type": str(fld.type).replace("item: ", "")} for fld in batch.schema]
        n = batch.num_rows
        picks = sorted({round(i * (n - 1) / (PEEK_ROWS - 1)) for i in range(PEEK_ROWS)}) if n > 0 else []
        return {"columns": columns, "rows": [[_cell(v) for v in row.values()] for row in batch.take(picks).to_pylist()]}

    def hide_states(self, states: object) -> None:
        """Set the states the spectrum and the fit leave out, for every viewer: a list of labels (empty for none).
        Raises ValueError for anything else."""
        if not (isinstance(states, list) and len(states) <= 64 and all(isinstance(x, str) and len(x) <= 64 for x in states)):
            raise ValueError("hidden must be a list of state labels")
        with self._lock:
            self.hidden = sorted(set(states))


@dataclass(frozen=True)
class CountRow:
    """One histogram on the wire: `kind` is "total" (the whole run, by channel) or "row" (a time-plot row,
    summed over channels, ch_num 0, `t` its start in seconds); `bins` its nonempty bins, `counts` their counts."""

    kind: str
    t: float | None
    state: str
    ch_num: int
    bins: NDArray[np.int32]
    counts: NDArray[np.int32]

    @classmethod
    def dense(cls, kind: str, t: float | None, state: str, ch_num: int, counts: NDArray) -> "CountRow":
        return cls(kind, t, state, ch_num, *_sparse(counts))


def _time_rows(history: list[tuple[int, dict[str, Sparse]]], row_us: int, nbins: int) -> list[tuple[int, str, NDArray]]:
    """The newest `ROWS` rows of the time plot, oldest first, as (start µs, state, counts summed over channels). A
    row starts at each state change and every `row_us` within a state, so a row is in one state; a state's last
    row, if shorter than half of `row_us`, joins the row before it."""
    starts: list[tuple[int, str, int]] = []  # (start µs, state, index in history of the row's first slice)
    for i, (start_us, by_state) in enumerate(history):
        state = next(iter(by_state))
        if len(starts) > 0 and state == starts[-1][1] and start_us < starts[-1][0] + row_us:
            continue
        if len(starts) > 1 and state != starts[-1][1] == starts[-2][1] and start_us - starts[-1][0] < row_us // 2:
            starts.pop()  # the state's short last row joins the row before
        starts.append((start_us, state, i))
    keep = starts[-ROWS:]
    out = []
    for (t, state, i), j in zip(keep, [i for _, _, i in keep[1:]] + [len(history)]):
        counts = np.zeros(nbins, np.int64)
        for _, by_state in history[i:j]:
            np.add.at(counts, *by_state[state])
        out.append((t, state, counts))
    return out


def _cell(v: object) -> str:
    """One value as the page's file view shows it: long lists as their first values and length, binary as its size."""
    if v is None:
        return "null"
    if isinstance(v, bytes):
        kind = "PNG image" if v.startswith(b"\x89PNG") else "binary"
        return f"{kind}, {len(v) / 1000:.1f} kB"
    if isinstance(v, list):
        shown = ", ".join(_cell(x) for x in v[:4])
        return f"[{shown}]" if len(v) <= 4 else f"[{shown}, … ({len(v):,} values)]"
    if isinstance(v, float):
        return f"{v:.6g}"
    return str(v)


def _sparse(counts: NDArray) -> Sparse:
    nz = np.flatnonzero(counts).astype(np.int32)
    return nz, counts[nz].astype(np.int32)


WIRE_SCHEMA = pa.schema([
    ("kind", pa.dictionary(pa.int8(), pa.utf8())),
    ("t", pa.float64()),
    ("state", pa.dictionary(pa.int16(), pa.utf8())),
    ("ch_num", pa.int32()),
    ("bins", pa.list_(pa.uint16())),  # the nonempty bins only
    ("counts", pa.list_(pa.int32())),
])  # fmt: skip


def to_arrow_ipc(info: dict, rows: list[CountRow]) -> bytes:
    """An Arrow IPC stream of `rows` (see WIRE_SCHEMA), with `info` as JSON in the schema metadata."""
    offsets = pa.array(np.concatenate([[0], np.cumsum([len(r.bins) for r in rows])]).astype(np.int32))
    bins = np.concatenate([np.zeros(0, np.uint16), *(r.bins for r in rows)]).astype(np.uint16)
    counts = np.concatenate([np.zeros(0, np.int32), *(r.counts for r in rows)]).astype(np.int32)
    table = pa.table(
        [
            pa.array([r.kind for r in rows], pa.utf8()).dictionary_encode().cast(WIRE_SCHEMA.field("kind").type),
            pa.array([r.t for r in rows], pa.float64()),
            pa.array([r.state for r in rows], pa.utf8()).dictionary_encode().cast(WIRE_SCHEMA.field("state").type),
            pa.array([r.ch_num for r in rows], pa.int32()),
            pa.ListArray.from_arrays(offsets, pa.array(bins)),
            pa.ListArray.from_arrays(offsets, pa.array(counts)),
        ],
        schema=WIRE_SCHEMA.with_metadata({"mass2.live": json.dumps(info)}),
    )
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    return sink.getvalue().to_pybytes()


def from_arrow_ipc(data: bytes) -> tuple[dict, pa.Table]:
    """The reverse of `to_arrow_ipc`: the info dict and the table."""
    table = pa.ipc.open_stream(data).read_all()
    return json.loads(table.schema.metadata[b"mass2.live"]), table


@dataclass
class MemorySampler:
    """The resident memory (RSS) of this process and of the controller's processes, read with `ps` (macOS and
    Linux) when a page asks, at most every `every_s` seconds, so a page can show what each process uses."""

    name: str
    controller: "Controller | None"
    every_s: float = 2.0
    _latest: list[dict] = field(default_factory=list, repr=False)
    _at: float = 0.0

    @property
    def latest(self) -> list[dict]:
        if time.time() - self._at > self.every_s:
            self._latest, self._at = (
                process_memory({self.name: os.getpid()} | ({} if self.controller is None else self.controller.pids())),
                time.time(),
            )
        return self._latest


def process_memory(pids: dict[str, int]) -> list[dict]:
    """[{"name", "pid", "rss_bytes"}] for those of `pids` still running."""
    try:
        out = subprocess.run(
            ["ps", "-o", "pid=,rss=", "-p", ",".join(map(str, pids.values()))], capture_output=True, text=True, timeout=5, check=False
        ).stdout
    except (OSError, subprocess.TimeoutExpired):
        return []
    rss = {int(pid): 1024 * int(kb) for pid, kb in (line.split() for line in out.splitlines() if line.strip())}  # ps reports KiB
    return [{"name": name, "pid": pid, "rss_bytes": rss[pid]} for name, pid in pids.items() if pid in rss]


T = TypeVar("T")


def _state_changes(history: list[tuple[int, dict[str, Sparse]]]) -> list[list]:
    """[[seconds since the epoch, label], ...] of each state change: a slice is in one state, so a slice whose
    state differs from the one before starts the new state."""
    out, last = [], NO_STATE
    for start_us, by_state in history:
        for state in by_state:
            if state not in {last, NO_STATE}:
                out.append([start_us / 1e6, state])
            last = state
    return out


def _read_if_exists(path: Path, read: Callable[[Path], T], default: T) -> T:
    """Read a file another process replaces atomically; it may not exist yet."""
    try:
        return read(path)
    except FileNotFoundError:
        return default


@dataclass(frozen=True)
class Site:
    """One run as the viewer serves it: its histograms, who runs its pipeline (if anyone), its memory report,
    and anything else its pages should be told."""

    store: HistogramStore
    controller: Controller | None
    memory: MemorySampler
    extra: Callable[[], dict] = dict


@dataclass(frozen=True)
class Route:
    """Where a request goes: a site and the path within it; or a redirect; or "busy" (no run for it now)."""

    site: Site | None
    path: str
    redirect: str | None = None
    busy: bool = False


class Router(Protocol):
    def route(self, path: str) -> Route: ...


@dataclass(frozen=True)
class OneSite:
    """Every request goes to the one run."""

    site: Site

    def route(self, path: str) -> Route:
        return Route(self.site, path)


def _handler_for(router: Router) -> type[BaseHTTPRequestHandler]:
    page = resources.files("mass2.live.viewer").joinpath("viewer.html").read_bytes()
    arrow_js = resources.files("mass2.live.viewer").joinpath(ARROW_JS).read_bytes()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            url = urlparse(self.path)
            r = router.route(url.path)
            if r.redirect:
                self.send_response(303)
                self.send_header("Location", r.redirect)
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                return
            if r.path == "/arrow.js":
                self._send(arrow_js, "text/javascript; charset=utf-8")
                return
            if r.path == "/":
                self._send(page, "text/html; charset=utf-8")
                return
            if r.busy:
                self.send_error(503, "every run is in use")
                return
            if r.site is None:
                self.send_error(404)
                return
            store, controller = r.site.store, r.site.controller
            if r.path == "/fits/latest.png":
                png = store.latest_png
                if png is None:
                    self.send_error(404)
                else:
                    self._send(png, "image/png")
            elif r.path == "/api/peek":
                try:
                    peek = store.peek(parse_qs(url.query).get("file", [""])[0])
                except KeyError:
                    self.send_error(404)
                    return
                except (StopIteration, pa.ArrowInvalid, OSError):  # not written yet, or not a whole batch yet
                    peek = {"columns": [], "rows": [], "note": "nothing written yet"}
                self._send(json.dumps(peek).encode(), "application/json")
            elif r.path == "/api/state":
                row_s = float(parse_qs(url.query).get("row_s", ["0"])[0])
                info, rows = store.state(row_s if row_s > 0 else None)
                info["controller"] = None if controller is None else controller.describe()
                if info["controller"] is not None and info["meta"] is not None:  # a demo adds what it knows: the array, the recipe
                    info["meta"] |= info["controller"].pop("dataset_meta", {})
                info["processes"] = r.site.memory.latest
                info |= r.site.extra()
                self._send(to_arrow_ipc(info, rows), "application/vnd.apache.arrow.stream")
            else:
                self.send_error(404)

        def do_POST(self) -> None:
            r = router.route(urlparse(self.path).path)
            if r.busy:
                self.send_error(503, "every run is in use")
                return
            if r.site is None or r.path not in {"/api/states", "/api/dataset", "/api/speed"}:
                self.send_error(404)
                return
            try:
                body = self.rfile.read(min(int(self.headers.get("Content-Length", 0)), 65536))
                reply = _carry_out(r.site, r.path, json.loads(body) if len(body) > 0 else {})
            except (KeyError, ValueError, TypeError, AttributeError):
                self.send_error(400, "bad request")
                return
            self._send(reply, "application/json")

        def _send(self, body: bytes, content_type: str, status: int = 200) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            if content_type != "image/png" and "gzip" in self.headers.get("Accept-Encoding", ""):
                body = gzip.compress(body, compresslevel=5)  # the counts are mostly digits: several times smaller
                self.send_header("Content-Encoding", "gzip")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args: object) -> None:
            pass  # keep the terminal for the tools' own output

    return Handler


def _carry_out(site: Site, path: str, request: dict) -> bytes:
    """Do what a page asked (POST `path` with JSON `request`) and return the reply; KeyError, ValueError or
    TypeError for a request that cannot be done."""
    if path == "/api/states":
        site.store.hide_states(request["hidden"])
        return b"{}"
    if site.controller is None:
        raise KeyError(path)  # only a demo can switch datasets or change speed
    if path == "/api/dataset":
        site.controller.switch(request["key"])
    else:
        site.controller.set_speed(float(request["speed"]))
    return json.dumps(site.controller.describe()).encode()


def start_server(store: HistogramStore, port: int = 8765, host: str = "127.0.0.1") -> tuple[ThreadingHTTPServer, int]:
    """Serve one run, with no demo controlling it, in a background thread. Returns the server and the actual port
    (useful with port=0)."""
    return start_router_server(OneSite(Site(store, None, MemorySampler("mass2-live-view", None))), port, host)


def start_router_server(router: Router, port: int = 8765, host: str = "127.0.0.1") -> tuple[ThreadingHTTPServer, int]:
    """Serve whatever `router` sends each request to, in a background thread."""
    server = ThreadingHTTPServer((host, port), _handler_for(router))
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, server.server_address[1]


def lan_address() -> str:
    """This machine's address on its local network, for opening the viewer from a phone."""
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        try:
            s.connect(("10.255.255.255", 1))  # no packet is sent; this only picks the outgoing interface
            return s.getsockname()[0]
        except OSError:
            return "127.0.0.1"


def viewer_urls(host: str, port: int) -> list[str]:
    """URLs to print for a server bound to `host`."""
    if host == "0.0.0.0":
        return [f"http://{lan_address()}:{port}/", f"http://127.0.0.1:{port}/"]
    return [f"http://{host}:{port}/"]


def main(argv: Sequence[str] | None = None) -> None:
    """Entry point for `mass2-live-view`."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("hist_dir", type=Path, help="histogram directory written by mass2-live-hist")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--lan", action="store_true", help="serve to other devices on this network, e.g. a phone")
    args = p.parse_args(argv)
    host = "0.0.0.0" if args.lan else "127.0.0.1"
    server, port = start_server(HistogramStore(args.hist_dir), args.port, host)
    print(f"mass2-live-view: {'  '.join(viewer_urls(host, port))}", flush=True)
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        server.shutdown()


if __name__ == "__main__":
    main()
