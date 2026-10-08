"""Give each visitor a run of their own.

Opening the viewer's address sends the visitor to an address of their own, /r/<id>/. The run itself (its own
simulator, applier and histogrammer, in its own folder) starts with the page's first request for data, so a
link preview or crawler, which fetches the page but runs no JavaScript, never starts one. Everything the page
does there (dataset, playback speed, states) affects only that run; sending someone the address shows them the
same run. A run nobody has viewed for `idle_s` seconds is stopped and its folder deleted. When `max_runs` are
already running, the page says so and keeps asking until one is free.
"""

import re
import secrets
import shutil
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from ..viewer.server import Controller, HistogramStore, MemorySampler, Route, Site

RUN_PATH = re.compile(r"^/r/([0-9a-f]{12})(/.*)?$")


@dataclass(frozen=True)
class VisitorRun:
    site: Site
    controller: Controller
    run_dir: Path


@dataclass
class VisitorRuns:
    """The runs, by id, and when each was last viewed; a `Router` for the viewer server. It starts and ends
    runs as visitors come and go, so it is not frozen. `new_controller(run_dir, store)` makes what runs a new
    run's pipeline (a `DemoController`)."""

    workdir: Path
    new_controller: Callable[[Path, HistogramStore], Controller]
    max_runs: int
    dataset: str
    idle_s: float = 180.0
    runs: dict[str, VisitorRun] = field(default_factory=dict)
    last_seen: dict[str, float] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def __post_init__(self) -> None:
        self.workdir = Path(self.workdir)
        shutil.rmtree(self.workdir / "runs", ignore_errors=True)  # runs from an earlier server are nobody's now

    def route(self, path: str) -> Route:
        if path == "/":
            return Route(None, path, redirect=f"/r/{secrets.token_hex(6)}/")
        if path.endswith("/arrow.js"):
            return Route(None, "/arrow.js")
        m = RUN_PATH.match(path)
        if m is None:
            return Route(None, path)
        run_id, rest = m.groups()
        if rest is None:
            return Route(None, path, redirect=f"/r/{run_id}/")
        if rest == "/":
            return Route(None, "/")  # the page alone; its first request for data starts the run
        run = self._run(run_id)  # an address whose run has ended starts a new one under the same address
        if run is None:
            return Route(None, rest, busy=True)
        self.last_seen[run_id] = time.time()
        return Route(run.site, rest)

    def _run(self, run_id: str) -> VisitorRun | None:
        """The run `run_id`, started if need be; None when every run is in use."""
        with self._lock:
            if run_id in self.runs:
                return self.runs[run_id]
            if len(self.runs) >= self.max_runs:
                return None
            run_dir = self.workdir / "runs" / run_id
            store = HistogramStore(run_dir / self.dataset / "hist")
            controller = self.new_controller(run_dir, store)
            memory = MemorySampler("mass2-live-demo (serves every run)", controller)
            run = VisitorRun(Site(store, controller, memory, extra=lambda: {"visitor": self.describe()}), controller, run_dir)
            self.runs[run_id], self.last_seen[run_id] = run, time.time()
        controller.switch(self.dataset)
        return run

    def describe(self) -> dict:
        return {"runs": len(self.runs), "max_runs": self.max_runs, "idle_s": self.idle_s}

    def tick(self, max_gb: float) -> None:
        """Keep every run going (see `DemoController.maintain`) and end the runs nobody is viewing."""
        now = time.time()
        with self._lock:
            idle = [rid for rid in self.runs if now - self.last_seen[rid] > self.idle_s]
            ended = [self.runs.pop(rid) for rid in idle]
            for rid in idle:
                del self.last_seen[rid]
            running = list(self.runs.values())
        for run in ended:
            _end(run)
        for run in running:
            run.controller.maintain(max_gb)

    def stop_all(self) -> None:
        with self._lock:
            ended, self.runs, self.last_seen = list(self.runs.values()), {}, {}
        for run in ended:
            _end(run)


def _end(run: VisitorRun) -> None:
    """Stop a run's tools and delete its folder."""
    run.controller.stop()
    shutil.rmtree(run.run_dir, ignore_errors=True)
