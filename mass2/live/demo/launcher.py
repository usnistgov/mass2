"""Run the whole live pipeline on `pulsedata` datasets, a run for each visitor, switchable from the viewer.

Each visitor gets a run of their own (see `visitors.py`). For the run's dataset (see `datasets.py`):
1. Start `mass2-live-sim`, replaying the LJH data (and its experiment states) into a growing stream, plus
   gain-shifted fake channels.
2. Start `mass2-live-apply` on that stream with the dataset's saved recipes (mass2/live/demo/recipes/<key>.pkl),
   one for every channel of the array: copies at a different gain have recipes learned from their own pulses.
3. Start `mass2-live-hist` on the analyzed records, refitting the line the original analysis fitted.
The viewer is served from this process. Picking another dataset in the page stops the run's tools and starts
them on it. A run plays its dataset once and then ends, keeping what it showed (`--repeats N` replays the data N
times, continuing the timeline; 0 forever). A change of playback speed never interrupts it; only when its files
reach `--max-gb` does it start over.

Each pipeline tool runs as its own process, exactly as it would from the command line, and exits if the demo
is gone. A run whose tool fails is started over. Ctrl-C stops them all.

Command line:  mass2-live-demo [WORKDIR] [--dataset bessy_20240727] [--speed 5] [--repeats 1] [--max-gb 4]
                               [--max-runs 8] [--idle 180] [--port 8765] [--lan] [--no-browser] [--public]

Everything is local by default. With --public it is also served on the internet through a Cloudflare quick
tunnel, if cloudflared is installed, and the public address is printed.
"""

import argparse
import functools
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import webbrowser
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import IO
from pathlib import Path

import mass2

from .datasets import DATASETS, DemoDataset
from .simulate import experiment_state_path
from .parent import PARENT_ENV
from ..viewer.server import HistogramStore, start_router_server, viewer_urls
from .visitors import VisitorRuns

MIN_SPEED, MAX_SPEED = 1.0, 30.0  # 1x to 30x real time: a laptop's pipeline keeps up with about 35x of BESSY


@dataclass
class DemoController:
    """Owns one run's simulator, applier and histogrammer processes, for one dataset at a time (so it is not frozen)."""

    workdir: Path
    store: HistogramStore
    repeats: int = 1
    sim_extra: Sequence[str] = ()  # e.g. ["--max-pulses", "2000"] in tests
    speed: float = 5.0  # playback speed, multiples of real time; the simulator re-reads it before every chunk
    active: str | None = None
    phase: str = "starting"
    _procs: dict[str, subprocess.Popen] = field(default_factory=dict, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _restarts: list[float] = field(default_factory=list, repr=False)  # times the run was restarted after a failure
    _checked: float = 0.0

    def describe(self) -> dict:
        return {
            "datasets": [{"key": d.key, "title": d.title} for d in DATASETS.values()],
            "active": self.active,
            "phase": self.phase,
            "speed": self.speed,
            "dataset_meta": {} if self.active is None else viewer_meta(self.active),
        }

    def set_speed(self, speed: float) -> None:
        """Change the playback speed of the running simulator (it picks this up before its next chunk).
        Raises ValueError outside the viewer's own range, so no request can make the simulator write without pause."""
        if not MIN_SPEED <= float(speed) <= MAX_SPEED:
            raise ValueError(f"speed must be between {MIN_SPEED} and {MAX_SPEED}")
        self.speed = float(speed)
        if self.active is not None:
            self._write_speed(self.workdir / self.active)

    def _write_speed(self, run_dir: Path) -> None:
        tmp = run_dir / "speed.txt.tmp"
        tmp.write_text(f"{self.speed}\n")
        tmp.replace(run_dir / "speed.txt")

    def switch(self, key: str) -> None:
        """Stop the current pipeline and start `key`'s. Raises KeyError if unknown. The states the viewer turned
        off stay off (the store keeps them)."""
        dataset = DATASETS[key]
        with self._lock:
            self._stop_procs()
            run_dir = self.workdir / key
            self.active = key
            run_dir.mkdir(parents=True, exist_ok=True)
            raw = run_dir / "pulses.arrows"
            for stale in [raw, experiment_state_path(raw), run_dir / "analyzed.arrows", *(run_dir / "hist").rglob("*.*")]:
                stale.unlink(missing_ok=True)  # never let the applier or viewer pick up a previous run
            self.store.reset(run_dir / "hist")
            self._write_speed(run_dir)
            self._procs = pipeline_processes(dataset, run_dir, self.repeats, self.sim_extra)
            self.phase = "running"

    def pids(self) -> dict[str, int]:
        """The running pipeline processes, by name."""
        with self._lock:
            return {name: proc.pid for name, proc in self._procs.items() if proc.poll() is None}

    def run_bytes(self) -> int:
        """Bytes in the running dataset's files."""
        if self.active is None:
            return 0
        return sum(p.stat().st_size for p in (self.workdir / self.active).rglob("*") if p.is_file())

    def maintain(self, max_gb: float) -> None:
        """Keep the run going; call it often. Starts the run over when a tool has failed (at most 3 times in 5
        minutes, then once a minute) or when its files reach `max_gb`. A run whose passes are done stays as it is."""
        if self.active is None:
            return
        now = time.time()
        if failed := self.failed():
            self._restarts = [t for t in self._restarts if now - t < 300]
            if len(self._restarts) >= 3 and now - self._restarts[-1] < 60:
                return
            print(f"mass2-live-demo: {', '.join(failed)} failed; starting the run over", flush=True)
            self._restarts.append(now)
            self.switch(self.active)
        elif now - self._checked > 10:
            self._checked = now
            if self.run_bytes() > max_gb * 1e9:
                self.switch(self.active)  # start over, so the run's files never outgrow max_gb

    def failed(self) -> list[str]:
        """Names of pipeline tools that exited with an error."""
        with self._lock:
            return [name for name, proc in self._procs.items() if proc.poll() not in {None, 0}]

    def stop(self) -> None:
        with self._lock:
            self._stop_procs()

    def _stop_procs(self) -> None:
        for proc in self._procs.values():
            proc.terminate()
        for proc in self._procs.values():
            proc.wait()
        self._procs = {}


def source_bytes(dataset: DemoDataset) -> dict[str, int]:
    """Sizes of the files a run starts from: the LJH files and the saved recipe."""
    files = [*Path(dataset.pulse_folder).glob("*_chan*.ljh"), dataset.recipe_path]
    return {p.name: p.stat().st_size for p in files}


@functools.cache
def viewer_meta(key: str) -> dict:
    """What the viewer shows about a demo dataset beyond the histogram files: the detector array, the recipe, the files."""
    dataset = DATASETS[key]
    recipes = mass2.misc.unpickle_object(dataset.recipe_path)
    return {
        "layout": {str(ch): pos for ch, pos in dataset.layout.items()},
        "recipe_steps": [type(step).__name__ for step in next(iter(recipes.values()))],
        "recipe_channels": sorted(recipes),
        "bin_source": dataset.bin_source,
        "recipe_file": dataset.recipe_path.name,
        "source_bytes": source_bytes(dataset),  # the LJH files and the recipe, by name
    }


def _tool(entry: str) -> list[str]:
    """The command running one of the core's command lines (core/cli.py) in this Python, set to exit when this
    demo is gone."""
    start = (
        f"from mass2.live.demo.parent import exit_with_parent; exit_with_parent(); from mass2.live.core.cli import {entry}; {entry}()"
    )
    return [sys.executable, "-c", start]


def pipeline_processes(dataset: DemoDataset, run_dir: Path, repeats: int, sim_extra: Sequence[str]) -> dict[str, subprocess.Popen]:
    """Start the simulator, the core loop, and the histogram loop (with its line fit) for `dataset`, all under `run_dir`."""
    raw, analyzed, hist, spec, roi = (
        run_dir / "pulses.arrows",
        run_dir / "analyzed.arrows",
        run_dir / "hist",
        dataset.spec,
        dataset.roi,
    )
    sim = [sys.executable, "-m", "mass2.live.demo.simulate", str(raw), "--ljh-folder", str(dataset.pulse_folder)]
    sim += ["--speed-file", str(run_dir / "speed.txt"), "--repeats", str(repeats), *sim_extra]
    sim += [f"--scale={s.source_ch}:{s.new_ch}:{s.gain}" for s in dataset.scales]
    apply = [*_tool("apply_main"), str(dataset.recipe_path), str(raw), str(analyzed)]
    histo = [*_tool("hist_main"), str(analyzed), str(hist), "--experiment-state", str(experiment_state_path(raw))]
    histo += ["--energy-col", dataset.energy_col, "--slice", str(spec.slice_s), "--e-lo", str(spec.e_lo), "--e-hi", str(spec.e_hi)]
    histo += [
        "--bin",
        str(spec.bin_width),
        "--line",
        str(roi.line),
        "--dlo",
        str(roi.dlo),
        "--dhi",
        str(roi.dhi),
        "--source",
        roi.source,
    ]
    env = os.environ | {PARENT_ENV: str(os.getpid())}  # each tool exits if this process is gone
    return {
        "mass2-live-sim": subprocess.Popen(sim, env=env),
        "mass2-live-apply": subprocess.Popen(apply, env=env),
        "mass2-live-hist": subprocess.Popen(histo, env=env),
    }


def public_tunnel(port: int) -> subprocess.Popen:
    """Start a Cloudflare quick tunnel to the local viewer and print its public address (a new one each time)."""
    proc = subprocess.Popen(
        ["cloudflared", "tunnel", "--no-autoupdate", "--protocol", "http2", "--url", f"http://127.0.0.1:{port}"],
        stderr=subprocess.PIPE,
        text=True,
    )

    def follow(stream: IO[str]) -> None:  # print the address once, and keep reading so cloudflared never blocks
        shown = False
        for line in stream:
            if not shown and (m := re.search(r"https://[-a-z0-9]+\.trycloudflare\.com", line)):
                print(f"mass2-live-demo: public address {m.group(0)}", flush=True)
                shown = True

    assert proc.stderr is not None
    threading.Thread(target=follow, args=(proc.stderr,), daemon=True).start()
    return proc


def main(argv: Sequence[str] | None = None) -> None:
    """Entry point for `mass2-live-demo`."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "workdir", type=Path, nargs="?", default=Path("mass2_live_demo"), help="output directory (default ./mass2_live_demo)"
    )
    p.add_argument("--dataset", choices=sorted(DATASETS), default="bessy_20240727", help="dataset a new run starts with")
    p.add_argument("--speed", type=float, default=5.0, help="playback speed a new run starts at, multiples of real time (default 5)")
    p.add_argument("--repeats", type=int, default=1, help="passes over the data in a run, then it ends (default 1; 0: no end)")
    p.add_argument("--max-gb", type=float, default=4.0, help="start a run over when its files reach this size, GB (default 4)")
    p.add_argument("--max-runs", type=int, default=8, help="runs at once, one per visitor (default 8)")
    p.add_argument("--idle", type=float, default=180, help="stop a run nobody has viewed for this long, seconds (default 180)")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--lan", action="store_true", help="serve the viewer to other devices on this network, e.g. a phone")
    p.add_argument("--no-browser", action="store_true")
    p.add_argument(
        "--public", action="store_true", help="also serve it on the internet: a Cloudflare quick tunnel (needs cloudflared)"
    )
    args = p.parse_args(argv)
    if args.public and shutil.which("cloudflared") is None:
        p.error(
            "--public needs cloudflared on the PATH (https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/)"
        )

    args.workdir.mkdir(parents=True, exist_ok=True)
    runs = VisitorRuns(
        args.workdir, lambda run_dir, store: DemoController(run_dir, store, args.repeats, speed=args.speed),
        max_runs=args.max_runs, dataset=args.dataset, idle_s=args.idle,
    )  # fmt: skip
    host = "0.0.0.0" if args.lan else "127.0.0.1"
    server, port = start_router_server(runs, args.port, host)
    urls = viewer_urls(host, port)
    print(f"mass2-live-demo: viewer at {'  '.join(urls)}  (Ctrl-C to stop)", flush=True)
    tunnel = public_tunnel(port) if args.public else None
    if not args.no_browser:
        webbrowser.open(urls[-1])

    # SIGTERM (e.g. from `kill` or `timeout`) must clean up the children exactly as Ctrl-C does.
    signal.signal(signal.SIGTERM, signal.default_int_handler)
    try:
        while True:
            runs.tick(args.max_gb)
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        if tunnel is not None:
            tunnel.terminate()
        runs.stop_all()
        server.shutdown()


if __name__ == "__main__":
    main()
