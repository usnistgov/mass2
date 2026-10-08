"""Tests for the demo environment, which sits on top of the core and the viewer.

10. The simulator writes 100-record chunks with all channels and their original timestamps, replays the
   experiment states, scales copies, and follows a speed change while running.
11. The demo gives a visitor a run of their own, which switches dataset and changes speed from its page, and shows
    its files' columns and a few rows.
12. The shareable page embeds an exact recording of the real pipeline.
13. A copy recorded at a different gain comes out at its source channel's energies, through its own recipe.
14. The demo refuses playback speeds outside the viewer's range.
15. Each visitor gets a run of their own, up to a limit, and a run nobody views is ended.
16. A run whose tool fails starts over.
17. A tool exits when the demo that started it is gone.
"""

import json
import sys
import subprocess
import os
import math
import threading
import time
import urllib.error
import urllib.request

import numpy as np
import polars as pl
import pulsedata
import pytest

import mass2
from mass2.live.core.apply import apply_recipes
from mass2.live.demo import export, launcher, simulate
from mass2.live.demo.datasets import DATASETS
from mass2.live.demo.visitors import VisitorRuns
from mass2.live.viewer import server

PULSE_FOLDER = pulsedata.pulse_noise_ljh_pairs["bessy_20240727"].pulse_folder


def test_10_simulator_chunks_states_copies_and_speed_changes(tmp_path):
    sources = simulate.load_ljh_sources(PULSE_FOLDER, max_pulses=2000)
    path, speed_file = tmp_path / "pulses.arrows", tmp_path / "speed.txt"
    speed_file.write_text("1\n")  # real time: 2000 pulses would take minutes...
    run = threading.Thread(
        target=simulate.simulate, args=(sources, path),
        kwargs=dict(scaled=[simulate.ScaledChannel(4219, 99, 1.05)], speed_file=speed_file,
                    states=simulate.load_experiment_states(PULSE_FOLDER)),
    )  # fmt: skip
    run.start()
    time.sleep(1.0)
    speed_file.write_text("1000\n")  # ...so speed it up while it runs
    run.join(timeout=60)
    assert not run.is_alive()

    df = pl.read_ipc_stream(path)
    assert len(df) == 3 * 2000 and set(df["ch_num"]) == {4219, 4220, 99}
    assert (df.group_by("chunk").len()["len"] == 100).all()
    first = sources[4219].timestamp_us.min()
    assert df.filter(pl.col("ch_num") == 4219)["timestamp"].dt.epoch("us").min() == first  # original timestamps
    assert "CAL2" in simulate.experiment_state_path(path).read_text()
    original = np.stack(df.filter(pl.col("ch_num") == 4219)["pulse"].to_numpy()).astype(float)
    copy = np.stack(df.filter(pl.col("ch_num") == 99)["pulse"].to_numpy()).astype(float)
    height = lambda p: p.max(axis=1) - p[:, :200].mean(axis=1)  # noqa: E731
    assert np.median(height(copy) / height(original)) == pytest.approx(1.05, abs=0.002)


def test_11_the_demo_gives_a_visitor_a_run_that_switches_dataset_and_speed(tmp_path):
    controllers = []

    def new_controller(run_dir, store):
        controllers.append(launcher.DemoController(run_dir, store, sim_extra=["--max-pulses", "3000"]))
        return controllers[-1]

    runs = VisitorRuns(tmp_path, new_controller, max_runs=2, dataset="bessy_20240727")
    srv, port = server.start_router_server(runs, port=0)
    run_url = urllib.request.urlopen(f"http://127.0.0.1:{port}/").url  # the visitor is sent to their own run

    def post(path, body):
        urllib.request.urlopen(urllib.request.Request(run_url + path, data=json.dumps(body).encode(), method="POST"))

    def state_when(ok, timeout=120):
        t0 = time.time()
        while time.time() - t0 < timeout:
            s, _ = server.from_arrow_ipc(urllib.request.urlopen(f"{run_url}api/state").read())
            if ok(s):
                return s
            assert not controllers[0].failed()
            time.sleep(0.5)
        raise AssertionError("timed out")

    try:
        assert "/r/" in run_url and "<canvas" in urllib.request.urlopen(run_url).read().decode()
        post("api/speed", {"speed": 20})
        run_dir = controllers[0].workdir / "bessy_20240727"
        assert float((run_dir / "speed.txt").read_text()) == 20.0
        s = state_when(lambda s: s["status"] and s["status"]["records"] > 2000)
        assert len(s["meta"]["layout"]) == 16 and s["meta"]["layout"]["4224"]["gain"] == 1.03
        assert s["controller"]["speed"] == 20 and s["visitor"]["runs"] == 1
        peek = json.loads(urllib.request.urlopen(f"{run_url}api/peek?file=analyzed.arrows").read())  # the file popup
        names = [c["name"] for c in peek["columns"]]
        assert "energy_5lagy_best" in names and "pulse" not in names and 1 <= len(peek["rows"]) <= 3
        assert all(len(row) == len(names) for row in peek["rows"])
        with pytest.raises(urllib.error.HTTPError):  # only the run's own files
            urllib.request.urlopen(f"{run_url}api/peek?file=../../speed.txt")

        post("api/dataset", {"key": "20230626"})
        s = state_when(lambda s: s["meta"] and s["meta"]["e_hi"] == 10000 and s["status"] and s["status"]["records"] > 0)
        assert s["controller"]["active"] == "20230626"
    finally:
        runs.stop_all()
        srv.shutdown()


def test_12_shareable_page_holds_an_exact_recording(tmp_path):
    dataset = DATASETS["gamma_20241005"]  # the smallest dataset
    rec = export.record(dataset, tmp_path / "gamma")
    results = pl.read_ipc_stream(tmp_path / "gamma" / "analyzed.arrows")
    good = results.filter(pl.col("good"), pl.col(dataset.energy_col).is_between(dataset.spec.e_lo, dataset.spec.e_hi, closed="left"))
    recorded = sum(sum(pairs[1::2]) for *_, sparse in rec["slices"] for chans in sparse.values() for pairs in chans.values())
    assert recorded == len(good)
    assert sum(r for _, r, _ in rec["slices"]) == len(results)
    assert rec["fits"]["fits"] and rec["fits"]["fits"][-1]["png"].startswith("data:image/png;base64,")

    page = tmp_path / "replay.html"
    export.write_replay_page([rec], page)
    embedded = page.read_text().split("window.MASS2_REPLAY = ", 1)[1].split(";</script>", 1)[0]
    assert json.loads(embedded)["datasets"][0]["key"] == "gamma_20241005"


MN = DATASETS["20230626"]


def test_13_a_copy_at_another_gain_matches_its_source_through_its_own_recipe(tmp_path):
    copy = next(p for p in MN.pixels if p.source_ch is not None and p.gain != 1.0)
    sources = simulate.load_ljh_sources(MN.pulse_folder, max_pulses=1500)
    path = tmp_path / "pulses.arrows"
    simulate.simulate(sources, path, pace=False, scaled=[simulate.ScaledChannel(copy.source_ch, copy.ch_num, copy.gain)])
    raw = pl.read_ipc_stream(path)
    recipes = mass2.misc.unpickle_object(MN.recipe_path)
    assert recipes[copy.ch_num] is not recipes[copy.source_ch]  # learned from the copy's own scaled pulses

    def energies(recipes: dict, ch: int) -> np.ndarray:
        return apply_recipes(recipes, raw).filter(pl.col("ch_num") == ch).sort("subframecount")[MN.energy_col].to_numpy()

    source, own = energies(recipes, copy.source_ch), energies(recipes, copy.ch_num)
    borrowed = energies({copy.ch_num: recipes[copy.source_ch]}, copy.ch_num)
    near_mn = (source > 5800) & (source < 6000)
    assert near_mn.sum() > 100
    assert np.median(np.abs(own[near_mn] - source[near_mn])) < 1.0  # eV
    assert abs(np.median(borrowed[near_mn] / source[near_mn]) - 1) > 0.01  # the source's recipe would be off by about the gain


def test_14_the_demo_refuses_speeds_outside_the_viewers_range(tmp_path):
    controller = launcher.DemoController(tmp_path, server.HistogramStore(tmp_path / "hist"))
    for ok in [1, 5, 30]:
        controller.set_speed(ok)
        assert controller.speed == ok
    for bad in [0, 0.5, -5, 31, 600, 1e308, math.inf, math.nan]:
        with pytest.raises(ValueError):
            controller.set_speed(bad)
    assert controller.speed == 30


class _FakeController:
    """Stands in for DemoController: records what it is asked to do, starts no processes."""

    def __init__(self, run_dir, store):
        self.run_dir, self.store, self.speed, self.active, self.stopped, self.maintained = run_dir, store, 5.0, None, False, 0

    def switch(self, key):
        self.active = key

    def describe(self):
        return {"active": self.active}

    @staticmethod
    def pids():
        return {}

    def maintain(self, max_gb):
        self.maintained += 1

    def stop(self):
        self.stopped = True


def test_15_each_visitor_gets_a_run_of_their_own(tmp_path):
    runs = VisitorRuns(tmp_path, _FakeController, max_runs=2, dataset="bessy_20240727", idle_s=60)
    a, b, c = runs.route("/").redirect, runs.route("/").redirect, runs.route("/").redirect
    assert len({a, b, c}) == 3 and a.startswith("/r/") and a.endswith("/")
    assert runs.route(a).site is None and runs.route(a).path == "/" and runs.runs == {}  # the page alone starts no run
    site_a, site_b = runs.route(a + "api/state").site, runs.route(b + "api/state").site
    assert runs.route(c + "api/state").busy  # a third visitor waits
    assert site_a is not site_b and site_a.controller.active == "bessy_20240727"
    assert runs.route(a + "api/state").path == "/api/state" and runs.route(a + "arrow.js").path == "/arrow.js"
    assert runs.route(a.rstrip("/")).redirect == a and runs.route("/r/nothex/").site is None

    runs.last_seen[b.split("/")[2]] -= 120  # b's page has been closed for two minutes
    runs.tick(max_gb=4)
    assert site_b.controller.stopped and not site_a.controller.stopped and site_a.controller.maintained == 1
    assert runs.route(c + "api/state").site is not None  # a slot is free again
    revived = runs.route(b + "api/state").site  # b's old address now starts a new run (or waits when all are in use)
    assert revived is None or revived is not site_b
    runs.stop_all()
    assert site_a.controller.stopped and runs.runs == {}


def test_16_a_run_whose_tool_fails_starts_over(tmp_path, monkeypatch):
    controller = launcher.DemoController(tmp_path, server.HistogramStore(tmp_path / "hist"))
    starts = []
    monkeypatch.setattr(controller, "switch", starts.append)
    controller.active = "20230626"
    monkeypatch.setattr(controller, "failed", lambda: ["mass2-live-apply"])
    for _ in range(5):
        controller.maintain(max_gb=4)
    assert starts == ["20230626"] * 3  # three quick restarts, then at most one a minute


def test_17_a_tool_exits_when_the_demo_that_started_it_is_gone(tmp_path):
    tool, demo = tmp_path / "tool.py", tmp_path / "demo.py"  # a stand-in demo that starts one tool and is then killed
    tool.write_text("import time\nfrom mass2.live.demo.parent import exit_with_parent\nexit_with_parent(0.05)\ntime.sleep(30)\n")
    demo.write_text(
        "import os, subprocess, sys, time\n"
        f"p = subprocess.Popen([sys.executable, {str(tool)!r}], env=os.environ | {{'MASS2_LIVE_PARENT_PID': str(os.getpid())}})\n"
        "print(p.pid, flush=True)\ntime.sleep(30)\n"
    )
    parent = subprocess.Popen([sys.executable, str(demo)], stdout=subprocess.PIPE, text=True)
    child = int(parent.stdout.readline())
    parent.kill()
    parent.wait()
    for _ in range(100):
        try:
            os.kill(child, 0)
        except ProcessLookupError:
            break
        time.sleep(0.05)
    else:
        os.kill(child, 9)
        raise AssertionError("the tool outlived its demo")
