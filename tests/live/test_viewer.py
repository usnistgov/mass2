"""Tests for the viewer server: what a page is sent.

6. A page is sent exactly what it draws: the run's totals and the newest time-plot rows.
7. The histograms survive the trip through Arrow IPC.
8. The server keeps the states turned off in the viewer in memory, for every viewer of the run, and refuses bad ones.
9. The page is told the resident memory of each process involved.
"""

import os
import subprocess
import sys
import time

import numpy as np
import polars as pl
import pytest

from mass2.live.core.arrow_stream import ArrowStreamWriter, write_stream_atomically
from mass2.live.core.histogram import HistogramSlice, HistogramSpec, slices_to_df
from mass2.live.viewer import server


def _store_with_slices(tmp_path, spec: HistogramSpec, n: int) -> tuple[server.HistogramStore, list[HistogramSlice]]:
    """A histogram directory holding `n` random slices in two states and three channels, and a store following it."""
    rng = np.random.default_rng(1)
    hist = tmp_path / "hist"
    hist.mkdir()
    slices = []
    for k in range(n):
        state = "CAL" if (k // 7) % 2 else "SCAN"
        counts = {(ch, state): rng.poisson(0.05, spec.nbins).astype(np.int64) for ch in (1, 2, 3)}
        slices.append(HistogramSlice(1_700_000_000_000_000 + k * spec.slice_us, spec.nbins, counts))
    with ArrowStreamWriter(hist / "histograms.arrows") as w:
        for k in range(0, n, 5):
            w.write(slices_to_df(slices[k : k + 5], spec, "energy"))
    return server.HistogramStore(hist), slices


def _decode(rows: pl.DataFrame, nbins: int) -> dict[tuple, np.ndarray]:
    out = {}
    for r in rows.iter_rows(named=True):
        a = np.zeros(nbins, np.int64)
        np.add.at(a, np.asarray(r["bins"], dtype=np.int64), np.asarray(r["counts"], dtype=np.int64))
        key = (r["kind"], r["t"], r["state"], r["ch_num"])
        out[key] = out.get(key, 0) + a
    return out


@pytest.mark.parametrize("row_slices", [1, 4, 6, 25])  # states change every 7 slices: at 6, a 1-slice last row joins the one before
def test_6_the_page_is_sent_exactly_what_it_draws(tmp_path, row_slices):
    spec = HistogramSpec(e_lo=0, e_hi=200, bin_width=0.5, slice_s=10)
    store, slices = _store_with_slices(tmp_path, spec, 203)
    opening = HistogramSlice(slices[-1].start_us + spec.slice_us, spec.nbins, {(2, "CAL"): np.arange(spec.nbins)})  # still filling
    write_stream_atomically(slices_to_df([opening], spec, "energy"), tmp_path / "hist" / "histograms_current.arrows")
    every = [*slices, opening]
    info, rows = store.state(row_s=row_slices * spec.slice_s)
    assert info["n_slices"] == 203 and info["row_s"] == row_slices * spec.slice_s and info["open_good"] == opening.total()
    got = _decode(pl.from_arrow(server.from_arrow_ipc(server.to_arrow_ipc(info, rows))[1]), spec.nbins)

    for ch, state in {k for s in every for k in s.counts}:  # totals: the whole run, by channel and state
        assert np.array_equal(got[("total", None, state, ch)], sum(s.counts.get((ch, state), 0) for s in every))
    row_us = row_slices * spec.slice_us
    expected: list[list] = []  # [start µs, state, counts]: a new row at each state change and every row_us within a state
    for s in every:
        (state,) = {st for _, st in s.counts}
        c = sum(s.counts.values())
        if len(expected) > 0 and state == expected[-1][1] and s.start_us < expected[-1][0] + row_us:
            expected[-1][2] += c
            continue
        if len(expected) > 1 and state != expected[-1][1] == expected[-2][1] and s.start_us - expected[-1][0] < row_us // 2:
            tail = expected.pop()  # a state's last row, under half a row, joins the row before it
            expected[-1][2] += tail[2]
        expected.append([s.start_us, state, c])
    sent = {k: v for k, v in got.items() if k[0] == "row"}  # only the newest rows are sent
    assert {("row", t / 1e6, st, 0) for t, st, _ in expected[-server.ROWS :]} == set(sent)
    for t, st, c in expected[-server.ROWS :]:
        assert np.array_equal(sent[("row", t / 1e6, st, 0)], c)


def test_7_histograms_survive_arrow_ipc():
    rows = [
        server.CountRow.dense("total", None, "CAL", 4219, np.array([0, 3, 0, 70000, 1])),
        server.CountRow.dense("row", 1722087040.0, "SCAN3", 0, np.zeros(5, np.int64)),
    ]
    info = {"run": 3, "status": {"records": 12}, "hidden_states": ["CAL"]}
    back_info, table = server.from_arrow_ipc(server.to_arrow_ipc(info, rows))
    assert back_info == info
    t = pl.from_arrow(table)
    assert t["kind"].cast(pl.String).to_list() == ["total", "row"] and t["t"].to_list() == [None, 1722087040.0]
    assert t["bins"].to_list() == [[1, 3, 4], []] and t["counts"].to_list() == [[3, 70000, 1], []]
    assert table.schema.field("bins").type.value_type.bit_width == 16


def test_8_the_server_keeps_the_hidden_states_and_refuses_bad_ones(tmp_path):
    store = server.HistogramStore(tmp_path / "hist")
    store.hide_states(["SCAN3", "CAL2"])
    assert store.state()[0]["hidden_states"] == ["CAL2", "SCAN3"] and list(tmp_path.rglob("*")) == []  # kept here only
    store.reset(tmp_path / "hist")
    assert store.state()[0]["hidden_states"] == ["CAL2", "SCAN3"]  # a run started over keeps them
    store.hide_states([])
    assert store.state()[0]["hidden_states"] == []
    for bad in [None, "CAL2", [1, 2], {"a": 1}, ["x" * 65], ["s"] * 65]:
        with pytest.raises(ValueError):
            store.hide_states(bad)


def test_9_the_page_is_told_each_process_memory():
    gone = subprocess.Popen([sys.executable, "-c", "pass"])
    gone.wait()
    child = subprocess.Popen([sys.executable, "-c", "import time; x = bytearray(80_000_000); time.sleep(30)"])
    try:
        for _ in range(50):  # until the child has allocated its 80 MB
            mem = {p["name"]: p for p in server.process_memory({"viewer": os.getpid(), "child": child.pid, "gone": gone.pid})}
            if mem.get("child", {}).get("rss_bytes", 0) > 80e6:
                break
            time.sleep(0.1)
        assert set(mem) == {"viewer", "child"} and mem["child"]["pid"] == child.pid  # a process no longer running is left out
        assert mem["child"]["rss_bytes"] > 80e6 and mem["viewer"]["rss_bytes"] > 10e6
    finally:
        child.kill()
