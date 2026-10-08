"""Refit one spectral line on the histograms, summed over channels and experiment states, with the same mass2 line
model and binning as `Channel.linefit`. run_live_hist (loop.py) feeds it the finished slices and appends each fit to
HIST/fits.arrows, one row with its `LineModelResult.plotm()` image.
"""

import io
import time
from dataclasses import dataclass, replace

import numpy as np
import polars as pl
from numpy.typing import NDArray

from .histogram import HistogramSlice, HistogramSpec


@dataclass(frozen=True)
class RoiFit:
    """A line to fit: the `line` argument of `Channel.linefit`, the window around its peak, and where it came from."""

    line: str | float
    dlo: float
    dhi: float
    source: str = ""

    @property
    def label(self) -> str:
        return self.line if isinstance(self.line, str) else f"{self.line:g} eV line"


def fit_line(counts: NDArray, spec: HistogramSpec, roi: RoiFit, title: str) -> dict | None:
    """Fit `roi` to `counts` (a histogram at `spec`'s bins) with mass2's `fit_line_counts`, as `Channel.linefit`
    does, and draw it with `LineModelResult.plotm()`: the fit's numbers and its PNG image, the columns of FIT_SCHEMA
    but t, slices and every_s; or None if the fit fails or its region has fewer than 50 counts. Unlike linefit's
    default the model has a linear background: a line on a continuum fitted without one comes out far too wide."""
    import matplotlib  # noqa: PLC0415  imported here so the other live tools never pay for the fitting stack

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    import mass2  # noqa: PLC0415

    t0 = time.perf_counter()
    model = mass2.calibration.algorithms.get_model(roi.line, has_linear_background=True)
    pe = model.spect.peak_energy
    lo = int(np.ceil((pe - roi.dlo - spec.e_lo) / spec.bin_width))
    hi = int(np.floor((pe + roi.dhi - spec.e_lo) / spec.bin_width))
    y = np.asarray(counts[lo:hi], dtype=float)
    if y.sum() < 50:
        return None
    bin_centers = spec.e_lo + (np.arange(lo, hi) + 0.5) * spec.bin_width
    try:
        result = mass2.calibration.algorithms.fit_line_counts(model, bin_centers, y)
    except Exception:  # a bad fit must never stop the live tools
        return None
    t1 = time.perf_counter()
    result.set_label_hints(
        binsize=spec.bin_width, ds_shortname="all channels", attr_str="energy", unit_str="eV", cut_hint="good pulses"
    )
    fig, ax = plt.subplots(figsize=(6.0, 3.8), dpi=80)
    result.plotm(ax=ax, title=title)
    fig.tight_layout()
    png = io.BytesIO()
    fig.savefig(png, format="png")
    plt.close(fig)
    row = {
        "fit_ms": round(1000 * (t1 - t0), 1), "plot_ms": round(1000 * (time.perf_counter() - t1), 1),
        "counts_in_roi": int(y.sum()), "redchi": float(result.redchi),
        "roi_lo": float(spec.e_lo + lo * spec.bin_width), "roi_hi": float(spec.e_lo + hi * spec.bin_width),
        "line": str(roi.line), "source": roi.source, "dlo": roi.dlo, "dhi": roi.dhi, "png": png.getvalue(),
    }  # fmt: skip
    for name, key in [("fwhm", "fwhm"), ("peak_ph", "peak"), ("integral", "integral"), ("background", "background")]:
        p = result.params[name]
        row[key], row[f"{key}_err"] = float(p.value), None if p.stderr is None else float(p.stderr)
    return row


@dataclass(frozen=True)
class LiveFitter:
    """The running sum of the finished slices, and when it was last fitted. A refit is due every `every_s` seconds
    with new slices, and at the end of the stream. Like mass2's `Channel`, each method returns a new fitter."""

    spec: HistogramSpec
    roi: RoiFit
    every_s: float = 10.0
    total: HistogramSlice | None = None  # every finished slice, added up
    slices: int = 0
    latest_us: int | None = None  # end of the newest slice's slice_s window
    fitted_slices: int = 0
    last_fit_s: float = 0.0

    def add(self, finished: list[HistogramSlice]) -> "LiveFitter":
        total = self.total
        for s in finished:
            total = s if total is None else total.plus(s)
        latest_us = self.latest_us if len(finished) == 0 else self.spec.boundary_after(finished[-1].start_us)
        return replace(self, total=total, slices=self.slices + len(finished), latest_us=latest_us)

    def due(self, ended: bool) -> bool:
        """Whether to refit now; `ended` is True at the end of the stream."""
        return ended or (time.time() - self.last_fit_s >= self.every_s and self.slices > self.fitted_slices)

    def fit(self) -> tuple["LiveFitter", pl.DataFrame | None]:
        """Fit the running sum: the fitter, and the fit as one row of FIT_SCHEMA; or None (see `fit_line`)."""
        fitter = replace(self, fitted_slices=self.slices, last_fit_s=time.time())
        if self.total is None or self.latest_us is None:
            return fitter, None
        title = f"{self.roi.label}, all channels, all states, {self.slices} slices"
        row = fit_line(self.total.summed(), self.spec, self.roi, title)
        if row is None:
            return fitter, None
        context = {"t": self.latest_us / 1e6, "slices": self.slices, "every_s": self.every_s}
        return fitter, pl.DataFrame([context | row], schema=FIT_SCHEMA)


FIT_SCHEMA = pl.Schema({
    "t": pl.Float64(), "slices": pl.Int64(), "fit_ms": pl.Float64(), "plot_ms": pl.Float64(),
    "counts_in_roi": pl.Int64(), "redchi": pl.Float64(), "roi_lo": pl.Float64(), "roi_hi": pl.Float64(),
    "fwhm": pl.Float64(), "fwhm_err": pl.Float64(), "peak": pl.Float64(), "peak_err": pl.Float64(),
    "integral": pl.Float64(), "integral_err": pl.Float64(), "background": pl.Float64(), "background_err": pl.Float64(),
    "line": pl.String(), "source": pl.String(), "dlo": pl.Float64(), "dhi": pl.Float64(), "every_s": pl.Float64(),
    "png": pl.Binary(),
})  # fmt: skip
