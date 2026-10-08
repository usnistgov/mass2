"""The `pulsedata` datasets the live demo can replay, each with the analysis that learned its recipe.

The recipes themselves are saved once, with `Channels.save_recipes`, as `mass2/live/demo/recipes/<key>.pkl` in the
package, and the live tools only ever load them from there. To add a dataset, add a `DemoDataset` to
`DATASETS` and run `python -m mass2.live.demo.datasets` to write its recipe (and rewrite the others). The learn
function is an ordinary `Channel -> Channel` analysis, as in the example notebooks.
"""

import argparse
import os
from collections.abc import Callable, Sequence
from importlib import resources
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import polars as pl
import pulsedata

import mass2
from ..core.fit import RoiFit
from ..core.histogram import HistogramSpec
from mass2.core.misc import PulseDataFromNumpy
from .simulate import ScaledChannel, scale_pulses


@dataclass(frozen=True)
class Pixel:
    """One detector in the array map: where it sits, and for a copy, which real channel it copies and its gain."""

    ch_num: int
    x: int
    y: int
    source_ch: int | None = None  # None for a real channel
    gain: float = 1.0


def detector_array(nx: int, ny: int, numbers: Sequence[int], real: Sequence[int], off: dict[int, float]) -> tuple[Pixel, ...]:
    """An nx by ny array; `numbers` are channel numbers in row order, `real` the ones with real data.

    Every other pixel is an exact copy of a real channel (gain 1), except the pixels in `off`, recorded at a
    different gain, as real detectors are; each of those gets its own recipe, which corrects it.
    """
    assert len(numbers) == nx * ny and set(real) <= set(numbers)
    pixels = []
    for k, ch in enumerate(numbers):
        x, y = k % nx, k // nx
        if ch in real:
            pixels.append(Pixel(ch, x, y))
        else:
            pixels.append(Pixel(ch, x, y, source_ch=real[k % len(real)], gain=off.get(ch, 1.0)))
    return tuple(pixels)


@dataclass(frozen=True)
class DemoDataset:
    """One replayable dataset: where it lives, how to learn its recipe, and how to show it."""

    key: str  # key into pulsedata.pulse_noise_ljh_pairs
    title: str
    learn: Callable[[mass2.Channel], mass2.Channel]
    energy_col: str
    spec: HistogramSpec  # bin_width = the finest bin used for resolution fits in the original analysis
    bin_source: str  # where that bin width comes from
    roi: RoiFit  # the line fitted in the original analysis, refitted live on all channels
    pixels: tuple[Pixel, ...]  # the detector array: real channels and gain-scaled copies
    default_speed: float = 5.0  # playback speed to start at, in multiples of real time

    @property
    def scales(self) -> tuple[ScaledChannel, ...]:
        """The simulated copies, as the simulator takes them."""
        return tuple(ScaledChannel(p.source_ch, p.ch_num, p.gain) for p in self.pixels if p.source_ch is not None)

    @property
    def layout(self) -> dict[int, dict]:
        """{channel: {x, y, copy_of, gain}} for the viewer's array map."""
        return {p.ch_num: {"x": p.x, "y": p.y, "copy_of": p.source_ch, "gain": p.gain} for p in self.pixels}

    @property
    def pulse_folder(self) -> Path:
        return pulsedata.pulse_noise_ljh_pairs[self.key].pulse_folder

    @property
    def noise_folder(self) -> Path:
        return pulsedata.pulse_noise_ljh_pairs[self.key].noise_folder

    @property
    def recipe_path(self) -> Path:
        """The saved recipe shipped with mass2 (written by `python -m mass2.live.datasets`)."""
        return Path(str(resources.files("mass2.live.demo").joinpath("recipes", f"{self.key}.pkl")))


def _learn_bessy(ch: mass2.Channel) -> mass2.Channel:
    """Soft x-ray lines, calibrated by fits during the CAL2 state (as in docs/getting_started.md)."""
    use_cal = pl.col("state_label") == "CAL2"
    rough_lines: list[Any] = [
        "CKAlpha",
        "NKAlpha",
        "OKAlpha",
        "FeLl",
        "FeLAlpha",
        "FeLBeta",
        "NiLAlpha",
        "NiLBeta",
        "CuLAlpha",
        "CuLBeta",
        980,
    ]
    ch = ch.summarize_pulses().with_good_expr_pretrig_rms_and_postpeak_deriv(8, 8).filter5lag(f_3db=10000)
    ch = ch.driftcorrect(indicator_col="pretrig_mean", uncorrected_col="5lagy", use_expr=pl.lit(True))
    ch = ch.rough_cal_combinatoric(rough_lines, "5lagy_dc", "energy_5lagy_dc", ph_smoothing_fwhm=6, use_expr=use_cal)
    fit = mass2.MultiFit(default_fit_width=80, default_use_expr=use_cal, default_bin_size=0.3)
    dlo: dict[Any, float] = {980.0: 20}
    dhi: dict[Any, float] = {"CuLAlpha": 12, "NiLAlpha": 12}
    fit_lines: list[Any] = ["CKAlpha", "NKAlpha", "OKAlpha", "FeLl", "FeLAlpha", "NiLAlpha", "CuLAlpha", 980.0]
    for line in fit_lines:
        fit = fit.with_line(line, dlo=dlo.get(line), dhi=dhi.get(line))
    return ch.multifit_mass_cal(fit, -1, "energy_5lagy_best")


def _learn_mnkalpha(ch: mass2.Channel) -> mass2.Channel:
    """Mn, Cu and Pd fluorescence lines (as in examples/ljh_mnkalpha.py)."""
    lines: list[Any] = ["MnKAlpha", "MnKBeta", "CuKAlpha", "CuKBeta", "PdLAlpha", "PdLBeta"]
    ch = ch.summarize_pulses().with_good_expr_pretrig_rms_and_postpeak_deriv().filter5lag(f_3db=10e3).driftcorrect()
    return ch.rough_cal_combinatoric(lines, "5lagy_dc", "energy_5lagy_dc", ph_smoothing_fwhm=40)


def _learn_gamma(ch: mass2.Channel) -> mass2.Channel:
    """97 and 103 keV gamma lines (as in examples/gamma_20241005.py)."""
    lines: list[Any] = [97431, 103180]
    ch = (
        ch
        .summarize_pulses()
        .with_good_expr_pretrig_rms_and_postpeak_deriv()
        .with_good_expr_nsigma_range_outlier_resistant(col_nsigma_pairs=[("pretrig_mean", 100)])
        .filter5lag()
        .driftcorrect(indicator_col="pretrig_mean", uncorrected_col="5lagy")
    )
    return ch.rough_cal_combinatoric(lines, "5lagy_dc", "energy_5lagy_dc", ph_smoothing_fwhm=50)


DATASETS: dict[str, DemoDataset] = {
    d.key: d
    for d in [
        DemoDataset(
            key="bessy_20240727",
            title="BESSY 2024-07-27, soft x-ray (CAL and SCAN states)",
            learn=_learn_bessy,
            energy_col="energy_5lagy_best",
            spec=HistogramSpec(e_lo=0, e_hi=1200, bin_width=0.25, slice_s=10),  # slices: 10 s of data
            bin_source="examples/bessy_20240727.py, linefit(..., binsize=0.25)",
            roi=RoiFit(600, 20, 20, "examples/bessy_20240727.py: linefit(600, dlo=20, dhi=20, binsize=0.25)"),
            pixels=detector_array(4, 4, range(4219, 4235), real=[4219, 4220], off={4224: 1.03, 4229: 0.975, 4233: 1.015}),
        ),
        DemoDataset(
            key="20230626",
            title="2023-06-26, Mn Kα, Cu Kα, Pd Lα",
            learn=_learn_mnkalpha,
            energy_col="energy_5lagy_dc",
            spec=HistogramSpec(e_lo=0, e_hi=10000, bin_width=0.5, slice_s=2),  # a 21-minute run: 2 s slices
            bin_source='examples/ljh_mnkalpha.py, linefit("MnKAlpha", ...) with its default binsize=0.5',
            roi=RoiFit("MnKAlpha", 50, 50, 'examples/ljh_mnkalpha.py: linefit("MnKAlpha"), default dlo=dhi=50, binsize=0.5'),
            pixels=detector_array(4, 3, range(4101, 4113), real=[4102, 4109], off={4106: 1.02, 4111: 0.985}),
        ),
        DemoDataset(
            key="gamma_20241005",
            title="Gamma 2024-10-05, 97 and 103 keV",
            learn=_learn_gamma,
            energy_col="energy_5lagy_dc",
            spec=HistogramSpec(e_lo=0, e_hi=150000, bin_width=8, slice_s=60),  # a 7-hour, low-rate run: 60 s slices
            bin_source="examples/gamma_20241005.py, linefit(..., binsize=8)",
            roi=RoiFit(97431, 150, 150, "examples/gamma_20241005.py: linefit(97431, dlo=150, dhi=150, binsize=8)"),
            pixels=detector_array(3, 3, range(1, 10), real=[2, 5], off={7: 1.02, 3: 0.985}),
        ),
    ]
}


def gain_copy(ch: mass2.Channel, ch_num: int, gain: float) -> mass2.Channel:
    """Channel `ch` as recorded by a detector of a different gain: the same pulses `mass2-live-sim` writes for
    the copy, as a Channel its own recipe can be learned from."""
    assert ch.pulseframer is not None, f"channel {ch.header.ch_num} has no raw pulses"
    pulses = ch.pulseframer.load_raw_chunk(0, ch.npulses)["pulse"].to_numpy()
    framer = PulseDataFromNumpy(scale_pulses(pulses, gain, ch.header.n_presamples))
    return replace(ch, header=replace(ch.header, ch_num=ch_num), pulseframer=framer)


def build_recipe(dataset: DemoDataset, path: str | Path) -> None:
    """Learn `dataset`'s analysis on its full LJH data and save the recipe that produces its energy column.

    Each copy at a wrong gain gets a recipe of its own, learned the same way from its own scaled pulses, so its
    calibration absorbs the gain and its energies agree with the real channels'. Copies at gain 1 are exact
    copies, saved with their source channel's recipe. So every channel of the array has a recipe in the file.
    """
    path = Path(path)
    data = mass2.Channels.from_ljh_folder(dataset.pulse_folder, dataset.noise_folder).with_experiment_state_by_path()
    copies = {
        p.ch_num: gain_copy(data.channels[p.source_ch], p.ch_num, p.gain)
        for p in dataset.pixels
        if p.source_ch is not None and p.gain != 1.0
    }
    data = replace(data, channels=data.channels | copies)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    recipes = data.map(dataset.learn).save_recipes(str(tmp), required_fields=dataset.energy_col)
    exact = {p.ch_num: recipes[p.source_ch] for p in dataset.pixels if p.source_ch is not None and p.gain == 1.0}
    mass2.misc.pickle_object(recipes | exact, str(tmp))
    os.replace(tmp, path)  # a concurrent reader never sees a half-written recipe


def main(argv: Sequence[str] | None = None) -> None:
    """Rewrite the saved recipes in mass2/live/demo/recipes/ (all datasets, or the ones named)."""
    p = argparse.ArgumentParser(description=main.__doc__)
    p.add_argument("keys", nargs="*", help=f"datasets to rebuild (default: all of {', '.join(DATASETS)})")
    args = p.parse_args(argv)
    for key in args.keys or DATASETS:
        if key not in DATASETS:
            p.error(f"unknown dataset {key!r}")
        dataset = DATASETS[key]
        dataset.recipe_path.parent.mkdir(parents=True, exist_ok=True)
        build_recipe(dataset, dataset.recipe_path)
        print(f"wrote {dataset.recipe_path}", flush=True)


if __name__ == "__main__":
    main()
