from dataclasses import dataclass, field
import lmfit
import copy
import mass
import math
import numpy as np
import polars as pl
from typing import Optional
import moss
import pylab as plt


def handle_none(val, default):
    if val is None:
        return copy.copy(default)
    return val


@dataclass(frozen=True)
class FitSpec:
    model: mass.GenericLineModel
    bin_edges: np.ndarray
    use_expr: pl.Expr
    params_update: lmfit.parameter.Parameters

    def params(self, bin_centers, counts):
        params = self.model.make_params()
        params = self.model.guess(counts, bin_centers=bin_centers, dph_de=1)
        params["dph_de"].set(1.0, vary=False)
        params = params.update(self.params_update)
        return params

    def fit_series_without_use_expr(self, series):
        bin_centers, counts = moss.misc.hist_of_series(series, self.bin_edges)
        params = self.params(bin_centers, counts)
        bin_centers, bin_size = moss.misc.midpoints_and_step_size(self.bin_edges)
        result = self.model.fit(counts, params, bin_centers=bin_centers)
        result.set_label_hints(
            binsize=bin_size,
            ds_shortname="??",
            unit_str="eV",
            attr_str=series.name,
            states_hint=f"{self.use_expr}",
            cut_hint="",
        )
        return result

    def fit_df(self, df: pl.DataFrame, col: str, good_expr: pl.Expr):
        series = moss.good_series(df, col, good_expr, use_expr=self.use_expr)
        return self.fit_series_without_use_expr(series)

    def fit_ch(self, ch, col: str):
        return self.fit_df(ch.df, col, ch.good_expr)


@dataclass(frozen=True)
class MultiFit:
    default_fit_width: float = 50
    default_bin_size: float = 0.5
    default_use_expr: bool = True
    default_params_update: dict = field(default_factory=lmfit.Parameters)
    fitspecs: list[FitSpec] = field(default_factory=list)
    results: Optional[list] = None

    def with_line(
        self, line, dlo=None, dhi=None, bin_size=None, use_expr=None, params_update=None
    ):
        model = mass.getmodel(line)
        peak_energy = model.spect.peak_energy
        dlo = handle_none(dlo, self.default_fit_width/2)
        dhi = handle_none(dhi, self.default_fit_width/2)
        bin_size = handle_none(bin_size, self.default_bin_size)
        params_update = handle_none(params_update, self.default_params_update)
        use_expr = handle_none(use_expr, self.default_use_expr)
        bin_edges = np.arange(-dlo, dhi + bin_size, bin_size) + peak_energy
        fitspec = FitSpec(model, bin_edges, use_expr, params_update)
        return self.with_fitspec(fitspec)

    def with_fitspec(self, fitspec):
        return MultiFit(
            self.default_fit_width,
            self.default_bin_size,
            self.default_use_expr,
            self.default_params_update,
            sorted(self.fitspecs + [fitspec], key=lambda x: x.model.spect.peak_energy),
            # make sure they're always sorted by energy
            self.results,
        )

    def with_results(self, results):
        return MultiFit(
            self.default_fit_width,
            self.default_bin_size,
            self.default_use_expr,
            self.default_params_update,
            self.fitspecs,
            results,
        )

    def results_params_as_df(self):
        result = self.results[0]
        param_names = result.params.keys()
        d = {}
        d["line"] = [fitspec.model.spect.shortname for fitspec in self.fitspecs]
        d["peak_energy_ref"] = [fitspec.model.spect.peak_energy for fitspec in self.fitspecs]
        d["peak_energy_ref_err"] = []
        # for quickline, position_uncertainty is a string
        # translate that into a large value for uncertainty so we can proceed without crashing
        for fitspec in self.fitspecs:
            if isinstance(fitspec.model.spect.position_uncertainty, str):
                v = 0.1*fitspec.model.spect.peak_energy  # 10% error is large!
            else:
                v = fitspec.model.spect.position_uncertainty
            d["peak_energy_ref_err"].append(v)
        for param_name in param_names:
            d[param_name] = [result.params[param_name].value for result in self.results]
            d[param_name+"_stderr"] = [result.params[param_name].stderr for result in self.results]
        return pl.DataFrame(d)

    def fit_series_without_use_expr(self, series: pl.Series):
        results = [fitspec.fit_series_without_use_expr(series) for fitspec in self.fitspecs]
        return self.with_results(results)

    def fit_df(self, df: pl.DataFrame, col: str, good_expr: pl.Expr):
        results = []
        for fitspec in self.fitspecs:
            result = fitspec.fit_df(df, col, good_expr)
            results.append(result)
        return self.with_results(results)

    def fit_ch(self, ch, col: str):
        return self.fit_df(ch.df, col, ch.good_expr)

    def plot_results(self, n_extra_axes=0):
        assert self.results is not None
        n = len(self.results)+n_extra_axes
        cols = min(3, n)
        rows = math.ceil(n / cols)
        fig, axes = plt.subplots(
            rows, cols, figsize=(cols * 4, rows * 4)
        )  # Adjust figure size as needed

        # If there's only one subplot, axes is not a list but a single Axes object.
        if rows == 1 and cols == 1:
            axes = [axes]
        elif rows == 1 or cols == 1:
            axes = axes.flatten()
        else:
            axes = axes.ravel()

        for result, ax in zip(self.results, axes):
            result.plotm(ax=ax)

        # Hide any remaining empty subplots
        for ax in axes[n:]:
            ax.axis("off")

        plt.tight_layout()
        return fig, axes

    def plot_results_and_pfit(self, uncalibrated_name, previous_energy2ph, n_extra_axes=0):
        _fig, axes = self.plot_results(n_extra_axes=1+n_extra_axes)
        ax = axes[len(self.results)]
        multifit_df = self.results_params_as_df()
        peaks_in_energy_rough_cal = multifit_df["peak_ph"].to_numpy()
        peaks_uncalibrated = np.array([previous_energy2ph(e) for e in peaks_in_energy_rough_cal])
        peaks_in_energy_reference = multifit_df["peak_energy_ref"].to_numpy()
        pfit_gain, rms_residual_energy = self.to_pfit_gain(previous_energy2ph)
        plt.sca(ax)
        x = np.linspace(0, np.amax(peaks_uncalibrated))
        plt.plot(x, pfit_gain(x), label="fit")
        gain = peaks_uncalibrated/peaks_in_energy_reference
        plt.plot(peaks_uncalibrated, gain, "o")
        plt.xlabel(uncalibrated_name)
        plt.ylabel("gain")
        plt.title(f"{rms_residual_energy=:.3f}")
        for name, x, y in zip(multifit_df["line"], peaks_uncalibrated, gain):
            ax.annotate(str(name), (x, y))
        return axes

    def to_pfit_gain(self, previous_energy2ph):
        multifit_df = self.results_params_as_df()
        peaks_in_energy_rough_cal = multifit_df["peak_ph"].to_numpy()
        peaks_uncalibrated = np.array([previous_energy2ph(e) for e in peaks_in_energy_rough_cal])
        peaks_in_energy_reference = multifit_df["peak_energy_ref"].to_numpy()
        gain = peaks_uncalibrated/peaks_in_energy_reference
        pfit_gain = np.polynomial.Polynomial.fit(peaks_uncalibrated, gain, deg=2)

        def ph2energy(ph):
            gain = pfit_gain(ph)
            return ph/gain
        e_predicted = ph2energy(peaks_uncalibrated)
        rms_residual_energy = moss.misc.root_mean_squared(e_predicted-peaks_in_energy_reference)
        return pfit_gain, rms_residual_energy

    def to_mass_cal(self, previous_energy2ph, curvetype="gain", approximate=False):
        df = self.results_params_as_df()
        maker = mass.calibration.EnergyCalibrationMaker(
            ph=np.array([previous_energy2ph(x) for x in df["peak_ph"].to_numpy()]),
            energy=df["peak_energy_ref"].to_numpy(),
            dph=df["peak_ph_stderr"].to_numpy(),
            de=df["peak_energy_ref_err"].to_numpy(),
            names=[name for name in df["line"]])
        cal = maker.make_calibration(curvename=curvetype, approximate=approximate)
        return cal


@dataclass(frozen=True)
class MultiFitQuadraticGainCalStep(moss.CalStep):
    pfit_gain: np.polynomial.Polynomial
    multifit: MultiFit
    rms_residual_energy: float

    def calc_from_df(self, df):
        # only works with in memory data, but just takes it as numpy data and calls function
        # is much faster than map_elements approach, but wouldn't work with out of core data without some extra book keeping
        inputs_np = [df[input].to_numpy() for input in self.inputs]
        out = self.ph2energy(inputs_np[0])
        df2 = pl.DataFrame({self.output[0]: out}).with_columns(df)
        return df2

    def dbg_plot(self, df):
        self.multifit.plot_results_and_pfit(uncalibrated_name=self.inputs[0],
                                            previous_energy2ph=self.energy2ph)

    def ph2energy(self, ph):
        gain = self.pfit_gain(ph)
        return ph/gain

    def energy2ph(self, energy):
        # ph2energy is equivalent to this with y=energy, x=ph
        # y = x/(c + b*x + a*x^2)
        # so
        # y*c + (y*b-1)*x + a*x^2 = 0
        # and given that we've selected for well formed calibrations,
        # we know which root we want
        cba = self.pfit_gain.convert().coef
        c, bb, a = cba*energy
        b = bb-1
        ph = (-b-np.sqrt(b**2-4*a*c))/(2*a)
        import math
        assert math.isclose(self.ph2energy(ph), energy, rel_tol=1e-6, abs_tol=1e-3)
        return ph

    @classmethod
    def learn(cls, ch, multifit_spec: MultiFit, previous_cal_step_index,
              calibrated_col, use_expr=True
              ):
        previous_cal_step = ch.steps[previous_cal_step_index]
        rough_energy_col = previous_cal_step.output[0]
        uncalibrated_col = previous_cal_step.inputs[0]

        multifit_with_results = multifit_spec.fit_ch(ch, col=rough_energy_col)
        # multifit_df = multifit_with_results.results_params_as_df()
        pfit_gain, rms_residual_energy = multifit_with_results.to_pfit_gain(previous_cal_step.energy2ph)
        step = cls(
            [uncalibrated_col],
            [calibrated_col],
            ch.good_expr,
            use_expr,
            pfit_gain,
            multifit_with_results,
            rms_residual_energy
        )
        return step


@dataclass(frozen=True)
class MultiFitMassCalibrationStep(moss.CalStep):
    cal: mass.EnergyCalibration
    multifit: MultiFit

    def calc_from_df(self, df):
        # only works with in memory data, but just takes it as numpy data and calls function
        # is much faster than map_elements approach, but wouldn't work with out of core data without some extra book keeping
        inputs_np = [df[input].to_numpy() for input in self.inputs]
        out = self.ph2energy(inputs_np[0])
        df2 = pl.DataFrame({self.output[0]: out}).with_columns(df)
        return df2

    def dbg_plot(self, df):
        axes = self.multifit.plot_results_and_pfit(uncalibrated_name=self.inputs[0],
                                                   previous_energy2ph=self.energy2ph,
                                                   n_extra_axes=1)
        ax = axes[-1]
        multifit_df = self.multifit.results_params_as_df()
        peaks_in_energy_rough_cal = multifit_df["peak_ph"].to_numpy()
        peaks_uncalibrated = np.array([self.energy2ph(e) for e in peaks_in_energy_rough_cal])
        peaks_in_energy_reference = multifit_df["peak_energy_ref"].to_numpy()
        plt.sca(ax)
        x = np.linspace(1, np.amax(peaks_uncalibrated))
        gain_from_cal = x/self.cal(x)
        plt.plot(x, gain_from_cal, label="mass cal")
        gain = peaks_uncalibrated/peaks_in_energy_reference
        plt.plot(peaks_uncalibrated, gain, "o")
        plt.xlabel(self.inputs[0])
        plt.ylabel("gain")
        plt.title("actual mass cal")
        for name, x, y in zip(multifit_df["line"], peaks_uncalibrated, gain):
            ax.annotate(str(name), (x, y))
        plt.legend()

    def ph2energy(self, ph):
        return self.cal.ph2energy(ph)

    def energy2ph(self, energy):
        return self.cal.energy2ph(energy)

    @classmethod
    def learn(cls, ch, multifit_spec: MultiFit, previous_cal_step_index,
              calibrated_col, use_expr=True
              ):
        """multifit then make a mass calibration object with curve_type="gain" and approx=False
        TODO: support more options"""
        previous_cal_step = ch.steps[previous_cal_step_index]
        rough_energy_col = previous_cal_step.output[0]
        uncalibrated_col = previous_cal_step.inputs[0]

        multifit_with_results = multifit_spec.fit_ch(ch, col=rough_energy_col)
        cal = multifit_with_results.to_mass_cal(previous_cal_step.energy2ph)
        step = cls(
            [uncalibrated_col],
            [calibrated_col],
            ch.good_expr,
            use_expr,
            cal,
            multifit_with_results,
        )
        return step
