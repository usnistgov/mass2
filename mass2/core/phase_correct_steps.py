import polars as pl
import pylab as plt
from dataclasses import dataclass
import mass2
from .cal_steps import RecipeStep
from .phase_correct import PhaseCorrector


@dataclass(frozen=True)
class PhaseCorrectMassStep(RecipeStep):
    line_names: list[str]
    line_energies: list[float]
    previous_step_index: int
    phase_corrector: PhaseCorrector

    def calc_from_df(self, df):
        # since we only need to load two columns I'm assuming we can fit them in memory and just
        # loading them whole
        # if it becomes an issues, use iter_slices or
        # add a user defined funciton in rust
        indicator_col, uncorrected_col = self.inputs
        corrected_col = self.output[0]
        indicator = df[indicator_col].to_numpy()
        uncorrected = df[uncorrected_col].to_numpy()
        corrected = self.phase_corrector(indicator, uncorrected)
        series = pl.Series(corrected_col, corrected)
        df2 = df.with_columns(series)
        return df2

    def dbg_plot(self, df):
        indicator_col, uncorrected_col = self.inputs
        df_small = df.lazy().filter(self.good_expr).filter(self.use_expr).select(self.inputs + self.output).collect()
        mass2.misc.plot_a_vs_b_series(df_small[indicator_col], df_small[uncorrected_col])
        mass2.misc.plot_a_vs_b_series(
            df_small[indicator_col],
            df_small[self.output[0]],
            plt.gca(),
        )
        plt.legend()
        plt.tight_layout()
        return plt.gca()


def phase_correct_mass_specific_lines(
    ch,
    indicator_col,
    uncorrected_col,
    corrected_col,
    previous_step_index,
    line_names,
    use_expr,
):
    previous_step, previous_step_index = ch.get_step(previous_step_index)
    (line_names, line_energies) = mass2.calibration.algorithms.line_names_and_energies(line_names)
    line_positions = [previous_step.energy2ph(line_energy) for line_energy in line_energies]
    [indicator, uncorrected] = ch.good_serieses([indicator_col, uncorrected_col], use_expr=use_expr)
    phase_corrector = mass2.core.phase_correct.phase_correct(
        indicator.to_numpy(),
        uncorrected.to_numpy(),
        line_positions,
        indicatorName=indicator_col,
        uncorrectedName=uncorrected_col,
    )
    return PhaseCorrectMassStep(
        inputs=[indicator_col, uncorrected_col],
        output=[corrected_col],
        good_expr=ch.good_expr,
        use_expr=use_expr,
        line_names=line_names,
        line_energies=line_energies,
        previous_step_index=previous_step_index,
        phase_corrector=phase_corrector,
    )
