import numpy as np
import pylab as plt
from dataclasses import dataclass
import moss
import polars as pl
import mass


def fourier_filter(avg_signal, n_pretrigger, noise_psd, noise_autocorr_vec, dt, fmax=None, f_3db=None, peak_signal=1.0):
    peak_signal = np.amax(avg_signal)-avg_signal[0]
    maker = mass.FilterMaker(avg_signal, n_pretrigger, noise_psd=noise_psd,
                             noise_autocorr=noise_autocorr_vec,
                             sample_time_sec=dt, peak=peak_signal)
    mass_filter = maker.compute_5lag(fmax=fmax, f_3db=f_3db)
    return Filter(filter=mass_filter.values,
                  v_dv_known_wrong=mass_filter.predicted_v_over_dv,
                  dt=dt,
                  filter_type="mass fourier")


@dataclass(frozen=True)
class Filter:
    filter: np.ndarray
    v_dv_known_wrong: float
    dt: float
    filter_type: str

    def plot(self, axis=None, **plotkwarg):
        if axis is None:
            plt.figure()
            axis = plt.gca()
        axis.plot(self.frequencies(), self.filter, label="fourier filter", **plotkwarg)
        axis.grid()
        axis.set_title(f"{self.filter_type=} v_dv_known_wrong={self.v_dv_known_wrong:.2f}")
        axis.set_ylabel("filter value")
        axis.set_xlabel("Lag Time (s)")
        axis.figure.tight_layout()

    def frequencies(self):
        n = len(self.filter)
        return np.arange(0, n, dtype=float) * 0.5 / ((n - 1) * self.dt)

    def __call__(self, pulse):
        return np.dot(self.filter, pulse)


def filter_data_5lag(filter_values, pulses):
    # These parameters fit a parabola to any 5 evenly-spaced points
    fit_array = (
        np.array(
            ((-6, 24, 34, 24, -6), (-14, -7, 0, 7, 14), (10, -5, -10, -5, 10)),
            dtype=float,
        )
        / 70.0
    )
    conv = np.zeros((5, pulses.shape[0]), dtype=float)
    conv[0, :] = np.dot(pulses[:, 0:-4], filter_values)
    conv[1, :] = np.dot(pulses[:, 1:-3], filter_values)
    conv[2, :] = np.dot(pulses[:, 2:-2], filter_values)
    conv[3, :] = np.dot(pulses[:, 3:-1], filter_values)
    conv[4, :] = np.dot(pulses[:, 4:], filter_values)

    param = np.dot(fit_array, conv)
    peak_x = -0.5 * param[1, :] / param[2, :]
    peak_y = param[0, :] - 0.25 * param[1, :] ** 2 / param[2, :]
    return peak_x, peak_y


@dataclass(frozen=True)
class Filter5LagStep(moss.CalStep):
    filter: Filter
    spectrum: moss.NoisePSD

    def calc_from_df(self, df):
        dfs = []
        for df_iter in df.iter_slices(10000):
            peak_x, peak_y = moss.filters.filter_data_5lag(
                self.filter.filter, df_iter[self.inputs[0]].to_numpy()
            )
            dfs.append(pl.DataFrame({"peak_x": peak_x, "peak_y": peak_y}))
        df2 = pl.concat(dfs).with_columns(df)
        df2 = df2.rename({"peak_x": self.output[0], "peak_y": self.output[1]})
        return df2

    def dbg_plot(self, df):
        return self.filter.plot()
