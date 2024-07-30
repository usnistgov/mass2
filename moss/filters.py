import numpy as np
import pylab as plt
from dataclasses import dataclass
import moss
import polars as pl

def fourier_filter(avg_signal, noise_psd, dt, fmax=None, f_3db=None, peak_signal=1.0):
    filter, variance = calc_fourier_filter(avg_signal, noise_psd, dt, fmax, f_3db, peak_signal)
    return Filter(filter, variance, dt, filter_type="fourier")    

@dataclass(frozen=True)
class Filter:
    filter: np.ndarray
    variance: float
    dt: float
    filter_type: str

    def plot(self, axis=None, **plotkwarg):
        if axis is None:
            plt.figure()
            axis = plt.gca()
        axis.plot(self.frequencies(), self.filter, label="fourier filter", **plotkwarg)
        axis.grid()
        axis.set_title(f"{self.filter_type=}")
        axis.set_ylabel("filter value")
        axis.set_xlabel("Lag Time (s)")
        axis.figure.tight_layout()

    def frequencies(self):
        n = len(self.filter)
        return np.arange(0, n, dtype=float) * 0.5 / ((n - 1) * self.dt)

    def __call__(self, pulse):
        return np.dot(self.filter, pulse)

def apply_fmax(signal_freq_domain, fmax, dt):
    if fmax is None:
        return signal_freq_domain
    n = len(signal_freq_domain)
    freq = np.arange(0, n, dtype=float) * 0.5 / ((n - 1) * dt)
    out = signal_freq_domain[:]
    out[freq>fmax]=0
    return out

def apply_f_3db(signal_freq_domain, f_3db, dt):
    if f_3db is None:
        return signal_freq_domain
    n = len(signal_freq_domain)
    freq = np.arange(0, n, dtype=float) * 0.5 / ((n - 1) * dt)
    return signal_freq_domain / (1 + (freq * 1.0 / f_3db)**2)

def normalize_filter(filter):
    filter -= np.mean(filter)
    return filter/np.sqrt(np.dot(filter, filter))

def calc_fourier_filter(avg_signal, noise_psd, dt, fmax=None, f_3db=None, peak_signal=1.0):
    """Compute the Fourier-domain filter and variances for signal processing.

    Args:
    - avg_signal (np.ndarray): Average signal data.
    - noise_psd (np.ndarray): Power Spectral Density (PSD) of the noise.
    - shorten (int, optional): Amount to shorten the signal by (default: 0).
    - fmax (float, optional): Maximum frequency limit (default: None).
    - f_3db (float, optional): 3dB frequency limit (default: None).
    - sample_time (float, optional): Sampling time (default: 1.0).
    - peak_signal (float, optional): Peak signal value (default: 1.0).

    Returns:
    - filt_fourier (np.ndarray): Filtered signal in Fourier domain.
    - variances (dict): Dictionary containing variance 'fourier'.
    """
    signal_freq_domain1 = np.fft.rfft(avg_signal)

    if len(signal_freq_domain1) != len(noise_psd):
        raise ValueError(f"Signal DFT and noise PSD are not the same length ({len(signal_freq_domain1)} and {len(noise_psd)})")
    n = len(signal_freq_domain1)
    signal_freq_domain2 = signal_freq_domain1 / noise_psd

    signal_freq_domain3 = apply_fmax(signal_freq_domain2, fmax, dt)
    signal_freq_domain4 = apply_f_3db(signal_freq_domain3, f_3db, dt)

    filter = np.fft.irfft(signal_freq_domain4)

    noise_ft_squared = (len(noise_psd) - 1) / dt * noise_psd
    kappa = (np.abs(signal_freq_domain4 * peak_signal)**2 / noise_ft_squared)[1:].sum()
    variance = 1 / kappa
    return normalize_filter(filter), variance

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