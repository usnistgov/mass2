import numpy as np
import pylab as plt #type: ignore
from dataclasses import dataclass, field
from numpy import ndarray
from typing import Tuple, Optional
import matplotlib.pyplot as plt

def calc_autocorrelation(data):
    ntraces, nsamples = data.shape
    ac = np.zeros(nsamples, dtype=float)

    for i in range(ntraces):
        pulse = data[i,:]
        pulse -= pulse.mean()
        ac += np.correlate(pulse, pulse, 'full')[nsamples - 1:]

    ac /= ntraces
    ac /= nsamples - np.arange(nsamples, dtype=float)
    return ac

def calc_autocorrelation_times(n, dt):
    return np.arange(n)*dt

def autocorrelation(data, dt):
    return AutoCorrelation(calc_autocorrelation(data), 
                           calc_autocorrelation_times(data.shape[1], dt))

@dataclass
class AutoCorrelation:
    ac: np.ndarray
    times: np.ndarray
    

    def plot(self, axis=None, **plotkwarg):
        if axis is None:
            plt.figure()
            axis = plt.gca()
        axis.plot(self.times[1:], self.ac[1:], **plotkwarg)
        axis.grid()
        axis.set_ylabel("Autocorrelation (abrs?)")
        axis.set_xlabel("Lag Time (s)")
        axis.figure.tight_layout()

def calc_psd(data: ndarray, dt: float, window: None=None) -> ndarray:
        """Process a data segment of length 2m using the window function
        given.  window can be None (square window), a callable taking the
        length and returning a sequence, or a sequence."""
        ntraces, m2 = data.shape
        data = data - np.mean(data, axis=0) # don't use -= here in case input array is read only
        if np.isnan(data).any():
            raise ValueError("data contains NaN")
        if window is None:
            wksp = data
            sum_window = m2
        else:
            try:
                w = window(m2)
            except TypeError:
                w = np.array(window)
            wksp = w * data
            sum_window = (w**2).sum()

        scale_factor = 2. / (sum_window * m2)
        scale_factor *= dt * m2
        wksp = np.fft.rfft(wksp, axis=1)

        # The first line adds 2x too much to the first/last bins.
        ps = np.abs(wksp)**2
        specsum = scale_factor * np.sum(ps, axis=0)
        return specsum / ntraces


def calc_psd_frequencies(nbins: int, dt: float) -> ndarray:
    return np.arange(nbins, dtype=float) / (2 * dt * nbins)

def noise_psd(data: ndarray, dt: float, window: None=None) -> "NoisePSD":
    psd = calc_psd(data, dt, window) 
    nbins = len(psd)
    frequencies = calc_psd_frequencies(nbins, dt)
    return NoisePSD(psd,
                         frequencies)

@dataclass
class NoisePSD:
    psd: np.ndarray
    frequencies: np.ndarray
    

    def plot(self, axis: Optional[plt.axis]=None, arb_to_unit_scale_and_label: Tuple[int, str]=(1, "arb"), sqrt_psd: bool=True, loglog: bool=True, **plotkwarg):
        if axis is None:
            plt.figure()
            axis = plt.gca()
        arb_to_unit_scale, unit_label = arb_to_unit_scale_and_label
        psd = self.psd[1:] * (arb_to_unit_scale**2)
        freq = self.frequencies[1:]
        if sqrt_psd:
            axis.plot(freq, np.sqrt(psd), **plotkwarg)
            axis.set_ylabel(f"Amplitude Spectral Density ({unit_label}$/\\sqrt{{Hz}}$)")
        else:
            axis.plot(freq, psd, **plotkwarg)
            axis.set_ylabel(f"Power Spectral Density ({unit_label}$^2$ Hz$^{{-1}}$)")
        if loglog: plt.loglog()
        axis.grid()
        axis.set_xlabel("Frequency (Hz)")
        axis.figure.tight_layout()


