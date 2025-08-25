## Fluorescence Lines

Mass includes numerous features to help you analyze and model the fluorescence emission of various elements. Mass can

1. Approximate the shape of the fluorescence line emission for certain lines (particularly the K-alpha and K-beta lines of elements from Mg to Zn, or Z=12 to 30).
2. Generate random deviates, drawn from these same energy distributions.
3. Fit a measured spectrum to on of these energy distributions.

### Examples

#### 1. Plot the distribution

Objects of the `SpectralLine` class are callable, and return their PDF given the energy as an array or scalar argument.
```python
# mkdocs: render
import mass2
import numpy as np
import pylab as plt
spectrum = mass2.spectra["MnKAlpha"]
plt.clf()
axis=plt.gca()
cm = plt.cm.magma
for fwhm in (3,4,5,6,8,10):
    spectrum.plot(axis=axis,components=False,label=f"FWHM: {fwhm} eV", setylim=False,
    instrument_gaussian_fwhm=fwhm, color=cm(fwhm/10-0.2));
plt.legend(loc="upper left")
plt.title("Mn K$\\alpha$ distribution at various resolutions")
plt.xlabel("Energy (eV)")
```

#### 2. Generate random deviates from a fluorescence line shape

Objects of the `SpectralLine` class roughly copy the API of the scipy type `scipy.stats.rv_continuous` and offer some of the methods, such as `pdf`, `rvs`.:

```python
# mkdocs: render
energies0 = spectrum.rvs(size=20000, instrument_gaussian_fwhm=0)
energies3 = spectrum.rvs(size=20000, instrument_gaussian_fwhm=3)
energies6 = spectrum.rvs(size=20000, instrument_gaussian_fwhm=6)

plt.clf()
erange=(5840, 5940)
for E, color in zip((energies0, energies3, energies6), ("r", "b", "purple")):
    contents, bin_edges, _ = plt.hist(E, 200, range=erange, histtype="step", color=color)
plt.xlabel("Energy (eV)")
plt.ylabel("Counts per bin")
plt.xlim((erange[0], erange[1]))
```

#### 3. Fit data to a fluorescence line model
```python
# mkdocs: render
model = mass2.spectra["MnKAlpha"].model()
contents3, bins = np.histogram(energies3, 200, range=erange)
bin_ctr = bins[:-1]  + 0.5 * (bins[1] - bins[0])
guess_params = model.guess(contents3, bin_ctr, dph_de=1.0)
result = model.fit(contents3, guess_params, bin_centers=bin_ctr)
result.plotm()
print(result.best_values)
fwhm = result.params["fwhm"]
print(f"Estimated resolution (FWHM) = {fwhm.value}±{fwhm.stderr}")
```
