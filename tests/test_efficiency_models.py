import numpy as np
import uncertainties
from uncertainties import unumpy as unp
from mass2.materials import uncertainties_helpers

import mass2.materials
import pylab as plt


found_xraydb = "xraydb" in mass2.materials.efficiency_models.__dict__


def test_dict():
    if not found_xraydb:
        return
    expected_models = ("EBIT 2018", "Horton 2018", "RAVEN1 2019")
    for k in expected_models:
        assert k in mass2.materials.filterstack_models
    for k, model in mass2.materials.filterstack_models.items():
        assert isinstance(model, mass2.materials.FilterStack), f"mass2.core.Filterstack_models['{k}'] not a mass2.core.FilterStack"


def test_ensure_uncertain():
    a = uncertainties_helpers.ensure_uncertain(1.0)
    assert a.nominal_value == 1
    assert a.std_dev == 1

    b = uncertainties_helpers.ensure_uncertain(uncertainties.ufloat(1, 1))
    assert b.nominal_value == 1
    assert b.std_dev == 1

    c = uncertainties_helpers.ensure_uncertain(np.arange(5))
    assert c[1].nominal_value == 1
    assert c[1].std_dev == 1
    assert len(c) == 5

    d = uncertainties_helpers.ensure_uncertain(unp.uarray(np.arange(5), 0.1 * np.arange(5)))
    assert d[1].nominal_value == 1
    assert d[1].std_dev == 0.1
    assert len(d) == 5

    e = uncertainties_helpers.ensure_uncertain(2)
    assert e.nominal_value == 2
    assert e.std_dev == 2
    f = uncertainties_helpers.with_fractional_uncertainty(np.int64(100), 0.03)
    assert f.nominal_value == 100
    assert f.std_dev == 3

    g = uncertainties_helpers.ensure_uncertain(np.float32(2.0))
    assert g.nominal_value == 2
    assert g.std_dev == 2
    h = uncertainties_helpers.with_fractional_uncertainty(np.float32(100.0), 0.5)
    assert h.nominal_value == 100
    assert h.std_dev == 50

    mixed = np.array([100.0, uncertainties.ufloat(200, 10)], dtype=object)
    assert uncertainties_helpers.ensure_uncertain(mixed) is mixed


def test_filter():
    """Make sure we can compute composite filter QE and that results are reasonable."""
    if not found_xraydb:
        return
    m = mass2.materials.filterstack_models["Horton 2018"]
    e = np.linspace(4000, 9000, 6)
    qe = m(e, uncertain=True)
    maxQE = qe.max()
    assert maxQE > 0.21, r"Horton filter model QE < 21% everywhere"
    assert maxQE < 0.25, r"Horton filter model QE > 25% somewhere"

    assert np.abs(qe[0].nominal_value / 0.07742066727279993 - 1) < 0.001

    def remove_whitespace(s):
        for k in " \n\r\t":
            s = s.replace(k, "")
        return s

    expect = """<class 'mass2.materials.efficiency_models.FilterStack'>(
Electroplated Au Absorber: <class 'mass2.materials.efficiency_models.Filter'>
    (Au 0.00186+/-0.00186 g/cm^2, fill_fraction=1.000+/-0.000, absorber=True)
50mK Filter: <class 'mass2.materials.efficiency_models.Filter'>(Al 0.00135+/-0.00135 g/cm^2,
    Al (1.27+/-1.27)e-06 g/cm^2, O (1.13+/-1.13)e-06 g/cm^2, fill_fraction=1.000+/-0.000, absorber=False)
3K Filter: <class 'mass2.materials.efficiency_models.Filter'>(Al 0.00135+/-0.00135 g/cm^2,
    Al (1.27+/-1.27)e-06 g/cm^2, O (1.13+/-1.13)e-06 g/cm^2, fill_fraction=1.000+/-0.000, absorber=False)
50K Filter: <class 'mass2.materials.efficiency_models.Filter'>(Al 0.00343+/-0.00343 g/cm^2,
    Al (1.27+/-1.27)e-06 g/cm^2, O (1.13+/-1.13)e-06 g/cm^2, fill_fraction=1.000+/-0.000, absorber=False)
Luxel Window TES: <class 'mass2.materials.efficiency_models.FilterStack'>(
    LEX_HT Film: <class 'mass2.materials.efficiency_models.Filter'>(C (6.70+/-0.20)e-05 g/cm^2,
        H (2.60+/-0.08)e-06 g/cm^2, N (7.20+/-0.22)e-06 g/cm^2, O (1.70+/-0.05)e-05 g/cm^2,
        Al (1.70+/-0.05)e-05 g/cm^2, fill_fraction=1.000+/-0.000, absorber=False)
    LEX_HT Mesh: <class 'mass2.materials.efficiency_models.Filter'>(Fe 0.0564+/-0.0011 g/cm^2,
        Cr 0.0152+/-0.0003 g/cm^2, Ni 0.00720+/-0.00014 g/cm^2, Mn 0.000800+/-0.000016 g/cm^2,
        Si 0.000400+/-0.000008 g/cm^2, fill_fraction=0.190+/-0.010, absorber=False)
)
)"""
    assert remove_whitespace(repr(m)) == remove_whitespace(expect)

    x = np.linspace(1000, 10000, 100)
    m.plot_efficiency(x)
    plt.close()
