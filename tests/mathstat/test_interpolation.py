"""
test_interpolation.py

Test that interpolation code works.

25 October 2016
Joe Fowler
"""

import numpy as np
import pytest
import mass2
from mass2.mathstat.interpolate import k_spline, GPRSpline


class Test_SmoothingSpline:
    @staticmethod
    def test_issue74():
        """This is a regression test to ensure that issue #74 is fixed and
        remains fixed.
        """
        ph = np.array([
            604.9186911,
            658.11682861,
            710.25965219,
            761.46157549,
            811.81613372,
            861.40110546,
            910.28210151,
            958.51515166,
            1006.14861411,
        ])
        e = np.array([
            3000.0,
            3333.33333333,
            3666.66666667,
            4000.0,
            4333.33333333,
            4666.66666667,
            5000.0,
            5333.33333333,
            5666.66666667,
        ])
        de = np.array([0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01])
        dph = np.array([0.60491869, 0.65811683, 0.71025965, 0.76146158, 0.81181613, 0.86140111, 0.9102821, 0.95851515, 1.00614861])

        # At time of issue #74, this crashed on next line for Joe, but not for Galen.
        mass2.mathstat.interpolate.SmoothingSplineLog(ph, e, de, dph)

        cal = mass2.calibration.EnergyCalibrationMaker.init(e**0.8, e, dph, de)
        # At time of issue #74, this crashed on next line for Galen, but not for Joe.
        cal.drop_one_errors()

    @staticmethod
    def test_three_points():
        """With 3 knots the spline must still be natural and match its reported chi-squared."""
        x, y = np.array([1.0, 2, 4]), np.array([0.0, 5, 1])
        ss = mass2.mathstat.interpolate.SmoothingSpline(x, y, np.ones(3))
        assert ss.actualchisq == pytest.approx(3.0)
        assert np.sum((y - ss(x)) ** 2) == pytest.approx(3.0)
        assert ss(4.0, der=2) == pytest.approx(0.0, abs=1e-12)

    @staticmethod
    def test_unsorted_and_integer_x():
        """Integer and unsorted x give the same spline as sorted float x."""
        x = np.arange(1, 11)
        y = np.sin(x)
        dy = np.full(10, 0.1)
        perm = np.random.default_rng(1).permutation(10)
        xtest = np.linspace(0, 12, 30)
        ref = mass2.mathstat.interpolate.SmoothingSpline(x.astype(float), y, dy)(xtest)
        assert np.allclose(mass2.mathstat.interpolate.SmoothingSpline(x, y, dy)(xtest), ref)
        assert np.allclose(mass2.mathstat.interpolate.SmoothingSpline(x[perm] * 1.0, y[perm], dy[perm])(xtest), ref)

    @staticmethod
    def test_line_meets_chisq():
        """When the weighted line meets maxchisq, the spline is that line, also when extrapolated."""
        x = np.arange(1.0, 11.0)
        for dy in (1.0, 1e-9):
            ss = mass2.mathstat.interpolate.SmoothingSpline(x, np.full(10, 5.0), np.full(10, dy))
            assert np.allclose(ss(np.array([-5.0, 5.0, 20.0])), 5.0)
            assert np.allclose(ss(np.array([-5.0, 5.0, 20.0]), der=1), 0.0)


class Test_GPR:
    @staticmethod
    def test_spline_covar():
        for x in np.linspace(0, 10):
            assert x**3 / 3 == k_spline(x, x)
            assert k_spline(x, 5.5) == k_spline(5.5, x)

    @staticmethod
    def test_gprspline():
        x = np.linspace(2, 10, 9)
        s = 1.0
        delta = np.array([
            -0.08414947,
            0.25100057,
            0.70287457,
            -0.9225354,
            -0.56127467,
            0.99469994,
            0.50381756,
            -0.53460321,
            -0.49538835,
        ])

        def actualf(x):
            return 25 - 10 * x + x * x

        y = actualf(x) + s * delta
        dy = np.ones_like(y)
        spl = GPRSpline(x, y, dy)
        yspl = spl(x)
        assert np.mean((yspl - y) ** 2) < s**2

        xtest = np.array([0, 5, 7, 10, 12, 15])
        var = spl.variance(xtest)
        assert (var[np.logical_and(xtest > x[0] - 1, xtest < x[-1] + 1)] < 2).all()
        allowed_diff = 2 * np.sqrt(var)
        for x, ad in zip(xtest, allowed_diff):
            assert abs(spl(x) - actualf(x)) < ad

    @staticmethod
    def test_gprspline_scalar_variance():
        x = np.linspace(2, 10, 9)
        g = GPRSpline(x, 25 - 10 * x + x * x + np.sin(7 * x), np.ones(9))
        assert g.variance(5.0) == g.variance([5.0])[0]
        assert g.covariance(5.0) == g.variance([5.0])[0]

    @staticmethod
    def test_gprspline_shift_invariant():
        x = np.linspace(2, 10, 9)
        y = 25 - 10 * x + x * x + np.sin(7 * x)
        g = GPRSpline(x, y, np.ones(9), sigmaf=1.0)
        h = GPRSpline(x - 20, y, np.ones(9), sigmaf=1.0)
        xtest = np.array([1.0, 4.5, 11.0])
        assert np.allclose(h(xtest - 20), g(xtest), rtol=0, atol=1e-8)
        assert np.allclose(h.covariance(xtest - 20), g.covariance(xtest), rtol=0, atol=1e-8)
        k = GPRSpline(x + 1e5, y, np.ones(9), sigmaf=1.0)
        assert np.allclose(k(xtest + 1e5), g(xtest), rtol=0, atol=1e-6)
        assert np.allclose(k.covariance(xtest + 1e5), g.covariance(xtest), rtol=0, atol=1e-6)

    @staticmethod
    def test_gprspline_unsigned_knots():
        x = np.array([1000, 2000, 3500, 5000, 6400, 8000], dtype=np.uint16)
        y = np.array([1.0, 2.1, 3.4, 5.2, 6.1, 8.3])
        g = GPRSpline(x.astype(float), y, 0.1 * np.ones(6), sigmaf=0.01)
        h = GPRSpline(x, y, 0.1 * np.ones(6), sigmaf=0.01)
        xtest = np.array([1500.0, 4000.0, 7000.0])
        assert np.allclose(h(xtest), g(xtest))
        assert np.allclose(h.variance(xtest), g.variance(xtest))
