import moss

def test_rank_combinatoric_assignments():
    import numpy as np

    # generate some truth data using a quadratic gain polynomial
    np.random.seed(0)
    pfit_gain_truth = np.polynomial.Polynomial([6, -1e-6, -1e-10])
    # e=(1011.77, 1486.708, 2622.44, 4090.735, 5898.801, 8638.906)
    # line_names=('ZnLAlpha', 'AlKAlpha', 'ClKAlpha', 'ScKAlpha', 'MnKAlpha', 'ZnKAlpha')
    e = np.arange(1000,9000,1000)
    line_names = [str(ee) for ee in e]
    cba_truth = pfit_gain_truth.convert().coef
    def energy2ph_truth(energy):
            # ph2energy is equivalent to this with y=energy, x=ph
            # y = x/(c + b*x + a*x^2)
            # so
            # y*c + (y*b-1)*x + a*x^2 = 0
            # and given that we've selected for well formed calibrations,
            # we know which root we want
            c,bb,a = cba_truth*energy
            b=bb-1
            ph = (-b-np.sqrt(b**2-4*a*c))/(2*a)
            return ph
    ph_truth = np.array([energy2ph_truth(ee) for ee in e])
    e_err_scale = 10
    ph_with_errs = np.array([energy2ph_truth(ee+np.random.randn()*e_err_scale) for ee in e])
    ph_extra = [energy2ph_truth(ee) for ee in [100, 500, 1500, 1700, 7900, 2200, 2300, 2400, 3300, 3700, 4500]]
    ph = np.hstack([ph_extra, ph_with_errs])
    np.random.shuffle(ph)

    # test peak assignment
    rms_e_residual, pha = moss.rough_cal.find_optimal_assignment(ph, e)
    assert np.allclose(pha, ph_with_errs, rtol=0.001)