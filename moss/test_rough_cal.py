import moss
import numpy as np


def make_truth_ph(e, e_spurious, e_err_scale, pfit_gain_truth=np.polynomial.Polynomial([6, -1e-6, -1e-10])):
    # return peak heights by inverting a quadratic gain curve and adding energy errors
    cba_truth = pfit_gain_truth.convert().coef
    assert len(cba_truth)==3
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
    ph_truth_with_err = np.array([energy2ph_truth(ee)+np.random.randn()*e_err_scale for ee in e])
    ph_spurious_with_err = np.array([energy2ph_truth(ee)+np.random.randn()*e_err_scale for ee in e_spurious])
    ph = np.hstack([ph_truth_with_err, ph_spurious_with_err])
    return ph, ph_truth_with_err
      

def test_find_optimal_assignment_many():
    np.random.seed(1) # set seed to control shuffle in the function and random errors in make_truth_ph
    e = np.arange(1000,9000,1000) # energies of "real" peaks
    e_spurious = [100, 500, 1500, 1700, 7900, 2200, 2300, 2400, 3300, 3700, 4500] # energies of spurious or fake peaks
    ph, ph_truth_with_err = make_truth_ph(e=e, 
                                            e_spurious=e_spurious, 
                                            e_err_scale=10)
    # ph contains pulseheights of both real and spurious with errs, 
    # first all the real peaks in e order, then all the spurious peaks in e_spurious order
    # ph_truth_with_err contains pulseheights of only real peaks, sorted to match the order of e
    # from all peaks (ph) find the one corresponding to energies e
    np.random.shuffle(ph) # in place shuffle
    result = moss.rough_cal.find_optimal_assignment2(ph, e, map(str, e))
    # test that assigned peaks match known peaks of energy e
    assert np.allclose(result.ph_assigned, ph_truth_with_err, rtol=0.001)
    assert np.allclose([result.ph2energy(result.energy2ph(ee)) for ee in e], e)


def test_find_optimal_assignment_1():
    e = np.array([1000]) # energies of "real" peaks
    e_spurious = [100, 500, 1500, 1700, 7900, 2200, 2300, 2400, 3300, 3700, 4500] # energies of spurious or fake peaks
    ph, ph_truth_with_err = make_truth_ph(e=e, 
                                            e_spurious=e_spurious, 
                                            e_err_scale=10)
    # ph contains pulseheights of both real and spurious with errs
    # ph_truth_with_err contains pulseheights of only real peaks, sorted to match the order of e
    # from all peaks (ph) find the one corresponding to energies e
    result = moss.rough_cal.find_optimal_assignment2(ph, e, map(str, e))
    # test that assigned peaks match known peaks of energy e
    assert np.allclose(result.ph_assigned, ph_truth_with_err, rtol=0.001)
    assert np.allclose([result.ph2energy(result.energy2ph(ee)) for ee in e], e)

def test_find_optimal_assignment_2():
    e = np.array([1000, 3000]) # energies of "real" peaks
    e_spurious = [100, 500, 1500, 1700, 7900, 2200, 2300, 2400, 3300, 3700, 4500] # energies of spurious or fake peaks
    ph, ph_truth_with_err = make_truth_ph(e=e, 
                                            e_spurious=e_spurious, 
                                            e_err_scale=10)
    # ph contains pulseheights of both real and spurious with errs
    # ph_truth_with_err contains pulseheights of only real peaks, sorted to match the order of e
    # from all peaks (ph) find the one corresponding to energies e
    result = moss.rough_cal.find_optimal_assignment2(ph, e, map(str, e))
    # test that assigned peaks match known peaks of energy e
    assert np.allclose(result.ph_assigned, ph_truth_with_err, rtol=0.001)
    assert np.allclose([result.ph2energy(result.energy2ph(ee)) for ee in e], e)


def test_find_optimal_assignment_3():
    e = np.array([1000, 3000, 5000]) # energies of "real" peaks
    e_spurious = [3300, 3700, 4500] # energies of spurious or fake peaks
    ph, ph_truth_with_err = make_truth_ph(e=e, 
                                            e_spurious=e_spurious, 
                                            e_err_scale=10)
    # ph contains pulseheights of both real and spurious with errs
    # ph_truth_with_err contains pulseheights of only real peaks, sorted to match the order of e
    # from all peaks (ph) find the one corresponding to energies e
    result = moss.rough_cal.find_optimal_assignment2(ph, e, map(str, e))
    # test that assigned peaks match known peaks of energy e
    print(f"{result.ph_assigned=}")
    print(f"{ph_truth_with_err=}")
    print(f"{ph=}")
    assert np.allclose(result.ph_assigned, ph_truth_with_err, rtol=0.001)
    assert np.allclose([result.ph2energy(result.energy2ph(ee)) for ee in e], e)



