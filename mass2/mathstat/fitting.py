"""
mass2.mathstat.fitting

Model-fitting utilities.

Joe Fowler, NIST
"""

from numpy.typing import ArrayLike, NDArray
import numpy as np
import scipy as sp

__all__ = ["kink_model", "fit_kink_model"]


def kink_model(k: float, x: ArrayLike, y: ArrayLike) -> tuple[NDArray, NDArray, float]:
    """Compute a kinked-linear model on data {x,y} with kink at x=k.

    The model is f(x) = a+b(x-k) for x<k and f(x)=a+c(x-k) for x>=k, where
    the 4 parameters are {k,a,b,c}, representing the kink at (x,y)=(k,a) and
    slopes of b and c for x<k and x>= k.

    For a fixed k, the model is linear in the other parameters, whose linear
    least-squares values can thus be found exactly by linear algebra. This
    function computes them.

    Returns (model_y, (a,b,c), X2) where:
    model_y is an array of the model y-values;
    (a,b,c) are the best-fit values of the linear parameters;
    X2 is the sum of square differences between y and model_y.

    Parameters
    ----------
    k : float
        Location of the kink, in x coordinates
    x : ArrayLike
        The input data x-values
    y : ArrayLike
        The input data y-values

    Returns
    -------
    model_y, abc, X2) where:
        model_y : NDArray[float]
            an array of the model y-values;
        abc : NDArray[float]
            the best-fit values of the linear parameters;
        X2 : float
            is the sum of square differences between y and model_y.

    Raises
    ------
    ValueError
        if k doesn't satisfy x.min() < k < x.max()
    """
    x = np.asarray(x)
    y = np.asarray(y)
    xi = x[x < k]
    yi = y[x < k]
    xj = x[x >= k]
    yj = y[x >= k]
    N = len(x)
    if len(xi) == 0 or len(xj) == 0:
        xmin = x.min()
        xmax = x.max()
        raise ValueError(f"k={k:g} should be in range [xmin,xmax], or [{xmin:g},{xmax:g}].")

    dxi = xi - k
    dxj = xj - k
    si = dxi.sum()
    sj = dxj.sum()
    si2 = (dxi**2).sum()
    sj2 = (dxj**2).sum()
    A = np.array([[N, si, sj], [si, si2, 0], [sj, 0, sj2]])
    v = np.array([y.sum(), (yi * dxi).sum(), (yj * dxj).sum()])
    abc = np.linalg.solve(A, v)
    model = np.hstack([abc[0] + abc[1] * dxi, abc[0] + abc[2] * dxj])
    X2 = ((model - y) ** 2).sum()
    return model, abc, X2


def fit_kink_model(x: ArrayLike, y: ArrayLike, kbounds: tuple[float, float] | None = None) -> tuple[NDArray, NDArray, float]:
    """Find the linear least-squares solution for a kinked-linear model.

    The model is f(x) = a+b(x-k) for x<k and f(x)=a+c(x-k) for x>=k, where
    the 4 parameters are {k,a,b,c}, representing the kink at (x,y)=(k,a) and
    slopes of b and c for x<k and x>= k.

    Given k, the model is linear in the other parameters, which can thus be
    found exactly by linear algebra. The best value of k is found by use of
    the Bounded method of the sp.optimize.minimize_scalar() routine.

    Parameters
    ----------
    x : ArrayLike
        The input data x-values
    y : ArrayLike
        The input data y-values
    kbounds : Optional[tuple[float, float]], optional
        Bounds on k, by default None.
        If (u,v), then the minimize_scalar is used to find the best k strictly in u<=k<=v.
        If None, then use the Brent method, which will start with (b1,b2) as a search bracket
        where b1 and b2 are the 2nd lowest and 2nd highest values of x.

    Returns
    -------
    model_y, abc, X2) where:
        model_y : NDArray[float]
            an array of the model y-values;
        kabc : NDArray[float]
            the best-fit values of the kink location and the 3 linear parameters;
        X2 : float
            is the sum of square differences between y and model_y.

    Raises
    ------
    ValueError
        if k doesn't satisfy x.min() < k < x.max()

    Examples
    --------
    x = np.arange(10, dtype=float)
    y = np.array(x)
    truek = 4.6
    y[x>truek] = truek
    y += np.random.default_rng().standard_normal(len(x))*.15
    model, (kbest,a,b,c), X2 = fit_kink_model(x, y, kbounds=(3,6))
    plt.clf()
    plt.plot(x, y, "or", label="Noisy data to be fit")
    xi = np.linspace(x[0], kbest, 200)
    xj = np.linspace(kbest, x[-1], 200)
    plt.plot(xi, a+b*(xi-kbest), "--k", label="Best-fit kinked model")
    plt.plot(xj, a+c*(xj-kbest), "--k")
    plt.legend()
    """
    x = np.asarray(x)
    y = np.asarray(y)

    def penalty(k: float, x: NDArray, y: NDArray) -> float:
        _, _, X2 = kink_model(k, x, y)
        return X2

    if kbounds is None:
        kbounds = (x.min(), x.max())
    elif kbounds[0] < x.min() or kbounds[1] > x.max():
        raise ValueError(f"kbounds ({kbounds}) must be within the range of x data")
    optimum = sp.optimize.minimize_scalar(penalty, args=(x, y), method="Bounded", bounds=kbounds)
    kbest = optimum.x
    model, abc, X2 = kink_model(kbest, x, y)
    return model, np.hstack([kbest, abc]), X2
