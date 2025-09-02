"""
Helpers for dealing with `uncertainties` scalars and arrays.
"""

import uncertainties
from uncertainties import unumpy as unp
import numpy as np
from typing import Any


def is_uncertain_scalar(x: Any) -> bool:
    """Is x a value with uncertainties?

    Parameters
    ----------
    x : Any
        A scalar

    Returns
    -------
    bool
        Whether it has the attribute "nominal_value", and hence is an `uncertainties` scalar.
    """
    return hasattr(x, "nominal_value")


def ensure_uncertain(x: Any) -> Any:
    """Return the argument, in a form that has uncertainties.

    * Given a scalar, returns a ufloat with 100% uncertainty
    * Given a numpy array of scalars, return a uarray with 100% uncertainty
    * Given a ufloat or uarray, return it unchanged

    The default uncertainty will be 100%, so people will know not to take it seriously until they've
    put it in manually. But if the argument is already uncertain, it will be returned unchanged.

    Parameters
    ----------
    x : Any
        The scalar or vector to be made uncertain

    Returns
    -------
    Any
        Either x itself, or an uncertain version of it (with ±100% uncertainty)

    Raises
    ------
    ValueError
        If type of `x` is unsupported
    """
    if isinstance(x, np.ndarray):
        if is_uncertain_scalar(x[0]):
            return x
        return unp.uarray(x, x)
    if isinstance(x, float):
        return uncertainties.ufloat(x, x)
    if is_uncertain_scalar(x):
        return x
    else:
        raise ValueError(f"{x} of type {type(x)} not supported")


def with_fractional_uncertainty(x: Any, fractional_uncertainty: float) -> Any:
    """Return a version of x with the given fractional uncertainty

    Parameters
    ----------
    x : Any
        The scalar or vector to make uncertain
    fractional_uncertainty : float
        The relative uncertainty to impose

    Returns
    -------
    Any
        An uncertain version of x

    Raises
    ------
    ValueError
        If type of `x` is unsupported
    """
    if isinstance(x, float):
        return uncertainties.ufloat(x, x * fractional_uncertainty)
    if isinstance(x, np.ndarray):
        return unp.uarray(x, x * fractional_uncertainty)
    else:
        raise ValueError(f"{x} of type {type(x)} not supported")
