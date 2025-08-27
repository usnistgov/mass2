import numpy as np
from numpy.typing import ArrayLike, NDArray
import numba


@numba.njit
def python_nearest_arrivals(reference_times: ArrayLike, other_times: ArrayLike) -> tuple[NDArray, NDArray]:
    """Identical to nearest_arrivals(...)."""
    reference_times = np.asarray(reference_times)
    other_times = np.asarray(other_times)
    nearest_after_index = np.searchsorted(other_times, reference_times)
    last_index = np.searchsorted(nearest_after_index, other_times.size, side="left")
    first_index = np.searchsorted(nearest_after_index, 1)

    nearest_before_index = np.copy(nearest_after_index)
    nearest_before_index[:first_index] = 1
    nearest_before_index -= 1
    before_times = reference_times - other_times[nearest_before_index]
    before_times[:first_index] = np.inf

    nearest_after_index[last_index:] = other_times.size - 1
    after_times = other_times[nearest_after_index] - reference_times
    after_times[last_index:] = np.inf

    return before_times, after_times
