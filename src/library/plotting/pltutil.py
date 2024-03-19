"""
Utility and convenience functions for plotting.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
from numpy.typing import NDArray


def get_errorbar_lengths(
    median: NDArray, percentiles: NDArray | Sequence[NDArray]
) -> NDArray:
    """
    Return the error bar lengths as (2, N) shape array.

    The percentiles must have shape (2, N) where N is the length of the
    median array. The first entry on axis zero must be the position of
    the lower percentiles, the second the positions of the upper
    percentiles. The returned array has the same shape, but as values
    it holds the length of the respective error bar.


    :param median: Array of median values of length N.
    :param percentiles: Array of lower and upper percentiles. Must be of
        shape (2, N). First entry of first axis must be the position of
        the lower percentiles, second entry that of the upper percentiles.
    :return: Array of shape (2, N) giving the length of the lower and
        upper error bar respectively.
    """
    lower_ebars = median - percentiles[0]
    upper_ebars = percentiles[1] - median
    return np.array([lower_ebars, upper_ebars])
