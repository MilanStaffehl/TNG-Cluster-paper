"""
Functions for processing gas temperatures.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


def get_temperature_distribution_histogram(
    gas_data: dict[str, NDArray],
    weight: str,
    n_bins: int = 50,
    log_temperature_range: tuple[float, float] = (3.0, 8.0),
    normalization: float = 1.0,
) -> NDArray:
    """
    Return the histogram of the temperature distribution.

    Depending on the weight chosen, the histogram may either weight the
    temperatures by gas fraction or by gas cell mass. The histogram is
    calculated for the decadic logarithm of temperature, i.e. the bins
    will be logarithmic.

    :param gas_data: A dictionary with keys 'Masses' and 'Temperatures'
        and corresponding arrays as values.
    :param weight: What weighting to choose. Can either be 'mass' for
        weighting the temperatures by gas cell mass or 'frac' for weighting
        the temperatures by gas fraction of the cell.
    :param n_bins: The number of bins for the histogram. Defaults to 50.
    :param log_temperature_range: The range of temperatures to consider,
        in log10 scale, i.e. a temperature range of 10^3 Kelvin to 10^8
        Kelvin would be specified as (3.0, 8.0). If temperature is
        normalized, the values are in dex, i.e. a range (-4, +4)
        corresponds to normalized temperatures from 10^-4 to 10^4.
        Defaults to (3.0, 8.0).
    :param normalization: A value to which the temperature is normalized
        BEFORE the logarithm is taken, i.e. the histogram bins are
        taken of the quantity ``temperature / normalization``. Defaults
        to 1 which means no normalization.
    :return: The histogram data as the height of the individual bins.
        Note that the bins are logarithmic, not linear.
    """
    # determine weights for hist
    if weight == "frac":
        total_gas_mass = np.sum(gas_data["Masses"])
        weights = gas_data["Masses"] / total_gas_mass
    else:
        weights = gas_data["Masses"]

    # generate and assign hist data
    hist, _ = np.histogram(
        np.log10(gas_data["Temperature"] / normalization),
        bins=n_bins,
        range=log_temperature_range,
        weights=weights,
    )
    return hist
