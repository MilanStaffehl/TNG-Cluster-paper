"""
Functions for processing gas temperatures.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal, Sequence

import numpy as np

import units

if TYPE_CHECKING:
    from numpy.typing import NDArray


def get_temperature_distribution_histogram(
    gas_data: dict[str, NDArray],
    weight: Literal["frac", "mass"],
    n_bins: int | Sequence[float] = 50,
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
        and corresponding arrays as values. 'Masses' must be in computational
        units and convertible by ``units.UnitConverter.convert``.
        'Temperature' must be given in units of Kelvin.
    :param weight: What weighting to choose. Can either be 'mass' for
        weighting the temperatures by gas cell mass or 'frac' for weighting
        the temperatures by gas fraction of the cell.
    :param n_bins: The number of bins for the histogram. Optionally, this
        can also be a sequence of floats for non-uniform bins. Defaults
        to 50.
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
        weights = units.UnitConverter.convert(gas_data["Masses"], "Masses")

    # generate and assign hist data
    hist, _ = np.histogram(
        np.log10(gas_data["Temperature"] / normalization),
        bins=n_bins,
        range=log_temperature_range,
        weights=weights,
    )
    return hist


def get_temperature_2d_histogram(
    gas_data: dict[str, NDArray],
    x_axis_field: str,
    ranges: NDArray,
    n_bins_temperature: int = 50,
    n_bins_x_axis: int = 50,
    convert_units: bool = True,
    normalization_t: float = 1,
    normalization_x: float = 1,
) -> NDArray:
    """
    Return a 2D histogram of temperature and another field.

    The values are weighted by gas fraction of every cell. Optionally,
    the axes can be normalized by a singular value. The temperature
    axis will be logarithmic.

    :param gas_data: A dictionary with keys 'Temperature', 'Masses' and
        whatever field ``x_axis_field`` specifies. The values must be
        arrays holding the corresponding values.
    :param x_axis_field: Name of the field to use on the x-axis.
    :param ranges: An array of shape (2, 2) giving the min and max value
        for the two axes as ``[[xmin, xmax], [tmin, tmax]]``. Note that
        temperatures are treated in log10, so ymin and ymax must be in
        units of log10 K (i.e. a temperature range from 10^3 to 10^8
        must be given as [3, 8]).
    :param n_bins_temperature: The number of bins for the temperature
        axis of the histogram. Defaults to 50.
    :param n_bins_x_axis: The number of bins for the second field on the
        x-axis. Defaults to 50.
    :param convert_units: Whether to attempt a unit conversion for the
        x-axis field. Defaults to True.
    :param normalization_t: A value by which to normalize the temperature.
        Defaults to 1 (no normalization).
    :param normalization_x: A value by which to normalize the x-axis
        values. Defaults to 1 (no normalization). Values are normalized
        AFTER unit conversion, so the normalization must have the correct
        physical units and value.
    :return: An array of shape (T, X) representing the gas mass fraction
        weighted 2D histogram data. T is the number of temperature bins
        and X the number of bins on the x-axis, given by
        ``n_bins_temperature`` and ``n_bins_x_axis`` respectively. Note
        that along the first axis, the order is such that the first row
        holds the values for the lowest y-bin, and the last row that of
        the highest y-bin (i.e. printing the resulting array results in
        a matrix form where the y-axis is inverted, pointing down). Note
        also that this is the transposed array from what ``np.histogram2d``
        returns normally (axes are switched).
    """
    if x_axis_field not in gas_data.keys():
        logging.error(
            f"The chosen field {x_axis_field} is not in the gas data "
            f"dictionary. Returning array of NaN instead."
        )
        fallback = np.empty((n_bins_x_axis, n_bins_temperature))
        fallback.fill(np.nan)
        return fallback

    # attempt unit conversion:
    if convert_units:
        xfield = units.UnitConverter.convert(
            gas_data[x_axis_field], x_axis_field
        )
    else:
        xfield = gas_data[x_axis_field]
    # calculate weights
    total_gas_mass = np.sum(gas_data["Masses"])
    weights = gas_data["Masses"] / total_gas_mass

    # calculate histogram
    hist, _, _ = np.histogram2d(
        xfield / normalization_x,
        np.log10(gas_data["Temperature"] / normalization_t),
        bins=[n_bins_x_axis, n_bins_temperature],
        range=ranges,
        weights=weights,
    )
    return hist.transpose()
