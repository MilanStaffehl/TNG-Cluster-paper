"""
Tools for statistics with temperature and gas cell data.
"""
from __future__ import annotations

import logging
from typing import Literal, Sequence, TypeVar

import numpy as np
import numpy.ma as ma
import scipy.stats
from numpy.typing import NDArray  # MUST be imported for type definition!

from library.processing import selection

Hist2D = TypeVar("Hist2D", bound=NDArray)


def nanpercentiles(a: NDArray, axis: int):
    """
    Wrapper around numpys ``nanpercentile`` function with fixed percentiles.

    Function returns the 16th and 84th percentiles of the given set of
    values ``a``. The shape of the output array is dependent on the
    shape of ``a`` and the choice of ``axis``. See the numpy documentation
    for details.

    This function is useful as it has the same signature as the other
    common numpy error function ``np.nanstd`` and can therefore be
    used almost interchangeably with it, notwithstanding the different
    return shape.

    :param a: Array of values.
    :param axis: The axis along which to compute the percentiles.
    :return: Array of shape (2, ...) containing the 16th and 84th
        percentile respectively.
    """
    return np.nanpercentile(a, np.array([16, 84]), axis=axis)


def stack_histograms(
    histograms: NDArray | Sequence[NDArray],
    method: Literal["median", "mean"],
) -> tuple[NDArray, NDArray, NDArray]:
    """
    Stack a set of histograms with the given method, including errors.

    The function takes a sequence of arrays, all representing ND
    histograms and computes either the mean or the median of these
    histograms bin-wise. It additionally provides error estimates for
    both cases. In the case of the mean, the error is simply the
    standard deviation; in the case of the median it is the 16th and 84th
    percentile, taken per bin.

    The function returns the stack as well as an estimate for the lower
    and upper error of every bin in the form of a tuple of arrays.

    The histograms may be of arbitrary dimensionality; the stack is
    always performed along the first axis of ``histograms``.

    :param histograms: A sequence or array of arrays of the same shape,
        representing histograms.
    :param method: The method to stack the histograms, can only be
        'mean' or 'median'.
    :return: Tuple of three arrays. First is the stack of the histograms,
        second the lower error, third the upper error. For method 'mean',
        the two error arrays are identical and represent the standard
        deviation. For 'median', the represent the 16th and 84th
        percentile respectively.
    """
    # stack histograms according to chosen method
    if method == "mean":
        stack_func = np.nanmean
        err_func = np.nanstd
    elif method == "median":
        stack_func = np.nanmedian
        err_func = nanpercentiles
    else:
        raise KeyError(f"Unknown stacking method: {method}.")
    # stack histograms
    stack = stack_func(histograms, axis=0)
    error = err_func(histograms, axis=0)
    # return the ordered results
    if method == "mean":
        return stack, error, error
    else:
        return stack, error[0], error[1]


def stack_histograms_per_mass_bin(
    histograms: NDArray,
    n_mass_bins: int,
    mass_bin_mask: NDArray,
) -> tuple[NDArray, NDArray, NDArray] | None:
    """
    Stack all histograms per mass bin into an average histogram.

    The method will average all histograms in every mass bin and
    return the resulting average histogram data. It also calculates the
    median and 18th and 84th percentiles of the bins and returns them
    alongside the mean.

    The input data must be an array of histogram arrays, with a
    corresponding masking array, assigning every histogram to a mass bin
    number.

    :param histograms: Array of shape (N, T) where N is the number of
        halos in the simulaton and T is the number of temperature bins
        of every histogram. Invalid histograms are expected to be filled
        with ``np.nan``.
    :param n_mass_bins: The number of mass bins.
    :param mass_bin_mask: A mask asigning every histogram in ``histograms``
        to a mass bin. This can be obtained from
        :func:``sort_masses_into_bins`. Every entry must be a number,
        assigning the corresponding histogram of the same array index to
        a mass bin.
    :return: A tuple of NDArrays, with the first being an array of shape
        (M, T) where M is the number of mass bins set by ``n_mass_bins``
        and T is the number of temperature bins, containing the mean
        histogram for every mass bin. The second array contains the
        median and the third has shape (M, 2, T), containing 16th and
        84th percentile of every histogram in the mass bin.
    """
    logging.info("Start post-processing of data (stacking hists).")
    n_halos, n_temperature_bins = histograms.shape
    if len(mass_bin_mask) != n_halos:
        logging.error(
            f"The number of halos ({n_halos}) does not match the length of "
            f"the masking array ({len(mass_bin_mask)})"
        )
        return
    histograms_mean = np.zeros((n_mass_bins, n_temperature_bins))
    histograms_median = np.zeros_like(histograms_mean)
    histograms_percentiles = np.zeros((n_mass_bins, 2, n_temperature_bins))
    for bin_num in range(n_mass_bins):
        # mask histogram data
        mask = np.where(mass_bin_mask == bin_num + 1, 1, 0)
        masked_hists = ma.masked_array(histograms).compress(mask, axis=0)
        # masked arrays need to be compressed into standard arrays
        halo_hists = masked_hists.compressed().reshape(masked_hists.shape)
        histograms_mean[bin_num] = np.nanmean(halo_hists, axis=0)
        histograms_median[bin_num] = np.nanmedian(halo_hists, axis=0)
        histograms_percentiles[bin_num] = np.nanpercentile(
            halo_hists,
            (16, 84),
            axis=0,
        )
        # diagnostics
        logging.debug(
            f"Empty halos in mass bin {bin_num}: "
            f"{np.sum(np.any(np.isnan(halo_hists), axis=1))}"
        )

    logging.info("Finished post-processing data.")
    return histograms_mean, histograms_median, histograms_percentiles


def stack_2d_histograms_per_mass_bin(
    histograms: Sequence[Hist2D],
    n_mass_bins: int,
    mass_bin_mask: NDArray,
) -> NDArray | None:
    """
    Stackk all 2D histograms per mass bin into an average histogram.

    The function will average all histograms in every mass bin and
    return the resulting average histogram data. Function also calculates
    the running average across all x-axis bins and returns alongside the
    averaged histograms.

    :param histograms: An array of 2D arrays of shape (N, X, Y) where
        N is the total number of halos, X is the number of x-axis bins
        in the 2D histograms and Y is the number of y-bins.
    :param n_mass_bins: The number of mass bins.
    :param mass_bin_mask: A mask asigning every histogram in ``histograms``
        to a mass bin. This can be obtained from
        :func:``sort_masses_into_bins``. Every entry must be a number,
        assigning the corresponding histogram of the same array index to
        a mass bin.
    :return: An array of shape (M, X, Y) where M is the number of mass
        bins, containing the averaged histograms for every mass bin. If
        the given number of bins does not match the histogram shape,
        returns None instead.
    """
    logging.info("Stacking 2D histograms for every mass bin.")
    n_halos, n_x_bins, n_y_bins = histograms.shape
    if len(mass_bin_mask) != n_halos:
        logging.error(
            f"The number of halos ({n_halos}) does not match the length of "
            f"the masking array ({len(mass_bin_mask)})"
        )
        return
    histograms_mean = np.zeros((n_mass_bins, n_x_bins, n_y_bins))
    for bin_num in range(n_mass_bins):
        # mask histogram data
        mask = np.where(mass_bin_mask == bin_num + 1, 1, 0)
        masked_hists = ma.masked_array(histograms).compress(mask, axis=0)
        # masked arrays need to be compressed into standard arrays
        halo_hists = masked_hists.compressed().reshape(masked_hists.shape)
        histograms_mean[bin_num] = np.nanmean(halo_hists, axis=0)
        # diagnostics
        logging.debug(
            f"Empty halos in mass bin {bin_num}: "
            f"{np.sum(np.any(np.isnan(halo_hists), axis=1))}"
        )

    logging.info("Finished post-processing data.")
    return histograms_mean


def get_2d_histogram_running_average(
    histogram: NDArray, yrange: tuple[float, float]
) -> NDArray:
    """
    Return the running average of the given 2D histogram.

    The function calculates, for every column of the 2D histogram given,
    the weighted average of its y-values with the weights being the
    histogram values of this column. The function returns the array of
    these averages for every column.

    :param histogram: Array of shape (Y, X), where Y is the number of
        bins on the y-axis and X is the number of bins on the x-axis.
        Must contain values of the histogram.
    :param yrange: The minimum and maximum values of the y-axis bins,
        i.e. the lower edge of the smallest y bin and the upper edge
        of the largest y bin.
    :return: Array of shape (X, ) containing the weighted average of
        every column in the histogram.
    """
    n_ybins = histogram.shape[0]
    ybin_width = abs(yrange[1] - yrange[0]) / n_ybins
    ybin_centers = np.min(yrange) + np.arange(.5, n_ybins + .5, 1) * ybin_width
    # Calculate the weighted average for every column: start by multiplying
    # every entry with its corresponding y-value:
    hist_weighted = (histogram.transpose() * ybin_centers).transpose()
    # Sum the weighted values for every column
    column_sum = np.sum(hist_weighted, axis=0)
    # Finally, get the actual average by normalizing it to the sum of the
    # weights of the colum
    return column_sum / np.sum(histogram, axis=0)


def get_binned_averages(
    values: NDArray, bin_mask: NDArray, n_bins: int = -1
) -> NDArray | None:
    """
    Return the averages and stds of the values in the specified bins.

    Function bins the values according to the bin mask provided and then
    finds, in every bin, the average values as well as their standard deviation.
    It returns an array of shape (2, N) with these values.

    :param values: Array of values of shape (N,).
    :param bin_mask: Bin mask, assigning to every array index a bin index.
        Can be obtained for example through ``np.digitize``.
    :param n_bins: The number of bins to consider, starting from 1. All
        bins with indices greater than this index are ignored. Defaults
        to -1, which means the bin number will be determined automatically
        as the highest index in ``bin_mask``.
    :return: Array of shape (3, N). First entry is the average, the second
        two are the standard deviation twice (for compatability with
        asymmetric binning functions that return upper and lower value).
    """
    if not values.shape == bin_mask.shape:
        logging.error(
            f"Received arrays of different shapes: values have shape "
            f"{values.shape}, bin mask has shape {bin_mask.shape}."
        )
        return
    avg = []
    std = []
    for binned_values in selection.bin_quantity(values, bin_mask, n_bins):
        avg.append(np.nanmean(binned_values))
        std.append(np.nanstd(binned_values))
    return np.array([np.array(avg), np.array(std), np.array(std)])


def get_binned_medians(
    values: NDArray, bin_mask: NDArray, n_bins: int = -1
) -> NDArray | None:
    """
    Return the median and 1 sigma of the values in the specified bins.

    Function bins the values according to the bin mask provided and then
    finds, in every bin, the median values as well as the one-sigma error
    on the median in the form of an errorbar length  (i.e. the function
    does not return the *position value* of the percentiles, but the
    *distance from the median* of the interval edges).

    :param values: Array of values of shape (N,).
    :param bin_mask: Bin mask, assigning to every array index a bin index.
        Can be obtained for example through ``np.digitize``.
    :param n_bins: The number of bins to consider, starting from 1. All
        bins with indices greater than this index are ignored. Defaults
        to -1, which means the bin number will be determined automatically
        as the highest index in ``bin_mask``.
    :return: Array of shape (3, N). First entry is the median, the second
        two are the lower and upper errors on the median, taken to be
        the 16th and 84th percentiles. Note that these values are not the
        percentiles themselves but the difference ``median - percentile``
        so that they may be directly used as errorbar lengths in plotting.
    """
    if not values.shape == bin_mask.shape:
        logging.error(
            f"Received arrays of different shapes: values have shape "
            f"{values.shape}, bin mask has shape {bin_mask.shape}."
        )
        return
    med = []
    lper = []  # lower percentiles
    uper = []  # upper percentiles
    for binned_values in selection.bin_quantity(values, bin_mask, n_bins):
        med.append(np.nanmedian(binned_values))
        lper.append(np.nanpercentile(binned_values, 16))
        uper.append(np.nanpercentile(binned_values, 84))
    lerr = np.abs(np.array(med) - np.array(lper))  # error below median
    uerr = np.abs(np.array(med) - np.array(uper))  # error above median
    return np.array([np.array(med), lerr, uerr])


def column_normalized_hist2d(
    x: NDArray,
    y: NDArray,
    bins: int | tuple[int, int] | NDArray | tuple[NDArray, NDArray],
    values: NDArray | None = None,
    ranges: NDArray | None = None,
    statistic: str = "sum",
    normalization: Literal["density", "range"] = "density",
) -> tuple[Hist2D, NDArray, NDArray] | None:
    """
    Return a 2D histogram normalized column-wise.

    Function takes a set of data points with positions given by ``x``
    and ``y`` and corresponding values and creates a 2D histogram of
    these values, which is normalized at every x bin, meaning that for
    a fixed x value, the values in the y bins along the y-axis are
    normalized. The exact type of normalization depends on the choice of
    ``normalization``:

    - ``density``: The histogram will be normalized such that the values
      of every column will add up to one, i.e. for a fixed x, all bin
      values summed up along the y-axis will be equal to one.
    - ``range``: The histogram will be normalized such that in every
      column, the maximum value is normalized to one, i.e. for fixed x
      every bin value along the x-axis will lie between 0 and 1 with
      the largest value at this x being equal to 1. Not that this does
      not assign the smallest value in the column to zero; it merely
      normalizes every value to the largest one in the column.

    The function also supports different types of bin statistics. The
    options are identical to those of ``scipy.stats.binned_statistics_2d``,
    see the`scipy documentation`_ for details. In order to get a normal
    weighted histogram, use the weights as values and set "statistics"
    to ``"sum"``. In order to get a count histogram, leave both the
    values and the statistics arguments at their default values.

    .. attention:: The histogram will have shape (ny, nx) - contrary to
        the standard order of most histogram generating functions! This
        is done to ensure that the array will have the more intuitive,
        readily understandable form wherein the first index selects a
        row, and the second index selects a column (i.e. an entry of the
        selected row). This shape is also expected for plotting functions
        such as ``mtplotlib.pyplot.imshow``. To return to the original
        shape, simply transpose the histogram array: ``hist.transpose()``.

    :param x: The array of shape (N, ) of x-positions of the data points.
    :param y: The array of shape (N, ) of y-positions of the data points.
    :param bins: The bins for the histogram. Can be one of the following:
        - int: In this case, both dimensions will be split into this
          number of bins.
        - tuple[int, int]: The histogram will be split into bins according
          to the two numbers given with (nx, ny) bins.
        - NDArray: The array will determine the bin edges in both
          dimensions.
        - tuple[NDArray, NDArray]: First array will specify the bin edges
          along the x-axis, the second the bin edges along the y-axis.
    :param values: The values belonging to each data point. Must be of
        shape (N, ). Optional, leave empty for a simple count statistic.
        Defaults to None, which means it will automatically be replaced
        by an array of ones of shape (N, ).
    :param ranges: The ranges along the x- and y-axis. Values outside of
        these ranges are ignored. Defaults to None, which means the
        ranges are automatically determined and will include all points.
    :param statistic: The bin statistic to use. See the
        `scipy documentation`_ for details. Defaults to "sum".
    :param normalization: The normalization to use along the columns.
        Choices are "density" or "range". Density normalization will
        normalize every column such that its values add up to one, while
        schoosing range will normalize every column to its maximum value,
        such that every column will have 1 as its maximum value. Defaults
        to "density".
    :raises RuntimeError: If an unsupported normalization is given.
    :return: The tuple of the histogram, the x-edges and the y-edges.
        The histogram is column-wise normalized according to the chosen
        method.

    .. _scipy documentation: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binned_statistic_2d.html
    """  # noqa: B950
    if x.shape != y.shape:
        logging.error(
            f"Received x and y data arrays of different shape: shape of x is "
            f"{x.shape} but y has shape {y.shape}."
        )
        return
    # if no values are given, assume a normal count/sum is desired
    if values is None:
        values = np.ones_like(x)

    # calculate histogram
    hist, xedges, yedges, _ = scipy.stats.binned_statistic_2d(
        x, y, values, statistic, bins, ranges
    )

    # normalize every column according to chosen normalization
    if normalization == "density":
        column_sums = np.sum(hist, axis=1)
        # broadcast column sum array to appropriate shape
        hist = np.divide(hist, column_sums[:, np.newaxis])
    elif normalization == "range":
        column_max = np.max(hist, axis=1)
        # broadcast column max to appropriate shape
        hist = np.divide(hist, column_max[:, np.newaxis])
    else:
        raise RuntimeError(f"Unsupported normalization {normalization}.")

    return hist.transpose(), xedges, yedges


def volume_normalized_radial_profile(
    radial_distances: NDArray,
    weight: NDArray,
    bins: int | NDArray,
    virial_radius: float | None = None
) -> tuple[NDArray, NDArray]:
    """
    Generate a radial profile, normalized by shell volume.

    The function generates a radial profile histogram of the quantity
    ``weight`` by binning it into radial bins and summing the weights
    per bin. The value of this sum is then normalized by the shell
    volume of the corresponding radial bin. This means that if the
    weighted sum in a bin is w, the function will calculate

    .. math::

        z = w / (\\frac{4}{3} \\pi (R_r^3 - R_l^3))

    where R_r and R_l are the right and left edge of the radial bin
    respectively. If the edges are given in units of the virial radius,
    specifying the virial radius will return them into a physical unit
    (namely the unit in which the virial radius is given). The unit of
    the x-axis however is always kept as it is given (i.e. even if a
    virial radius is specified, the x-axis will be normalized to units
    of the virial radius again).

    Function returns both the normalized histogram as well as the array
    of its bin edges.

    :param radial_distances: The array of radial distances.
    :param weight: The array of weights to sum per bin. Must have the
        same shape as ``radial_distances``.
    :param bins: The number of radial bins or the array of bin edges.
    :param virial_radius: If the radial distances are given in units of
        the virial radius, specifying the virial radius will return them
        to physical units by multiplying the radial distances with it.
        Subsequently, the quantity in the histogram will also be in
        physical units then.
    :return: The tuple of the shell volume normalized histogram and the
        array of bin edges.
    """
    # check if radial distances need to be unit adjusted
    if virial_radius is not None:
        radial_distances *= virial_radius
    # bin quantity into distance bins
    hist, edges = np.histogram(
        radial_distances,
        bins=bins,
        weights=weight,
    )
    # normalize every column by the shell volume
    shell_volumes = 4 / 3 * np.pi * (edges[1:]**3 - edges[:-1]**3)

    # return x-axis to units of virial radii
    if virial_radius:
        edges = edges / virial_radius

    return hist / shell_volumes, edges
