"""
Tools for statistics with temperature and gas cell data.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Sequence

import numpy as np
import numpy.ma as ma

if TYPE_CHECKING:
    from numpy.typing import NDArray


def sort_masses_into_bins(
    masses: NDArray, mass_bins: Sequence[float]
) -> NDArray:
    """
    Sort the given masses into the bins and return mask index array.

    The returned array is of the same length as ``masses`` and contains
    the index of the bin into which the mass of the same array index
    falls, starting from 1.

    :param masses: Array of masses.
    :param mass_bins: List of bin edges, must have same unit as ``masses``.
    :return: Array of mask indices for masses.
    """
    return np.digitize(masses, mass_bins)


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
