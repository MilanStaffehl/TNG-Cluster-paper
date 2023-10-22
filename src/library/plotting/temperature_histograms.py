"""
Plotting tools for histograms of temperature distribution.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma

from library.plotting import util

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from numpy.typing import NDArray


def plot_temperature_distribution(
    mean: NDArray,
    median: NDArray,
    errors: NDArray,
    mass_bin_edges: tuple[float, float],
    temperature_range: tuple[float, float],
    facecolor: str,
    xlabel: str,
    ylabel: str,
    log: bool = True,
) -> tuple[Figure, Axes]:
    """
    Plot the given distribution of halo temperatures and return figure.

    Plots a histogram using the data of all halos in the specified
    mass bin. The data must already exist as histogram data, for example
    created by ``np.histogram``.

    Function returns figure and axis objects, and does NOT save the plot
    to file or displays it. Returned figure must be saved separately.

    :param mean: Array of mean values per bin.
    :param median: Array of median in every bin.
    :param errors: the 16th and 84th percentile in every bin. Must be an
        array of shape (2, N) where N is the length of mean and median,
        i.e. the number of bins. Values must be the length of the error
        bars, NOT their position.
    :param mass_bin_edges: The mass bin edges in units of log10 solar
        masses (so for example for the mass bin from 10^8 to 10^9 solar
        masses, this tuple would be (8, 9)).
    :param temperature_range: Range of temperatures in log10 Kelvin, i.e.
        the left edge of the first and the right edge of the last bin
        (for example for a range from 10^3 K to 10^8 K, this would be the
        tuple (3, 8))
    :param facecolor: The fill color for the area under the histogram.
        Must be a valid matplotlib color.
    :param xlabel: The label for the x-axis. Can be a raw string and
        include TeX formulas.
    :param ylabel: The label for the y-axis. Can be a raw string and
        inlcude TeX formulas.
    :param log: Whether to plot the y-axis in log scale. Defaults to True.
    :return: The figure and axis objects with the data drawn onto them.
    """
    low_bound = np.log10(mass_bin_edges[0])
    upp_bound = np.log10(mass_bin_edges[1])
    logging.info(
        f"Plotting temperature hist for mass bin 10^{low_bound:.0f} - "
        f"10^{upp_bound:.0f}."
    )
    # create and configure figure and axes
    fig, axes = plt.subplots(figsize=(5, 4))
    fig.set_tight_layout(True)
    axes.set_title(
        r"$M_{200c}$: "
        rf"${low_bound:.1f} < \log \ M_\odot < {upp_bound:.1f}$"
    )
    labelsize = 12
    axes.set_xlabel(xlabel, fontsize=labelsize)
    axes.set_ylabel(ylabel, fontsize=labelsize)

    # calculate bin positions
    _, bins = np.histogram(
        np.array([0]), bins=len(mean), range=temperature_range
    )
    centers = (bins[:-1] + bins[1:]) / 2

    # plot data
    plot_config = {
        "histtype": "stepfilled",
        "facecolor": facecolor,
        "edgecolor": "black",
        "log": log,
    }
    # hack: produce exactly one entry for every bin, but weight it
    # by the histogram bar length, to achieve a "fake" bar plot
    axes.hist(
        centers,
        bins=bins,
        range=temperature_range,
        weights=mean,
        **plot_config
    )
    # plot error bars
    error_config = {
        "marker": "x",
        "linestyle": "none",
        "ecolor": "dimgrey",
        "color": "dimgrey",
        "alpha": 0.8,
        "capsize": 2.0,
    }
    axes.errorbar(centers, median, yerr=errors, **error_config)
    return fig, axes


def get_errorbar_lengths(median: NDArray, percentiles: NDArray) -> NDArray:
    """
    Return the error bar lengths as (2, N) shape array.

    The percentles must have shape (2, N) where N is the length of the
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


def overplot_virial_temperatures(
    fig: Figure,
    axes: Axes,
    virial_temperatures: NDArray,
    mass_bin_index: int,
    mass_bin_mask: NDArray,
    color: str = "blue",
    omit_ranges: bool = False,
) -> tuple[Figure, Axes]:
    """
    Overplot the range of virial temperatures onto the given axes.

    Method calculates the virial temperatures for all halos in the
    given mass bin and overplots onto the given axes object the
    range from lowest to highest virial temperature found as well as
    the mean virial temperature in the bin. The region between min
    and max virial temperature is shaded in color, to signify the
    region as the possible ranges of virial temperature.

    Returns the updated figure and axes for convenience, but since the
    object is changed in place, re-assigning it is not necessary.

    :param fig: Figure object, returned as-is.
    :param axes: Axes object upon which to overplot the virial
        temperature range
    :param virial_temperatures: Array of virial temperatures for all
        halos in the simulation.
    :param mass_bin_index: Mass bin index, starting from zero.
    :param mass_bin_mask: A mass bin mask as returned by
        :func:`statistics.sort_masses_into_bins`, that is an array of
        bin numbers into which the halo of the corresponding array index
        falls. Note that mass bin numbers start at 1, not 0.
    :param color: The color to use for the overplot. Must be a matplotlib
        color name. Defaults to 'blue'.
    :param omit_ranges: Whether to plot the ranges or leave them out.
        Defaults to False (i.e. plotting ranges).
    :return: Tuple of figure and axes, updated for overplot.
    """
    # find virial temperatures, only for current bin
    mask = np.where(mass_bin_mask == mass_bin_index + 1, 1, 0)
    virial_temperatures = (
        ma.masked_array(virial_temperatures).compress(mask, axis=0)
    )

    # find min and max as well as the average temperature
    min_temp = np.min(virial_temperatures)
    max_temp = np.max(virial_temperatures)
    mean_temp = np.average(virial_temperatures)

    # overplot these into the plot
    logging.debug("Overplotting virial temperatures.")
    plot_config = {
        "color": color,
        "linewidth": 1.0,
        "alpha": 0.6,
    }
    axes.axvline(np.log10(mean_temp), linestyle="dashed", **plot_config)

    # optionally include the min-max region
    if not omit_ranges:
        axes.axvline(np.log10(min_temp), linestyle="solid", **plot_config)
        axes.axvline(np.log10(max_temp), linestyle="solid", **plot_config)
        # shade region
        xs = np.arange(np.log10(min_temp), np.log10(max_temp), 0.01)
        fill_config = {
            "transform": axes.get_xaxis_transform(),
            "alpha": 0.1,
            "color": color,
        }
        axes.fill_between(xs, 0, 1, **fill_config)
    return fig, axes


def plot_temperature_distribution_gallery(
    n_plots: int,
    halo_ids: NDArray,
    histograms: NDArray,
    virial_temperatures: NDArray,
    temperature_range: tuple[float, float],
    mass_bin_edges: tuple[float, float],
    xlabel: str = "Gas temperature [log K]",
) -> tuple[Figure, Axes]:
    """
    Plot a gallery of temperature distributions for the given mass bin.

    The function creates a gallery of temperature distribution plots
    for all selected halos in the given mass bin and saves it to file.

    :param n_plots: The number of halos to plot in the gallery.
    :param halo_ids: An 1D array of halo IDs for halos to plot in the
        gallery. Must have at least length ``n_plots``.
    :param histograms: An array of histogram data for all halos. Must
        have shape (N, T) where N is the length of ``halo_ids`` and T is
        the number of temperature bins in the histograms. Halos without
        histogram data are expected to have all entries of their histogram
        be ``np.nan``. Histogram bins are expected to be in units log K.
    :param virial_temperatures: An array of virial temperatures for every
        halo. Must have the same length as ``halo_ids`` and the temperatures
        must be given in units of Kelvin.
    :param temperature_range: The range of temperatures in the histogram
        in log10, i.e. a range of 10^3 to 10^8 Kelvin corresponds to a
        tuple (3.0, 8.0).
    :param mass_bin_edges: Edges of the current mass bin in log10 M_sol.
        Only used for logging.
    :param xlabel: The axis label for the x-axis. Use when the temperature
        in the histograms has been normalized.
    :return: Tuple of the figure and axes object with the plots applied
        to them. Figure is not saved!
    """
    low_bound = np.log10(mass_bin_edges[0])
    upp_bound = np.log10(mass_bin_edges[1])
    logging.info(
        f"Plotting temperature hist for mass bin 10^{low_bound:.0f} - "
        f"10^{upp_bound:.0f}."
    )
    # figure set-up
    fig, axes = plt.subplots(figsize=(8, 10), ncols=2, nrows=int(n_plots / 2))
    fig.set_tight_layout(True)

    axes = axes.flatten()  # allows simple iteration
    drawn_plots = 0
    for i, halo_id in enumerate(halo_ids):
        if np.any(np.isnan(histograms[i])):
            logging.debug(f"Skipped halo {halo_id} due to being empty.")
            continue  # halo had no gas
        if drawn_plots == n_plots:
            break  # all subplots have been populated.
        # axes config
        axes[drawn_plots].set_xlabel(xlabel)
        axes[drawn_plots].set_ylabel("Gas mass fraction")
        axes[drawn_plots].set_ylim(1e-6, 1)
        axes[drawn_plots].set_title(f"Halo ID: {halo_id}")
        # calculate bin positions
        _, bins = np.histogram(
            np.array([0]), bins=len(histograms[0]), range=temperature_range
        )
        centers = (bins[:-1] + bins[1:]) / 2

        # plot data
        plot_config = {
            "histtype": "stepfilled",
            "facecolor": "lightblue",
            "edgecolor": "black",
            "log": True,
        }
        # hack: produce exactly one entry for every bin, but weight it
        # by the histogram bar length, to achieve a "fake" bar plot
        axes[drawn_plots].hist(
            centers,
            bins=bins,
            range=temperature_range,
            weights=histograms[i],
            **plot_config
        )
        # overplot virial temperature
        line_config = {
            "color": "blue",
            "linewidth": 1.0,
            "alpha": 0.6,
            "linestyle": "dashed",
        }
        axes[drawn_plots].axvline(
            np.log10(virial_temperatures[i]), **line_config
        )
        drawn_plots += 1  # increment counter

    return fig, axes


def plot_temperature_distributions_in_one(
    means: NDArray,
    temperature_range: tuple[float, float],
    mass_bin_edges: NDArray,
    xlabel: str,
    ylabel: str,
    colormap: str = "cividis",
    log: bool = True,
) -> tuple[Figure, Axes]:
    """
    Return figure and axis of a combined plot of temperature distributions.

    Function plots the given temperature distribution histograms and
    returns the figure and axes objects. The plots are colored according
    to the given colormap.

    :param means: An array of shape (M, T) containing the histogram mean
        values. M is the number of mass bins, T the number of temperature
        bins.
    :param temperature_range: The range of temperatures of the histograms
        in units of log10(Kelvin).
    :param mass_bin_edges: The edges of the mass bins in units of solar
        masses.
    :param xlabel: Label for the x axis.
    :param ylabel: Label for the y axis.
    :param colormap: A valid string name for a matplotlib colormap. The
        colors of the different mass bins will be sampled from this map
        by evenly spacing sampling points. Defaults to "cividis".
    :param log: Whether to plot the y-axis in log scale. Defaults to True.
    :return: A tuple of the matplotlib figure and axes objects, with the
        plot drawn onto them.
    """
    logging.info("Plotting all temperature distributions combined.")
    fig, axes = plt.subplots(figsize=(6, 5))
    fig.set_tight_layout(True)

    axes.set_title("Temperature distributions")
    labelsize = 12
    axes.set_xlabel(xlabel, fontsize=labelsize)
    axes.set_ylabel(ylabel, fontsize=labelsize)

    # calculate bin positions
    _, bins = np.histogram(
        np.array([0]), bins=len(means[0]), range=temperature_range
    )
    centers = (bins[:-1] + bins[1:]) / 2

    for idx, hist in enumerate(means):
        # get the color
        color = util.sample_cmap(colormap, len(means), idx)

        # plot data
        plot_config = {
            "histtype": "step",
            "label":
                (
                    rf"${np.log10(mass_bin_edges[idx]):.0f} < \log \ M_\odot "
                    rf"< {np.log10(mass_bin_edges[idx + 1]):.0f}$"
                ),
            "edgecolor": color,
            "log": log,
        }
        # hack: produce exactly one entry for every bin, but weight it
        # by the histogram bar length, to achieve a "fake" bar plot
        axes.hist(
            centers,
            bins=bins,
            range=temperature_range,
            weights=hist,
            **plot_config
        )
    axes.legend(
        loc="lower center", bbox_to_anchor=(0.5, 1.1), ncol=len(means) // 2
    )
    return fig, axes
