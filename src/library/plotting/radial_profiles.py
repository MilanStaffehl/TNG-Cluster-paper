"""
Tools for plotting radial profiles.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Sequence

import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.colors import Colormap
    from matplotlib.figure import Figure
    from numpy.typing import NDArray


def plot_radial_temperature_profile(
    histogram2d: NDArray,
    log_msg: str,
    ranges: Sequence[float, float, float, float],
    xlabel: str = r"Distance from halo center [$R_{200c}$]",
    ylabel: str = r"Temperature [$\log K$]",
    title: str | None = None,
    colormap: str = "inferno",
    cbar_label: str = "Count",
) -> tuple[Figure, Axes]:
    """
    Plot the given 2D histogram of temperature vs. halocentric distance.

    Plots a histogram using the data of all halos in the specified
    mass bin. The data must already exist as histogram data, for example
    created by ``np.histogram2d``.

    Function returns figure and axis objects, and does NOT save the plot
    to file or displays it. Returned figure must be saved separately.

    :param histogram2d: Array of 2D histograms of shape (N, R, T) where
        N is the total number of halos, R the number of radial bins of
        the histogram and T the number of temperature bins.
    :param log_msg: The log message suffix to log when execution begins.
        This message will be used to complete the log message "Plotting
        radial temperature profile for >``log_msg``<". Set accordingly.
    :param ranges: The min and max value for every axis in the format
        [xmin, xmax, ymin, ymax].
    :param xlabel: The label for the x-axis; can be a raw string.
    :param ylabel:The label for the y-axis; can be a raw string.
    :param title: Title of the figure. Set to None to leave the figure
        without a title. Can be a raw string to use formulas.
    :param colormap: A matplotlib colormap for the plot. Defaults to
        "inferno".
    :param cbar_label: The label for the colorbar data. Defaults to
        "Count".
    :return: The figure and axes objects with the data drawn onto them.
    """
    logging.info(f"Plotting radial temperature profile for {log_msg}.")
    # create and configure figure and axes
    fig, axes = plt.subplots(figsize=(6, 5))
    fig.set_tight_layout(True)
    if title:
        axes.set_title(title)

    labelsize = 12
    axes.set_xlabel(xlabel, fontsize=labelsize)
    axes.set_ylabel(ylabel, fontsize=labelsize)

    # plot the 2D hist
    hist_config = {
        "cmap": colormap,
        "interpolation": "nearest",
        "origin": "lower",
        "aspect": "auto",
        "extent": ranges,
    }
    profile = axes.imshow(histogram2d, **hist_config)
    # add colorbar
    fig.colorbar(profile, ax=axes, location="right", label=cbar_label)

    axes.legend()
    return fig, axes


def overplot_running_average(
    figure: Figure,
    axes: Axes,
    averages: NDArray,
    ranges: tuple[float, float, float, float],
) -> tuple[Figure, Axes]:
    """
    Overplot a running average onto a 2D histogram.

    Function requires the running averages to be calclated separately and
    passed to this function as an array of exactly as many values as the
    histogram has x-bins.

    :param figure: The figure object onto which to plot. Remains unaltered,
        only included to keep common signature of functions in this module.
    :param axes: The axes object onto which to plot. Altered in place.
    :param averages: The array of averages. Must be a 1D array of shape
        (B, ) where B is the number of x-bins in the histogram.
    :param ranges:  The min and max value for every axis in the format
        [xmin, xmax, ymin, ymax].
    :return: The figure and axes objects with the data drawn onto them.
    """
    # plot the running average
    xbin_width = (ranges[1] - ranges[0]) / len(averages)
    xs = np.arange(ranges[0], ranges[1], xbin_width)
    avg_config = {
        "where": "post",
        "color": "white",
        "linewidth": 1,
        "label": "Running average"
    }
    axes.step(xs, averages, **avg_config)
    return figure, axes


def generate_generic_radial_profile(
    radial_distance: NDArray,
    y_data: NDArray,
    y_label: str,
    weights: NDArray | None = None,
    colorbar_label: str = "Count",
    title: str | None = None,
    xbins: int = 50,
    ybins: int = 50,
    colormap: str | Colormap = "inferno",
    density: bool = False,
) -> tuple[Figure, Axes, NDArray]:
    """
    Return a 2D histogram of the given data (y-data vs radial distance).

    The function plots the datapoints, given by the two arrays for
    radial distance (x-data) and the other quantity (y-data) as a 2D
    histogram radial profile. The function returns the figure and axes
    object as well as the 2D array and the edges describing the histogram.

    :param radial_distance: The shape (N, ) array of radial distances
        for all data points in units of virial radii.
    :param y_data: The shape (N, ) array of the quantity to plot vs the
        radial distance (e.g. temperature).
    :param y_label: The labe for the y-axis of the plot.
    :param weights: The shpe (N, ) array of weights to apply to the
        individual data points. Set to None to use the count of data
        points instead. Defaults to None.
    :param colorbar_label: The label of the colorbar. If weights are used,
        this label should reflect the weighting. Defaults to "Count".
    :param title: Title for the figure. If set to None, the figure will
        have no title. Defaults to None.
    :param xbins: The number of bins to use along the x-axis. Ddefaults
        to 50.
    :param ybins: The number of bins to use along the y-axis. Ddefaults
        to 50.
    :param colormap: Name or colormap class to use for the 2D histogram.
        Defaults to "inferno".
    :param density: Whether to treat the plot as a density plot, i.e.
        whether to normalize the histogram such that all data adds up
        to one. Defaults to False.
    :return: A tuple containing the following entries:
        - The figure object.
        - The axes object.
        - The (bins, bins) shape array of histogram data, givig the
          histogram value of every bin.
        - The x-edges of the histogram.
        - The y-edges of the histogram.
    """
    logging.info(f"Plotting radial profile for {y_label}.")
    # figure set-up
    fig, axes = plt.subplots(figsize=(6, 5))
    axes.set_xlabel(r"Radial distance from center [$R_{200c}$]")
    axes.set_ylabel(y_label)
    if title is not None:
        axes.set_title(title)

    # plot data
    hist_config = {
        "cmap": colormap,
        "bins": (xbins, ybins),
        "density": density,
    }
    if weights is not None:
        hist_config.update({"weights": weights})
    hist = axes.hist2d(radial_distance, y_data, **hist_config)
    # add colorbar
    fig.colorbar(hist[-1], ax=axes, location="right", label=colorbar_label)
    return fig, axes, hist[0], hist[1], hist[3]
