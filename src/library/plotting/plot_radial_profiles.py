"""
Tools for plotting radial profiles.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal, Sequence

import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.colors import Colormap
    from matplotlib.figure import Figure
    from numpy.typing import NDArray


def plot_1d_radial_profile(
    histogram: NDArray,
    edges: NDArray,
    xlabel: str = r"Distance from halo center [$R_{200c}$]",
    ylabel: str = r"Average density in radial shell [$M_\odot / kpc^3$]",
    title: str | None = None,
    log: bool = True,
    xlims: tuple[float, float] = (0, 2),
) -> tuple[Figure, Axes]:
    """
    Plot and return a radial profile given as a histogram.

    The function requires a radial profile to be given as an array of
    values for a set of radial bins specified by ``histogram`` and
    ``edges`` respectively. The radial profile is then plotted as a
    step histogram.

    :param histogram: The array of the values for every radial bin.
        Shape (N, ).
    :param edges: The array of the radial bin edges, must be of shape
        (N + 1, ).
    :param xlabel: The label for the x-axis.
    :param ylabel: The label for the y-axis.
    :param title: Optional title for the figure.
    :param log: Whether to plot the profile in log scale. Defaults to
        True.
    :param xlims: The limits of the x-axis values. This is useful to
        prevent matplotlib from adding a margin left and right of the
        limiting values.
    :return: A tuple of the figure and axes object with the plot on them.
    """
    fig, axes = plt.subplots(figsize=(5, 4))
    fig.set_tight_layout(True)

    axes.set_xlabel(xlabel)
    axes.set_xlim(xlims)
    axes.set_ylabel(ylabel)
    if title:
        axes.set_title(title)

    # plot histogram
    bin_centers = (edges[:-1] + edges[1:]) / 2
    plot_config = {
        "histtype": "step",
        "color": "black",
        "log": log,
    }
    axes.hist(bin_centers, bins=edges, weights=histogram, **plot_config)
    return fig, axes


def plot_2d_radial_profile(
    histogram2d: NDArray,
    ranges: Sequence[float, float, float, float],
    xlabel: str = r"Distance from halo center [$R_{200c}$]",
    ylabel: str = r"Temperature [$\log K$]",
    title: str | None = None,
    colormap: str = "inferno",
    cbar_label: str | Colormap = "Count",
    cbar_ticks: NDArray | None = None,
    scale: Literal["linear", "log"] = "linear",
    log_msg: str | None = None,
) -> tuple[Figure, Axes]:
    """
    Plot the given 2D histogram of temperature vs. halocentric distance.

    Plots a histogram using the data of all halos in the specified
    mass bin. The data must already exist as histogram data, for example
    created by ``np.histogram2d``.

    Function returns figure and axis objects, and does NOT save the plot
    to file or displays it. Returned figure must be saved separately.

    .. attention:: The expected histogram array is ordered (y, x), as
        opposed to the order of the return value of many histogram-
        generating functions such as ``numpy.histogram2d``, which will
        give an array of shape (x, y).

    :param histogram2d: Array of 2D histograms of shape (Y, R) where
        R is the number of radial bins of the histogram and Y the number
        of y-bins, for example temperature bins.
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
    :param cbar_ticks: Sequence or array of the tick markers for the
        colorbar. Optional, defaults to None (automatically chosen ticks).
    :param scale: If the histogram data is not already given in log
        scale, this parameter can be set to "log" to plot the log10 of
        the given histogram data.
    :param log_msg: The log message suffix to log when execution begins.
        This message will be used to complete the log message "Plotting
        radial temperature profile for >``log_msg``<". Set accordingly.
        Defaults to None, which means no message is logged.
    :return: The figure and axes objects with the data drawn onto them.
    """
    if log_msg is not None:
        logging.info(f"Plotting radial temperature profile for {log_msg}.")
    # create and configure figure and axes
    fig, axes = plt.subplots(figsize=(5, 4))
    fig.set_tight_layout(True)
    if title:
        axes.set_title(title)

    labelsize = 12
    axes.set_xlabel(xlabel, fontsize=labelsize)
    axes.set_ylabel(ylabel, fontsize=labelsize)

    # scaling
    if scale == "log":
        histogram2d = np.log10(histogram2d)

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
    cbar_config = {
        "location": "right",
        "label": cbar_label,
    }
    if cbar_ticks is not None:
        cbar_config.update({"ticks": cbar_ticks})
    fig.colorbar(profile, ax=axes, **cbar_config)

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
    axes.legend()
    return figure, axes
