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
    averages: NDArray,
    mass_bin_edges: tuple[float, float],
    ranges: Sequence[float, float, float, float],
    xlabel: str = r"Distance from halo center $r / R_{200c}$",
    ylabel: str = r"Temperature [$\log K$]",
    colormap: str = "inferno",
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
    :param averages: Array of running average along x-axis. Must have
        shape (R,).
    :param mass_bin_edges: The two edges of the current bin to plot.
    :param ranges: The min and max value for every axis in the format
        [xmin, xmax, ymin, ymax].
    :param xlabel: The label for the x-axis; can be a raw string.
    :param ylabel:The label for the y-axis; can be a raw string.
    :param colormap: A matplotlib colormap for the plot. Defaults to
        "inferno"
    :return: The figure and axes objects with the data drawn onto them.
    """
    low_bound = np.log10(mass_bin_edges[0])
    upp_bound = np.log10(mass_bin_edges[1])
    logging.info(
        f"Plotting radial temperature profile for mass bin 10^{low_bound:.0f} "
        f"- 10^{upp_bound:.0f}."
    )
    # create and configure figure and axes
    fig, axes = plt.subplots(figsize=(6, 5))
    fig.set_tight_layout(True)
    axes.set_title(
        r"$M_{200c}$: "
        rf"${low_bound:.1f} < \log \ M_\odot < {upp_bound:.1f}$"
    )
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
    fig.colorbar(profile, ax=axes, location="right", label="Gas fraction")

    # plot the running average
    xbin_width = (ranges[1] - ranges[0]) / histogram2d.shape[1]
    xs = np.arange(ranges[0], ranges[1], xbin_width)
    avg_config = {
        "where": "post",
        "color": "white",
        "linewidth": 1,
        "label": "Running average"
    }
    axes.step(xs, averages, **avg_config)
    axes.legend()

    return fig, axes


def generate_generic_radial_profile(
    radial_distance: NDArray,
    y_data: NDArray,
    y_label: str,
    weights: NDArray | None = None,
    colorbar_label: str = "Count",
    title: str | None = None,
    bins: int = 50,
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
    :param bins: The number of bins to use along each axis. Ddefaults
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
        "bins": bins,
        "density": density,
    }
    if weights:
        hist_config.update({"weights": weights})
    hist = axes.hist2d(radial_distance, y_data, **hist_config)
    # add colorbar
    fig.colorbar(hist[-1], ax=axes, location="right", label=colorbar_label)
    return fig, axes, hist[0], hist[1], hist[3]
