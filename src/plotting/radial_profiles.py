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
    fig, axes = plt.subplots(figsize=(5, 5))
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
        "aspect": "equal",
    }
    axes.imshow(histogram2d, **hist_config)
    # adjust axis tick labels to achieve physical axis ticks

    # overplot the running average
    # _, bins = np.histogram(
    #     np.array([0]), bins=len(averages), range=ranges[:2]
    # )
    # centers = (bins[:-1] + bins[1:]) / 2
    # plot_config = {
    #     "histtype": "step",
    #     "edgecolor": "white",
    # }
    # axes.hist(
    #     centers,
    #     bins=bins,
    #     range=ranges[:2],
    #     weights=averages,
    #     **plot_config
    # )
    return fig, axes
