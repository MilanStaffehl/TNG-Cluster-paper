"""
Tools for plotting radial profiles.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal, Sequence

import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.colors import Colormap
    from matplotlib.figure import Figure
    from numpy.typing import NDArray


def plot_1d_radial_profile(
    axes: Axes,
    histogram: NDArray,
    edges: NDArray,
    xlabel: str | None = r"Distance from halo center [$R_{200c}$]",
    ylabel: str | None = r"Density in radial shell [$M_\odot / kpc^3$]",
    title: str | None = None,
    log: bool = True,
    xlims: tuple[float, float] = (0, 2),
    color: str | Sequence[float] = "black",
    label: str | None = None,
    linestyle: str = "solid",
) -> Axes:
    """
    Plot and return a radial profile given as a histogram.

    The function requires a radial profile to be given as an array of
    values for a set of radial bins specified by ``histogram`` and
    ``edges`` respectively. The radial profile is then plotted as a
    step histogram.

    :param axes: The axes onto which to plot the histogram.
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
    :param color: The color of the histogram in the plot. Defaults to
        black.
    :param label: A label for the plot. Not that the legend will not
        be created by default and must be created on the axes by
        calling ``axes.legend()``.
    :param linestyle: The linestyle for the histogram.
    :return: The axes with the histogrm added to it; returned for
        convenience, axes is altered in place.
    """
    axes.set_xlim(*xlims)
    if xlabel is not None:
        axes.set_xlabel(xlabel)
    if ylabel is not None:
        axes.set_ylabel(ylabel)
    if title:
        axes.set_title(title)

    # plot histogram
    bin_centers = (edges[:-1] + edges[1:]) / 2
    plot_config = {
        "histtype": "step",
        "color": color,
        "log": log,
        "linestyle": linestyle,
    }
    if label:
        plot_config.update({"label": label})
    axes.hist(bin_centers, bins=edges, weights=histogram, **plot_config)
    return axes


def plot_2d_radial_profile(
    fig: Figure,
    axes: Axes,
    histogram2d: NDArray,
    ranges: Sequence[float] | NDArray,
    xlabel: str | None = r"Distance from halo center [$R_{200c}$]",
    ylabel: str | None = r"Temperature [$\log K$]",
    title: str | None = None,
    value_range: Sequence[float] | None = None,
    colormap: str | Colormap = "inferno",
    cbar_label: str = "Count",
    cbar_ticks: NDArray | None = None,
    cbar_limits: Sequence[float | None] | None = None,
    scale: Literal["linear", "log"] = "linear",
    log_msg: str | None = None,
    labelsize: int = 12,
    suppress_colorbar: bool = False,
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

    :param fig: The figure object onto whose axes the histogram will be
        plotted.
    :param axes: The axes unto which to plot the histogram and the
        colorbar.
    :param histogram2d: Array of 2D histograms of shape (Y, R) where
        R is the number of radial bins of the histogram and Y the number
        of y-bins, for example temperature bins.
    :param ranges: The min and max value for every axis in the format
        [xmin, xmax, ymin, ymax].
    :param xlabel: The label for the x-axis; can be a raw string. Can be
        set to None, to not set an axes label.
    :param ylabel: The label for the y-axis; can be a raw string. Can be
        set to None, to not set an axes label.
    :param title: Title of the figure. Set to None to leave the figure
        without a title. Can be a raw string to use formulas.
    :param value_range: The range of values for the histogram. If given,
        all values are limited to this range. Must be of the form
        [vmin, vmax].
    :param colormap: A matplotlib colormap for the plot. Defaults to
        "inferno".
    :param cbar_label: The label for the colorbar data. Defaults to
        "Count".
    :param cbar_ticks: Sequence or array of the tick markers for the
        colorbar. Optional, defaults to None (automatically chosen ticks).
    :param cbar_limits: The lower and upper limit of the colorbars as a
        sequence [lower, upper]. All values above and below will be
        clipped. Setting these values will assume that the colorbar is
        not showing the full range of values, so the ends of the colorbar
        will be turned into open ends (with an arrow-end instead of a
        flat cap). To only limit the colorbar in one direction, set the
        other to None: ``cbar_limits=(-1, None)``. Set to None to show
        the full range of values in the colorbar. If ``log`` is set to
        True, the limits must be given in logarithmic values.
    :param scale: If the histogram data is not already given in log
        scale, this parameter can be set to "log" to plot the log10 of
        the given histogram data.
    :param log_msg: The log message suffix to log when execution begins.
        This message will be used to complete the log message "Plotting
        radial temperature profile for >``log_msg``<". Set accordingly.
        Defaults to None, which means no message is logged.
    :param labelsize: Size of the axes labels in points. Optional,
        defaults to 12 pt.
    :param suppress_colorbar: When set to True, no colorbar is added to
        the figure.
    :return: The figure and axes objects as tuple with the histogram
        added to them; returned for convenience, axes object is altered
        in place.
    """
    if log_msg is not None:
        logging.info(f"Plotting radial profile for {log_msg}.")

    # axes config
    if title:
        axes.set_title(title)
    if xlabel:
        axes.set_xlabel(xlabel, fontsize=labelsize)
    if ylabel:
        axes.set_ylabel(ylabel, fontsize=labelsize)

    # scaling
    if scale == "log":
        histogram2d = np.log10(histogram2d)

    # clipping (clip values and determine open ends of colorbar)
    if cbar_limits is not None:
        if len(cbar_limits) != 2:
            logging.warning(
                "The sequence of limits for the colorbar does not have length "
                "2. Only first two values will be used for limits. This might "
                "cause unexpected behavior!"
            )
        lower_limit, upper_limit = -np.inf, np.inf
        cbar_extend = "neither"
        if cbar_limits[0] is not None:
            lower_limit = cbar_limits[0]
            cbar_extend = "min"
        if cbar_limits[1] is not None:
            upper_limit = cbar_limits[1]
            cbar_extend = "max"
        # clip histogram
        histogram2d = np.clip(histogram2d, lower_limit, upper_limit)
        # determine correct colorbar extent
        if all(cbar_limits):
            cbar_extend = "both"

    # plot the 2D hist
    hist_config = {
        "cmap": colormap,
        "interpolation": "nearest",
        "origin": "lower",
        "aspect": "auto",
        "extent": ranges,
    }
    if value_range is not None:
        hist_config.update({"vmin": value_range[0], "vmax": value_range[1]})
    profile = axes.imshow(histogram2d, **hist_config)

    # add colorbar
    if not suppress_colorbar:
        cbar_config = {
            "location": "right",
            "label": cbar_label,
        }
        if cbar_ticks is not None:
            cbar_config.update({"ticks": cbar_ticks})
        if cbar_limits is not None:
            cbar_config.update({"extend": cbar_extend})
        fig.colorbar(profile, ax=axes, **cbar_config)

    return fig, axes


def overplot_running_average(
    figure: Figure,
    axes: Axes,
    averages: NDArray,
    ranges: tuple[float, float, float, float] | NDArray,
    suppress_label: bool = False,
) -> tuple[Figure, Axes]:
    """
    Overplot a running average onto a 2D histogram.

    Function requires the running averages to be calculated separately and
    passed to this function as an array of exactly as many values as the
    histogram has x-bins.

    :param figure: The figure object onto which to plot. Remains unaltered,
        only included to keep common signature of functions in this module.
    :param axes: The axes object onto which to plot. Altered in place.
    :param averages: The array of averages. Must be a 1D array of shape
        (B, ) where B is the number of x-bins in the histogram.
    :param ranges:  The min and max value for every axis in the format
        [xmin, xmax, ymin, ymax].
    :param suppress_label: When set to True, no label describing the
        line will be added.
    :return: The figure and axes objects with the data drawn onto them.
    """
    # plot the running average
    xbin_width = (ranges[1] - ranges[0]) / len(averages)
    xs = np.arange(ranges[0], ranges[1] + xbin_width, xbin_width)
    avg_config = {
        "where": "post",
        "color": "white",
        "linewidth": 1,
        "label": "Running average"
    }
    # Matplotlib is at it again with its bullshit: last step does not show
    # up without this workaround: duplicate last entry of averages (so that
    # the previous one will show up, and this duplicated last one is
    # discarded instead)
    extended_averages = np.append(averages, averages[-1])
    axes.step(xs, extended_averages, **avg_config)
    if not suppress_label:
        axes.legend()
    return figure, axes


def overplot_temperature_divisions(
    axes: Axes,
    divisions: Sequence[float],
    xmin: float,
    xmax: float,
) -> Axes:
    """
    Overplot two lines for the temperature divisions between regimes.

    Function lots two horizontal dashed lines into the radial temperature
    profile plot to denote the limiting temperatures of the regimes,
    i.e. the division lines between hot/warm and cool/warm gas.

    :param axes: The axes onto which to plot the line.
    :param divisions: A sequence of the two values. Must be in the
        same units and scale as the data on the y-axis.
    :param xmin: Leftmost x value, from where to draw the line towards
        the right.
    :param xmax: Rightmost x value, up to where to draw the line.
    :return: Axes, for convenience. Axes object is altered in place.
    """
    line_config = {
        "colors": "white",
        "linestyles": "dashed",
    }
    axes.hlines(divisions, xmin=xmin, xmax=xmax, **line_config)
    return axes
