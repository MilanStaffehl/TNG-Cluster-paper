"""
Tools for plotting radial profiles.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from numpy.typing import NDArray


def plot_velocity_distribution(
    axes: Axes,
    histogram: NDArray,
    edges: NDArray,
    xlabel: str | None = None,
    ylabel: str | None = None,
    title: str | None = None,
    color: str | Sequence[float] = "black",
    alpha: float = 1.0,
    label: str | None = None,
    linestyle: str = "solid",
) -> Axes:
    """
    Plot a velocity distribution histogram as a solid line.

    The plot will show the histogram with the given edges as a line plot
    with the specified properties.

    :param axes: The axes object onto which to plot the graph.
    :param histogram: The array containing the values of the histogram
        of velocities. Typically, this would be the cumulative gas
        fraction per velocity bin.
    :param edges: The array of velocity bin edges.
    :param xlabel: x-axis label for the plot, optional. If set to None,
        no axis label will be added.
    :param ylabel: y-axis label for the plot, optional. If set to None,
        no axis label will be added.
    :param title: Title for the figure, optional. If set to None, the
        axes will not receive a title.
    :param color: Color for the line. Can be a named color or an RGB/RGBA
        color as a sequence of floats.
    :param alpha: The transparency of the line between 0 and 1. Defaults
        to 1 (fully opaque).
    :param label: A label for the line to be displayed in alegend. Note
        that this function will **not** add a legend to the axes.
    :param linestyle: The line style for the line.
    :return: The axes object with the line plotted onto it. Purely for
        convenience; axes is altered in place.
    """
    xlims = (edges[0], edges[-1])
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
        "marker": "none",
        "color": color,
        "linestyle": linestyle,
        "alpha": alpha,
    }
    if label:
        plot_config.update({"label": label})
    axes.plot(bin_centers, histogram, **plot_config)
    return axes
