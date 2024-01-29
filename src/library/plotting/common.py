"""
Common plotting utilities.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Sequence

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from numpy.typing import NDArray

temperature_colors_rgb = {
    "cool": (30, 144, 255),  # dodgerblue
    "warm": (128, 0, 128),  # purple
    "hot": (220, 20, 60),  # crimson
}

temperature_colors_named = {
    "cool": "dodgerblue",
    "warm": "purple",
    "hot": "crimson",
}


def overplot_datapoints(
    x_data: NDArray,
    y_data: NDArray,
    x_err: NDArray,
    y_err: NDArray,
    axes: Axes
) -> Axes:
    """
    Overplot the given data onto the given axes.

    :param x_data: x-values, shape (N, ).
    :param y_data: y-values, shape (N, ).
    :param x_err: Error on the x values, shape (2, N).
    :param y_err: Error on the y values, shape (2, N).
    :param axes: The axes object onto which to plot the data.
    :return: The updated axes object. Axes is altered in place, so this
        is merely a convenience. The original axes does not need to be
        replaced.
    """
    plot_config = {
        "marker": "o",
        "markersize": 4,
        "linestyle": "none",
        "capsize": 2,
        "color": "black",
        "zorder": 10,
    }
    axes.errorbar(
        x_data,
        y_data,
        xerr=x_err,
        yerr=y_err,
        **plot_config,
    )
    return axes


def plot_curve_with_error_region(
    x_data: NDArray,
    y_data: NDArray,
    x_err: Any,
    y_err: NDArray | None,
    axes: Axes,
    linestyle: str = "solid",
    color: str | Sequence[float] = "black",
    label: str | None = None,
    suppress_error_line: bool = False,
    suppress_error_region: bool = False,
) -> Axes:
    """
    Overplot the given values with a shaded error region onto the axes.

    :param x_data: x-values, shape (N, ).
    :param y_data: y-values, shape (N, ).
    :param x_err: Error on the x values, shape (2, N). Dummy argument,
        only used to for signature compliance.
    :param y_err: Error on the y values, as an array [lower, upper] of
        shape (2, N).
    :param axes: The axes object onto which to plot the data.
    :param linestyle: The line style for the central line. Should not
        be 'dotted' as this is used for the errors.
    :param color: The color for the lines and shaded region.
    :param label: A label for the curve. Note that this function will
        not create a legend on the plot.
    :param suppress_error_line: Set to True to not draw lines for the
        error.
    :param suppress_error_region: Set to True to not draw the shaded
        region between the errors.
    :return: The updated axes object. Axes is altered in place, so this
        is merely a convenience. The original axes does not need to be
        replaced.
    """
    curve_config = {
        "linestyle": linestyle,
        "color": color,
        "linewidth": 1,
        "marker": None,
        "zorder": 10,
    }
    error_config = {
        "linestyle": "dotted",
        "color": color,
        "linewidth": 1,
        "marker": None,
        "zorder": 10,
    }
    fill_config = {
        "alpha": 0.1,
        "color": color,
    }
    if label is not None:
        curve_config.update({"label": label})
    axes.plot(x_data, y_data, **curve_config)
    if y_err is not None and not suppress_error_line:
        axes.plot(x_data, y_data - y_err[0], **error_config)
        axes.plot(x_data, y_data + y_err[1], **error_config)
    if y_err is not None and not suppress_error_region:
        axes.fill_between(
            x_data, y_data - y_err[0], y_data + y_err[1], **fill_config
        )
    return axes
