"""
Common plotting utilities.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from numpy.typing import NDArray


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


def overplot_running_median(
    x_data: NDArray,
    y_data: NDArray,
    x_err: NDArray,
    y_err: NDArray,
    axes: Axes,
) -> Axes:
    """
    Overplot the given median curve onto the given axes.

    :param x_data: x-values, shape (N, ).
    :param y_data: y-values, shape (N, ).
    :param x_err: Error on the x values, shape (2, N). Dummy argument,
        only used to for signature compliance.
    :param y_err: Error on the y values, shape (2, N).
    :param axes: The axes object onto which to plot the data.
    :return: The updated axes object. Axes is altered in place, so this
        is merely a convenience. The original axes does not need to be
        replaced.
    """
    median_config = {
        "linestyle": "solid",
        "color": "black",
        "linewidth": 1,
        "marker": None,
        "zorder": 10,
    }
    error_config = {
        "linestyle": "dotted",
        "color": "black",
        "linewidth": 1,
        "marker": None,
        "zorder": 10,
    }
    fill_config = {
        "alpha": 0.1,
        "color": "black",
    }
    axes.plot(x_data, y_data, **median_config)
    axes.plot(x_data, y_data - y_err[0], **error_config)
    axes.plot(x_data, y_data + y_err[1], **error_config)
    axes.fill_between(
        x_data, y_data - y_err[0], y_data + y_err[1], **fill_config
    )
