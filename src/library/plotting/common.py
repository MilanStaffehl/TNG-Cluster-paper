"""
Common plotting utilities.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np

from library import constants

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.colors import Colormap, Normalize
    from matplotlib.figure import Figure
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
    zorder: int = 10,
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
    :param zorder: The zorder of the points and error bars. Defaults to
        10, which is higher than the default 0.
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
        "zorder": zorder,
    }
    error_config = {
        "linestyle": "dotted",
        "color": color,
        "linewidth": 1,
        "marker": None,
        "zorder": zorder,
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


def plot_scatterplot(
    figure: Figure,
    axes: Axes,
    x_values: NDArray,
    y_values: NDArray,
    color_quantity: NDArray | None = None,
    color: str = "black",
    cmap: str | Colormap = "viridis",
    marker_style: str = "o",
    marker_size: int | NDArray = 8,
    legend_label: str | None = None,
    alpha: float = 1.0,
    norm: Normalize | None = None,
    cbar_label: str = "Color",
    cbar_range: Sequence[float, float] | None = None,
    cbar_caps: str = "neither",
    suppress_colorbar: bool = False
) -> tuple[Figure, Axes]:
    """
    Plot a scatterplot of the given values, optionally colored by a third.

    The function will also automatically add a colorbar when a colored
    quantity is used. The colorbar will be placed into the existing
    axes passed as ``axes``, so space will be stolen from this axes to
    accomodate the colorbar.

    Note that while a legend label can be set, the function will **not**
    add a legend to the figure/axes objects.

    :param figure: The figure onto which to add the color bar.
    :param axes: The axes onto which to plot the scatter plot.
    :param x_values: The array of x-values.
    :param y_values: The array of y-values.
    :param color_quantity: The optional quantity by which to color the
        scatter points. Set to None to use uniform color.
    :param color: The color to use when coloring points uniformly.
        Defaults to black. Has no effect when ``color_quantity`` is set.
    :param cmap: The colormap to use to color the points when a
        ``color_quantity`` is used. Defaults to 'viridis'. Has no effect
        when ``color_quantity`` is None.
    :param marker_style: The style for the markers, e.g. "o" or "x".
        Defaults to "o" (circular markers).
    :param marker_size: The size of the markers. Can also be an array of
        the same length as the x- and y-values to give points different
        sizes. Defaults to 4.
    :param legend_label: A label for the plot legend. Note that no legend
        will be created, the label is merely added to the paths collection.
    :param alpha: The opacity of the data points. Defaults to 1.
    :param norm: A norm object to use to define range and norm of the
        colors in the scatter plot. When given, the ``cbar_range``
        parameter has no effect.
    :param cbar_label: The label of the colorbar.
    :param cbar_range: The minimum and maximum value between which the
        colored values will lie. Has no effect when ``colored_quantity``
        is None. To automatically determine range, set to None.
    :param cbar_caps: The cap parameters for the lower and upper end
        of the colorbar. Must be one of the following: 'neither', 'both',
        'min', 'max'.
    :param suppress_colorbar: Set to True to not add a colorbar.
    :return: The tuple of the figure and the axes object, with the plot
        drawn onto them. Returned for convenience, the objects are
        altered in place.
    """
    if norm is not None and cbar_range is not None:
        logging.warning(
            "Cannot use both a norm object and a range. Will ignore value "
            "range and use only norm."
        )
        cbar_range = None

    scatter_config = {
        "c": color,
        "s": marker_size,
        "marker": marker_style,
        "alpha": alpha,
        "zorder": 10,
    }
    # overwrite color if a color quantity is given
    if color_quantity is not None:
        scatter_config.update({"c": color_quantity, "cmap": cmap})
        if norm is not None:
            scatter_config.update({"norm": norm})
        # limit ranges if required
        if cbar_range:
            if len(cbar_range) > 2:
                logging.warning(
                    f"Expected a colorbar and value range of length 2, got "
                    f"length {len(cbar_range)} instead. The first two entries "
                    f"will be used as vmin and vmax respectively. This may "
                    f"cause issues."
                )
            scatter_config.update(
                {
                    "vmin": cbar_range[0], "vmax": cbar_range[1]
                }
            )
    # add a legend label if given
    if legend_label:
        scatter_config.update({"label": legend_label})

    # plot the data
    sc = axes.scatter(x_values, y_values, **scatter_config)

    # add a colorbar
    if color_quantity is not None and not suppress_colorbar:
        figure.colorbar(sc, ax=axes, label=cbar_label, extend=cbar_caps)

    return figure, axes


def make_redshift_plot(axes: Axes) -> NDArray:
    """
    Prepare the given axes to be a plot over redshift.

    Function adds two x-axes to the given ``Axes`` object: one for
    redshift at the bottom of the figure and one for lookback time at
    the top. It returns a list of redshift values to use for this type
    of plot. This array has shape (100, ) and assumes that one data
    point for every snapshot will be made.

    :param axes: The axis object whose x-axis to transform into a
        redshift axis.
    :return: An array of 100 x-values equivalent to the redshifts of
        the 100 snapshots of the TNG simulations to use in plotting.
    """
    axes.set_xlabel("Redshift")
    xticks = np.array([0, 0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10])
    axes.set_xticks(np.log10(xticks + 0.01), labels=xticks)
    return np.log10(np.array(constants.REDSHIFTS) + 0.01)
