"""
Common plotting utilities.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Literal, Sequence, TypeAlias

import astropy.cosmology
import astropy.units
import cmasher  # noqa: F401
import matplotlib.cm
import matplotlib.colors
import matplotlib.patches
import matplotlib.pyplot as plt
import numpy as np

from library import constants
from library.plotting import colormaps

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from numpy.typing import NDArray

RGBColor: TypeAlias = tuple[float, float, float]
RGBAColor: TypeAlias = tuple[float, float, float, float]

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
    linewidth: float | None = None,
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
    :param linewidth: The stroke width for the data line. Defaults to
        None which then uses the matplotlib default line width.
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
    if linewidth is not None:
        curve_config.update({"linewidth": linewidth})
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
    cmap: str | matplotlib.colors.Colormap = "viridis",
    marker_style: str = "o",
    marker_size: int | NDArray = 8,
    legend_label: str | None = None,
    alpha: float = 1.0,
    norm: matplotlib.colors.Normalize | None = None,
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
        To create a legend, use ``axes.legend()`` after calling this
        function.
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


def plot_4d_histogram(
    axes: Axes,
    hues: NDArray,
    values: NDArray,
    xedges: NDArray,
    yedges: NDArray,
    hmap: str | matplotlib.colors.Colormap | None = None,
    hue_scale: Literal["linear", "log"] = "linear",
    value_scale: Literal["linear", "log"] = "linear",
    color_range: Sequence[float] | None = None,
    hue_label: str | None = None,
    value_label: str | None = None,
    cbar_axes: Axes | None = None,
    cbar_anchor: tuple[float, float] = (0.15, 0.11),
    cbar_width: float = 0.2,
    cbar_linecolor: str | RGBColor | RGBAColor = "black",
    cbar_labelsize: str | float = "normal",
    suppress_colorbar: bool = False,
    use_saturation: bool = False,
) -> tuple[Axes, Axes]:
    """
    Create a 2D histogram with a two-dimensional colormap.

    The function takes two 2D histograms ``hues`` and ``values`` and
    combines them by interpreting the value of these histograms as the
    hue and value of colors on the chosen colorbar respectively. The two
    histograms must match in bin number, bin width and bin edges.

    The color for each bin is then chosen such that the _hue_ of the bin
    represents the value of the ``hues`` histogram and the _value_ (i.e.
    the "darkness" of the color) the value of the ``values`` histogram.
    The third channel of the HSV color space, the saturation is fixed at
    100% for every bin to ensure visual clarity.

    .. note:: This also means that colormaps will not look as one might
        expect: colormaps that make use of dark colors, including
        especially black, will not represent these dark colors, as value
        is controlled by the data. Similarly, many standard diverging
        colormaps will not lead to the desired result, since they are
        often not continuous in hue, but have two discrete hue values
        left and right of the center point.

    The function additionally adds a two-dimensional colorbar to the
    plot, onto a given secondary axes. If no such axes is given, an
    inset axes will be added to the given ``axes`` and positioned
    according to the specifications.

    NaN values in either hue or value are marked with white color.

    The range of hues and values can be limited using the ``color_range``
    parameter. Note that in this scenario, values outside the range are
    clipped (i.e. mapped to the nearest edge) and not marked with a
    special color. The colorbar will additionally not indicate possible
    extent of values beyond its edges (as it would in a 1D colorbar with
    open-ended caps).

    :param axes: The axes object onto which to draw the histogram.
    :param hues: The 2D histogram of values that will determine the
        color hue of the histogram. Must be an array of shape (X, Y),
        where X denotes the columns and Y the rows of the histogram.
    :param values: The 2D histogram of values that will determine the
        color value of the histogram. Must be an array of shape (X, Y),
        where X denotes the columns and Y the rows of the histogram.
    :param xedges: A 1D array of the bin edges of the x-values of the
        ``hue`` and ``value`` histograms. Must have shape (X + 1, ).
        Required for labeling the axes with the correct tick labels.
    :param yedges: A 1D array of the bin edges of the y-values of the
        ``hue`` and ``value`` histograms. Must have shape (Y + 1, ).
        Required for labeling the axes with the correct tick labels.
    :param hmap: The colormap to use for the hues. Note that colormaps
        with varying saturation and values will not look as expected,
        as this function extracts _exclusively_ the hue of the colormap.
        Can either be a named colormap or a custom colormap. It is
        recommended to use custom colormaps at constant value and
        saturation to ensure a visually pleasing result.
    :param hue_scale: The scale for the hues. Can either be ``linear``
        for linear scaling or ``log`` for logarithmic scaling. When set
        to logarithmic scaling, the logarithm will be taken _before_ the
        hues are clipped to any hue range provided, which can lead to
        zero-hues turning into NaN. To avoid this, clip the hues
        manually before calling this function.
    :param value_scale: The scale for the values. Can either be ``linear``
        for linear scaling or ``log`` for logarithmic scaling. When set
        to logarithmic scaling, the logarithm will be taken _before_ the
        values are clipped to any value range provided, which can lead to
        zero-values turning into NaN. To avoid this, clip the values
        manually before calling this function.
    :param color_range: A sequence of four floats giving the range of
        hue and values in the form ``[hmin, hmax, vmin, vmax]``.
        Values outside the corresponding range are mapped to the end
        of the range. If the scale of hue or value is set to log, the
        corresponding range must be given in log scale. Each value may
        be set to None individually, in which case the corresponding
        value is automatically determined as the min/max of the hue/value.
        If the parameter is set to None, the entire range is determined
        automatically. Defaults to None, which means automatic range
        determination.
    :param hue_label: The label for the colorbar hue axis, i.e. a
        description of the quantity in the ``hues`` histogram. Optional,
        defaults to None, which results in no axis label.
    :param value_label: The label for the colorbar value axis, i.e. a
        description of the quantity in the ``values`` histogram. Optional,
        defaults to None, which results in no axis label.
    :param cbar_axes: The axes onto which to draw the colorbar. This can
        be an axis from a completely separate figure, if the colorbar
        shall be separate from the plot itself. If set to None, an inset
        axes will be added to ``axes`` and the colorbar will be plotted
        onto this inset axes. Optional, defaults to None.
    :param cbar_anchor: When ``cbar_axes`` is set to None, this determines
        the position of the bottom left corner of the inset axes in
        relative axes coordinates. Must be a tuple of two floats between
        0 and 1. Defaults to (0.15, 0.11).
    :param cbar_width: When ``cbar_axes`` is set to None, this determines
        the width of the inset axes in relative axes coordinates. Must
        be a float between 0 and 1. Defaults to 0.2, i.e. 20% width.
    :param cbar_linecolor: The color to use in the axes ticks, spines,
        and label text. Can be useful if the colorbar is placed over
        dark regions of the figure. Defaults to black.
    :param cbar_labelsize: The size of the tick labels and axes labels
        of the colorbar. Can either be a float or a fontsize directive
        from matplotlib (e.g. 'x-small'). Defaults to matplotlib's
        'normal' font size.
    :param suppress_colorbar: When set to True, the creation of a colorbar
        is skipped. Useful for creating many plots that share the same
        colorbar. Defaults to False.
    :param use_saturation: When set to True, the function will vary the
        saturation of the color, instead of its value. In essence, this
        means that the color will go from fully saturated color to
        white instead of black. In this case, NaN values are marked as
        black. Defaults to False.
    :return: The axes object with the histogram drawn onto it, and the
        axes object onto which the colorbar was drawn. The colorbar
        axes will be the newly created inset axes if no colorbar axes
        was provided. Both axes are altered in place; manually updating
        them is not necessary.
    """
    # set defaults
    if hmap is None:
        # map from red to blue at constant lightness and saturation
        hmap = colormaps.custom_cmap((255, 0, 0), (0, 0, 255))
    elif isinstance(hmap, str):
        hmap = plt.get_cmap(hmap)

    if use_saturation:
        fixed_channel = 2  # value is fixed
        value_channel = 1  # saturation is variable
        fault_color = (0, 0, 0)
    else:
        fixed_channel = 1  # saturation is fixed
        value_channel = 2  # value is variable
        fault_color = (1, 1, 1)

    # set scale if desired
    if hue_scale == "log":
        with np.errstate(invalid="ignore"):
            hues = np.log10(hues)
    if value_scale == "log":
        with np.errstate(invalid="ignore"):
            values = np.log10(values)

    # determine value and hue range, clip hues and values accordingly
    hvrange = [
        np.nanmin(hues), np.nanmax(hues), np.nanmin(values), np.nanmax(values)
    ]
    if color_range is not None:
        for i in range(4):
            if color_range[i] is not None:
                hvrange[i] = color_range[i]
    hues = np.clip(hues, hvrange[0], hvrange[1])
    values = np.clip(values, hvrange[2], hvrange[3])

    # extract color from colormap (in HSV space)
    norm = matplotlib.colors.Normalize(hvrange[0], hvrange[1])
    color = matplotlib.colors.rgb_to_hsv(hmap(norm(hues))[:, :, :3])

    # normalize colormap to constant saturation/value
    color[:, :, fixed_channel] = 1  # fix value at maximum

    # turn values into lightness
    value_range = hvrange[3] - hvrange[2]
    values_normed = (values - hvrange[2]) / value_range
    color[:, :, value_channel] = values_normed
    # return to rgb
    color_rgb = matplotlib.colors.hsv_to_rgb(color)

    # treat NaN entries
    where_nan = np.logical_or(np.isnan(hues), np.isnan(values))
    color_rgb[where_nan] = fault_color

    # create histogram
    axes.pcolormesh(xedges, yedges, color_rgb, shading="flat", rasterized=True)

    if not suppress_colorbar:
        # determine axes to place colorbar on and configure it
        if cbar_axes is None:
            bounds = (cbar_anchor[0], cbar_anchor[1], cbar_width, cbar_width)
            cbar_axes = axes.inset_axes(bounds)
        cbar_axes.tick_params(colors=cbar_linecolor, labelsize=cbar_labelsize)
        plt.setp(cbar_axes.spines.values(), color=cbar_linecolor)
        # labels
        cbar_axes.set_xlabel(hue_label, c=cbar_linecolor, size=cbar_labelsize)
        cbar_axes.set_ylabel(
            value_label, c=cbar_linecolor, size=cbar_labelsize
        )

        # prepare a colorbar; start by creating dummy values
        cbar_values = np.linspace(0, 1, num=100)
        cbar_values = np.broadcast_to(cbar_values[:, None], (100, 100))

        # turn values into a colorbar
        c = cbar_values.transpose()
        cbar_color = matplotlib.colors.rgb_to_hsv(hmap(c)[:, :, :3])
        cbar_color[:, :, value_channel] = cbar_values
        cbar_color[:, :, fixed_channel] = 1
        cbar_rgb = matplotlib.colors.hsv_to_rgb(cbar_color)

        # colorbar coordinate grid for correct tick labels
        hedges = np.linspace(hvrange[0], hvrange[1], num=100)
        vedges = np.linspace(hvrange[2], hvrange[3], num=100)

        # place colorbar
        cbar_axes.pcolormesh(
            hedges, vedges, cbar_rgb, shading="gouraud", rasterized=True
        )

    return axes, cbar_axes


def make_redshift_plot(
    axes: Axes,
    start: int = 0,
    stop: int = 99,
    zero_sentinel: float = 1e-3,
) -> NDArray:
    """
    Prepare the given axes to be a plot over redshift.

    Function adds two x-axes to the given ``Axes`` object: one for
    redshift at the bottom of the figure and one for lookback time at
    the top. It returns a list of redshift values to use for this type
    of plot. This array has shape (100, ) and assumes that one data
    point for every snapshot will be made.

    Optionally, the range of snapshots (and therefore redshifts) can be
    set with the ``start`` and ``stop`` snapshots. The axes will then
    only go from the redshift associated with ``start`` to that associated
    with ``stop`` and the returned x-values will have a length and values
    to match.

    :param axes: The axis object whose x-axis to transform into a
        redshift axis.
    :param start: The snapshot at which to start. The resulting x-values
        will be limited to start from this snapshot. Defaults to snap 0.
    :param stop: The snapshot at which to stop. The resulting x-values
        will be limited to end at this snapshot. The result will include
        the ``stop`` snapshot. Defaults to snap 99.
    :param zero_sentinel: The sentinel value to use instead of 0 for
        redshift zero. This is needed to show redshift zero at a finite
        position in logarithmic scaling. Must be set lower than the
        smallest redshift value above zero, which for TNG is roughly
        0.0095. Defaults to ``1e-3``.
    :return: An array of 100 x-values equivalent to the redshifts of
        the 100 snapshots of the TNG simulations to use in plotting.
    """
    planck15 = astropy.cosmology.Planck15
    redshifts = np.array(constants.REDSHIFTS)
    redshifts[-1] = zero_sentinel  # avoid log-problems with zero

    # axes set-up
    axes.set_xlabel("Redshift")
    axes.set_xscale("log")
    xticks = np.array([zero_sentinel, 0.01, 0.1, 0.5, 1, 2, 5, 10])
    xtick_labels = [f"{x:g}" for x in xticks]
    xtick_labels[0] = "0"  # set label for zero to actually say zero
    axes.set_xticks(xticks, labels=xtick_labels)
    axes.set_xlim(redshifts[stop], redshifts[start])

    # secondary axis setup
    sec_axes = axes.twiny()
    sec_axes.set_xlabel("Lookback time [Gyr]")
    sec_axes.set_xscale("log")
    ticks = np.array([0.1, 1., 2., 5., 8., 11., 13.])
    tick_labels = [f"{t:g}" for t in ticks]
    lookback_times = astropy.units.Quantity(ticks, unit="Gyr")
    tick_pos = astropy.cosmology.z_at_value(
        planck15.lookback_time, lookback_times
    )
    sec_axes.set_xticks(tick_pos.value, labels=tick_labels)
    sec_axes.set_xlim(redshifts[stop], redshifts[start])

    # set minor ticks for lookback time correctly
    minor_ticks = np.concatenate(
        [
            np.arange(0.01, 0.1, 0.01),
            np.arange(0.1, 1, 0.1),
            np.arange(1, 13, 1),
        ]
    )
    minor_labels = ["" for _ in minor_ticks]
    lookback_times_minor = astropy.units.Quantity(minor_ticks, unit="Gyr")
    minor_tick_pos = astropy.cosmology.z_at_value(
        planck15.lookback_time, lookback_times_minor
    )
    sec_axes.set_xticks(minor_tick_pos.value, minor=True, labels=minor_labels)

    # redshifts as proxy values for snapnum
    redshift_proxies = redshifts[start:stop + 1]
    return redshift_proxies


def label_snapshots_with_redshift(
    axes: Axes,
    start: int,
    stop: int,
    axis_limits: tuple[float, float] | None = None,
    which_axis: Literal["x", "y"] = "x",
    tick_positions_z: NDArray = np.array([0.01, 0.1, 0.5, 1, 2, 5]),
    tick_positions_t: NDArray = np.array([0.1, 1, 5, 8, 11, 13]),
) -> NDArray | None:
    """
    Label an axis with snapshot numbers with redshift instead.

    Function takes an axis, which has snapshots at either x or y-axis
    and labels them with redshift. The spacing between the snapshots is
    not changed, meaning they remain evenly spaced. This is the difference
    to :func:`make_redshift_axis`, which changes the spacing between the
    snapshots.

    Function also adds a secondary axis showing the lookback time.

    :param axes: The figure axes object for which to label the snapshot
        axes with redshifts.
    :param start: The number of the first snapshot to place on the axis.
    :param stop: The number of the last snapshot to place on the axis.
    :param axis_limits: The limits of the axis in units of snapshot
        numbers. This must be set to correctly align lookback time with
        the corresponding redshifts. If not set or set to None, the
        axis limits will be set to ``[start, stop]``.
    :param which_axis: Which of the two axes to treat as redshift axis.
        Must be either ``x`` or ``y``. Defaults to x-axis.
    :param tick_positions_z: Array of redshifts at which to place a
        major tick and corresponding label. Defaults to numbered ticks
        at redshifts 0.01, 0.1, 0.5, 1, 2, and 5.
    :param tick_positions_t: Array of lookback times in Gyr at which to
        place a major tick and a corresponding label. Defaults to
        numbered ticks at lookback times 0.1 Gyr, 1 Gyr, 5 Gyr, 8 Gyr,
        11 Gyr, and 13 Gyr.
    :return: The array of snapshots, evenly spaced between ``start`` and
        ``stop``, to use as values for the transformed axis.
    """
    if which_axis not in ["x", "y"]:
        logging.error(f"Invalid axis for redshift given: {which_axis}.")
        return

    # set axis limits
    if axis_limits is None:
        axis_limits = (start, stop)

    # set up vars for interpolation
    snaps = np.flip(np.arange(0, 100, step=1))
    zs = np.flip(constants.REDSHIFTS)
    ts = np.flip(constants.LOOKBACK_TIMES)

    # set up secondary axis
    if which_axis == "x":
        sec_axes = axes.twiny()
    else:
        sec_axes = axes.twinx()

    # zip args
    args = (
        (axes, tick_positions_z, zs, "Redshift z"),
        (sec_axes, tick_positions_t, ts, "Lookback time [Gyr]")
    )

    # set up the axes
    for ax, ticks, interp_xs, label in args:
        tick_positions_s = np.interp(ticks, interp_xs, snaps)
        # set tick labels
        getattr(ax, f"set_{which_axis}label")(label)
        tick_labels = [f"{x:g}" for x in ticks]
        getattr(ax, f"set_{which_axis}ticks")(
            tick_positions_s, labels=tick_labels
        )
        # set minor ticks
        minor_ticks = np.concatenate(
            [
                np.arange(0.01, 0.1, 0.01),
                np.arange(0.1, 1, 0.1),
                np.arange(1, 15, 1),
            ]
        )
        minor_ticks_s = np.interp(minor_ticks, interp_xs, snaps)
        getattr(ax, f"set_{which_axis}ticks")(minor_ticks_s, minor=True)
        # set axis limits
        getattr(ax, f"set_{which_axis}lim")(axis_limits)

    return np.arange(start, stop + 1, step=1)


def plot_cluster_line_plot(
    figure: Figure,
    axes: Axes,
    xs: NDArray,
    quantity: NDArray,
    cluster_masses: NDArray,
    cmap: str | matplotlib.colors.Colormap = "jet",
    min_mass: float = 14.2,
    max_mass: float = 15.4,
    alpha: float = 0.15,
    use_discrete_norm: bool = False,
) -> tuple[Figure, Axes]:
    """
    Plot a line for every cluster, colored by their mass.

    Function takes a quantity in the form of an array of shape (N, X)
    where N is the number of clusters and X is the number of x-values as
    well as an array of masses of shape (N, ) and plots the values for
    every cluster as a line, which is colored according to the cluster
    mass in the specified colormap.

    .. attention:: Masses must be in units of log10 of solar masses!

    :param figure: The figure object onto which the colorbar is added.
    :param axes: The axes object onto which the lines are drawn.
    :param xs: The array of x-values of shape (X, ).
    :param quantity: The array of y-values for every cluster, of shape
        (N, X).
    :param cluster_masses: The array of cluster masses of shape (N, ).
        The masses must be supplied in log(M_sol)!
    :param cmap: The name of the colormap to use or the colormap object
        itself.
    :param min_mass: Minimum mass to show on the colorbar, in units of
        log10 of solar masses. Defaults to 14.2.
    :param max_mass: Maximum mass to show on the colorbar, in units of
        log10 of solar masses. Defaults to 15.4.
    :param use_discrete_norm: When set to True, the colorbar will not be
        continuous but rather discrete, coloring all masses in bins of
        0.2 dex from ``min_mass`` to ``max_mas`` the same color.
    :param alpha: The alpha-value for the colored lines of individual
        clusters.
    :return: Returns the figure and axes objects as tuple for convenience;
        both are however altered in place and do not need to be updated.
    """
    n_clusters = cluster_masses.shape[0]

    # create colormap
    if isinstance(cmap, str):
        cmap = matplotlib.cm.get_cmap(cmap)

    if use_discrete_norm:
        boundaries = np.arange(min_mass, max_mass, step=0.2)
        norm = matplotlib.colors.BoundaryNorm(boundaries, cmap.N, clip=True)
    else:
        norm = matplotlib.colors.Normalize(vmin=min_mass, vmax=max_mass)
    colors = [cmap(norm(mass)) for mass in cluster_masses]

    # plot mean, median, etc.
    plot_config = {
        "marker": "none",
        "linestyle": "solid",
        "alpha": alpha,
    }
    for i in range(n_clusters):
        axes.plot(
            xs,
            quantity[i],
            color=colors[i],
            zorder=cluster_masses[i],
            **plot_config,
        )
    figure.colorbar(
        matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap),
        ax=axes,
        location="right",
        label="$log_{10} M_{200}$ at z = 0",
    )

    # plot mean and median
    m_config = {"marker": "none", "color": "black", "zorder": 20}
    mean = np.nanmean(quantity, axis=0)
    axes.plot(xs, mean, ls="solid", **m_config)
    median = np.nanmedian(quantity, axis=0)
    axes.plot(xs, median, ls="dashed", **m_config)

    return figure, axes
