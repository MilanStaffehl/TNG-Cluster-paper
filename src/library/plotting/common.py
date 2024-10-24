"""
Common plotting utilities.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Literal, Sequence

import astropy.cosmology
import astropy.units
import matplotlib.cm
import matplotlib.colors
import numpy as np

from library import constants

if TYPE_CHECKING:
    from matplotlib.axes import Axes
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
    cmap: str | matplotlib.colors.Colormap = "plasma",
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
    :return: Returns the figure and axes objects as tuple for convenience;
        both are however altered in place and do not need to be updated.
    """
    n_clusters = cluster_masses.shape[0]

    # create colormap
    if isinstance(cmap, str):
        cmap = matplotlib.cm.get_cmap(cmap)
    norm = matplotlib.colors.Normalize(vmin=14.0, vmax=15.4)
    colors = [cmap(norm(mass)) for mass in cluster_masses]

    # plot mean, median, etc.
    plot_config = {
        "marker": "none",
        "linestyle": "solid",
        "alpha": 0.1,
    }
    for i in range(n_clusters):
        axes.plot(
            xs,
            quantity[i],
            color=colors[i],
            **plot_config,
        )
    figure.colorbar(
        matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap),
        ax=axes,
        location="right",
        label="$log_{10} M_{200c}$ at z = 0",
    )

    # plot mean and median
    m_config = {"marker": "none", "color": "black"}
    mean = np.nanmean(quantity, axis=0)
    axes.plot(xs, mean, ls="solid", **m_config)
    median = np.nanmedian(quantity, axis=0)
    axes.plot(xs, median, ls="dashed", **m_config)

    return figure, axes
