"""
Plotting tools for mass trends.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from library.plotting import util

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from numpy.typing import NDArray


def plot_gas_mass_trends(
    halo_masses: NDArray,
    cool_gas: NDArray,
    warm_gas: NDArray,
    hot_gas: NDArray,
    cool_gas_err: NDArray,
    warm_gas_err: NDArray,
    hot_gas_err: NDArray
) -> tuple[Figure, Axes]:
    """
    Plot the mass trends of gas mass and gas fraction with halo mass.

    All gas data arrays are expected to have shape (2, 2, M) where
    the first axis differentiates the weight type (fraction and mass),
    the second axis the data type (mean and median) and the third
    axis the mass bins, i.e. the values. The individual values are
    either the gas fraction of the corresponding regime in that mass
    bin or the total gas mass in units of solar masses.

    Similarly, the error arrays are of shape (2, 2, M), where the
    first axis again holds the weight type, and the second axis the
    lower and upper error values.

    Function does not save the figure to file but returns it.

    :param halo_masses: Array of shape (M, ) holding the halo masses
        in units of solar masses.
    :param cool_gas: Cool gas data, shape (2, 2, M)
    :param warm_gas: Warm gas data, shape (2, 2, M)
    :param hot_gas: Hot gas data, shape (2, 2, M)
    :param cool_gas_err: Cool gas errors, shape (2, 2, M)
    :param warm_gas_err: Warm gas errors, shape (2, 2, M)
    :param hot_gas_err: Hot gas errors, shape (2, 2, M)
    :return: Figure and axes objects after being created.
    """
    raise NotImplementedError("Not implemented yet.")
    # fig, axes = plt.subplots(nrows=2, figsize=(3.5, 6), sharex=True)
    # axes[1].set_xlabel(r"Halo mass [$\log M_\odot$]")
    # axes[0].set_ylabel("Gas fraction")
    # axes[0].set_yscale("log")
    # axes[1].set_ylabel(r"Gas mass [$M_\odot$]")
    # axes[1].set_yscale("log")

    # return fig, axes


def plot_gas_mass_trends_individuals(
    halo_masses: NDArray,
    gas_data: NDArray,
    n_bins: int = 60,
) -> tuple[Figure, Axes]:
    """
    Plot the mass trends of gas mass and gas fraction with halo mass.

    The plot will consist of two columns and three rows, making six
    subplots. The left column will show the gas fraction trend and
    the right column the gas mass trend. The first row shows that trend
    for cold gas, the second for warm gas, and the third for hot gas.

    The figures will show a 2D histogram of the distribution of the data
    points. This is necessary especially for large data sets. The number
    of bins can optionally be set as well.

    Function does not save the figure to file but returns it.

    :param halo_masses: The array of halo masses of shape (H, ) where
        H is the total number of halos in the snapshot. Must be given in
        units of log M_sol.
    :param gas_data: The array containing the data points for the gas
        fraction and mass for all three temperature regimes for every
        halo. Shape must be (H, 2, 3), where the second axis contains
        the fraction and mass data respectively and the third axis the
        cool, warm and hot gas respectively.
    :param n_bins: The number of bins for the histogram in both x and
        y direction. Optional, defaults to 60.
    :return: Figure and axes objects after being created.
    """
    # set limits on plottable area
    mass_lims = (1e7, 5e14)
    frac_lims = (5e-4, 1.5)
    xlims = (8, 15)  # in log M_sol
    # create bins for the histograms in log scale for y-axis
    xbins = np.linspace(xlims[0], xlims[1], n_bins, endpoint=True)
    frac_bins = np.logspace(
        np.log10(frac_lims[0]), np.log10(frac_lims[1]), n_bins, endpoint=True
    )
    mass_bins = np.logspace(
        np.log10(mass_lims[0]), np.log10(mass_lims[1]), n_bins, endpoint=True
    )
    # create figure
    fig, axes = plt.subplots(
        nrows=3,
        ncols=2,
        figsize=(5, 6),
        sharex=True,
        sharey=False,
        gridspec_kw={"hspace": 0, "wspace": 0}
    )
    fig.set_tight_layout(True)
    axes[-1][0].set_xlabel(r"Halo mass [$\log M_\odot$]")
    axes[-1][1].set_xlabel(r"Halo mass [$\log M_\odot$]")
    axes[0][0].set_title("Gas fraction")
    axes[0][1].set_title("Gas mass")
    for i in range(3):
        for j in range(2):
            axes[i][j].set_yscale("log")
            if j == 0:
                # left column
                axes[i][j].set_ylabel("Gas fraction")
                axes[i][j].set_ylim(frac_lims)
            else:
                # right column
                axes[i][j].set_ylabel(r"Gas mass [$M_\odot$]")
                axes[i][j].set_ylim(mass_lims)
                axes[i][j].yaxis.set_label_position("right")
                axes[i][j].yaxis.tick_right()

    # colors
    colors = {
        "cool": (0, 0, 205),  # mediumblue
        "warm": (128, 0, 128),  # purple
        "hot": (255, 140, 0),  # darkorange
    }

    # font size
    fontsize = 8

    # cool gas fraction
    h = axes[0][0].hist2d(
        halo_masses,
        gas_data[:, 0, 0],
        cmap=util.custom_cmap(colors["cool"]),
        range=(xlims, frac_lims),
        bins=(xbins, frac_bins),
        norm="log",
    )
    inset = inset_axes(axes[0][0], width="4%", height="35%", loc="upper right")
    cb = fig.colorbar(
        h[3], cax=inset, orientation="vertical", ticks=[1, 10, 100]
    )
    cb.ax.yaxis.set_ticks_position("left")
    cb.ax.tick_params(labelsize=fontsize)

    # cool gas mass
    h = axes[0][1].hist2d(
        halo_masses,
        gas_data[:, 1, 0],
        cmap=util.custom_cmap(colors["cool"]),
        range=(xlims, mass_lims),
        bins=(xbins, mass_bins),
        norm="log",
    )
    inset = inset_axes(axes[0][1], width="4%", height="35%", loc="upper right")
    cb = fig.colorbar(
        h[3], cax=inset, orientation="vertical", ticks=[1, 10, 100]
    )
    cb.ax.yaxis.set_ticks_position("left")
    cb.ax.tick_params(labelsize=fontsize)

    # warm gas fraction
    h = axes[1][0].hist2d(
        halo_masses,
        gas_data[:, 0, 1],
        cmap=util.custom_cmap(colors["warm"]),
        range=(xlims, frac_lims),
        bins=(xbins, frac_bins),
        norm="log",
    )
    inset = inset_axes(axes[1][0], width="4%", height="35%", loc="upper right")
    cb = fig.colorbar(
        h[3], cax=inset, orientation="vertical", ticks=[1, 10, 100]
    )
    cb.ax.yaxis.set_ticks_position("left")
    cb.ax.tick_params(labelsize=fontsize)

    # warm gas mass
    axes[1][1].hist2d(
        halo_masses,
        gas_data[:, 1, 1],
        cmap=util.custom_cmap(colors["warm"]),
        range=(xlims, mass_lims),
        bins=(xbins, mass_bins),
        norm="log",
    )
    inset = inset_axes(axes[1][1], width="4%", height="35%", loc="upper right")
    cb = fig.colorbar(
        h[3], cax=inset, orientation="vertical", ticks=[1, 10, 100]
    )
    cb.ax.yaxis.set_ticks_position("left")
    cb.ax.tick_params(labelsize=fontsize)

    # hot gas fraction
    h = axes[2][0].hist2d(
        halo_masses,
        gas_data[:, 0, 2],
        cmap=util.custom_cmap(colors["hot"]),
        range=(xlims, frac_lims),
        bins=(xbins, frac_bins),
        norm="log",
    )
    inset = inset_axes(axes[2][0], width="4%", height="35%", loc="lower right")
    cb = fig.colorbar(
        h[3], cax=inset, orientation="vertical", ticks=[1, 10, 100]
    )
    cb.ax.yaxis.set_ticks_position("left")
    cb.ax.tick_params(labelsize=fontsize)

    # hot gas mass
    h = axes[2][1].hist2d(
        halo_masses,
        gas_data[:, 1, 2],
        cmap=util.custom_cmap(colors["hot"]),
        range=(xlims, mass_lims),
        bins=(xbins, mass_bins),
        norm="log",
    )
    inset = inset_axes(axes[2][1], width="4%", height="35%", loc="lower right")
    cb = fig.colorbar(
        h[3], cax=inset, orientation="vertical", ticks=[1, 10, 100]
    )
    cb.ax.yaxis.set_ticks_position("left")
    cb.ax.tick_params(labelsize=fontsize)

    return fig, axes


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
