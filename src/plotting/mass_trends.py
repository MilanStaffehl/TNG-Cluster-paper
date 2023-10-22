"""
Plotting tools for mass trends.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

from plotting import util

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
    fig, axes = plt.subplots(nrows=2, figsize=(3.5, 6), sharex=True)
    axes[1].set_xlabel(r"Halo mass [$\log M_\odot$]")
    axes[0].set_ylabel("Gas fraction")
    axes[0].set_yscale("log")
    axes[1].set_ylabel(r"Gas mass [$M_\odot$]")
    axes[1].set_yscale("log")

    # colors
    colors = {
        "cool": "mediumblue",
        "warm": "purple",
        "hot": "darkorange",
    }
    plot_config = {
        "marker": "o",
        "markersize": 4,
        "linestyle": "none",
    }
    ebar_config = {
        "fmt": "x",
        "markersize": 4,
        "linestyle": "none",
    }

    # plot points for mean fraction
    axes[0].plot(
        halo_masses,
        cool_gas[0][0],
        color=colors["cool"],
        label="Cool",
        **plot_config
    )
    axes[0].plot(
        halo_masses,
        warm_gas[0][0],
        color=colors["warm"],
        label="Warm",
        **plot_config
    )
    axes[0].plot(
        halo_masses,
        hot_gas[0][0],
        color=colors["hot"],
        label="Hot",
        **plot_config
    )
    # plot points for median fraction
    axes[0].errorbar(
        halo_masses,
        cool_gas[0][1],
        yerr=cool_gas_err[0],
        color=colors["cool"],
        **ebar_config
    )
    axes[0].errorbar(
        halo_masses,
        warm_gas[0][1],
        yerr=warm_gas_err[0],
        color=colors["warm"],
        **ebar_config
    )
    axes[0].errorbar(
        halo_masses,
        hot_gas[0][1],
        yerr=hot_gas_err[0],
        color=colors["hot"],
        **ebar_config
    )

    # plot points for mean mass
    axes[1].plot(
        halo_masses,
        cool_gas[1][0],
        color=colors["cool"],
        label="Cool",
        **plot_config
    )
    axes[1].plot(
        halo_masses,
        warm_gas[1][0],
        color=colors["warm"],
        label="Warm",
        **plot_config
    )
    axes[1].plot(
        halo_masses,
        hot_gas[1][0],
        color=colors["hot"],
        label="Hot",
        **plot_config
    )
    # plot points for median fraction
    axes[1].errorbar(
        halo_masses,
        cool_gas[1][1],
        yerr=cool_gas_err[1],
        color=colors["cool"],
        **ebar_config
    )
    axes[1].errorbar(
        halo_masses,
        warm_gas[1][1],
        yerr=warm_gas_err[1],
        color=colors["warm"],
        **ebar_config
    )
    axes[1].errorbar(
        halo_masses,
        hot_gas[1][1],
        yerr=hot_gas_err[1],
        color=colors["hot"],
        **ebar_config
    )

    axes[0].legend()
    axes[1].legend()

    return fig, axes


def plot_gas_mass_trends_individuals(
    halo_masses: NDArray,
    gas_data: NDArray,
    binned_halo_masses: NDArray,
    binned_halo_masses_err: NDArray,
    cool_gas: NDArray,
    warm_gas: NDArray,
    hot_gas: NDArray,
    cool_gas_err: NDArray,
    warm_gas_err: NDArray,
    hot_gas_err: NDArray
) -> tuple[Figure, Axes]:
    """
    Plot the mass trends of gas mass and gas fraction with halo mass.

    The lot will consist of two columns and three rows, making six
    subplots. The left column will show the gas fraction trend and
    the right column the gas mass trend. The first row shows that trend
    for cold gas, the second for warm gas, and the third for hot gas.
    All subplots will also show the

    All gas data arrays are expected to have shape (2, M) where
    the first axis differentiates the weight type (fraction and mass),
    the second axis the mass bins, i.e. the values. The individual values
    are either the gas fraction of the corresponding regime in that mass
    bin or the total gas mass in units of solar masses.

    Opposed to this, the error arrays are of shape (2, 2, M), where the
    first axis again holds the weight type, the second axis the lower
    and upper length of the error bars, and the last axis hold the values.

    Function does not save the figure to file but returns it.

    :param halo_masses: The array of halo masses of shape (H, ) where
        H is the total number of halos in the snapshot. Must be given in
        units of log M_sol.
    :param gas_data: The array containing the data points for the gas
        fraction and mass for all three temperature regimes for every
        halo. Shape must be (H, 2, 3), where the second axis contains
        the fraction and mass data respectively and the third axis the
        cool, warm and hot gas respectively.
    :param binned_halo_masses: Array of shape (M, ) holding the average
        halo masses per mass bin in units of log M_sol.
    :param binned_halo_masses_err: The error in the halo masses per bin,
        shape (M, ). Must be in units of log M_sol.
    :param cool_gas: Cool gas data, shape (2, M)
    :param warm_gas: Warm gas data, shape (2, M)
    :param hot_gas: Hot gas data, shape (2, M)
    :param cool_gas_err: Cool gas errors, shape (2, 2, M)
    :param warm_gas_err: Warm gas errors, shape (2, 2, M)
    :param hot_gas_err: Hot gas errors, shape (2, 2, M)
    :return: Figure and axes objects after being created.
    """
    # set limits on plottable area
    mass_lims = (5e7, 5e14)
    frac_lims = (5e-4, 1.5)
    xlims = (8, 15)  # in log M_sol
    # create bins for the histograms in log scale for y-axis
    N_BINS = 60
    xbins = np.linspace(xlims[0], xlims[1], N_BINS, endpoint=True)
    frac_bins = np.logspace(
        np.log10(frac_lims[0]), np.log10(frac_lims[1]), N_BINS, endpoint=True
    )
    mass_bins = np.logspace(
        np.log10(mass_lims[0]), np.log10(mass_lims[1]), N_BINS, endpoint=True
    )
    # create figure
    fig, axes = plt.subplots(
        nrows=3,
        ncols=2,
        figsize=(5, 8),
        sharex=True,
        sharey=False,
        gridspec_kw={"hspace": 0, "wspace": 0}
    )
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
    plot_config = {
        "marker": "o",
        "markersize": 4,
        "linestyle": "none",
        "capsize": 2,
        "color": "black",
        "zorder": 10,
    }

    # cool gas fraction
    axes[0][0].errorbar(
        binned_halo_masses,
        cool_gas[0],
        xerr=binned_halo_masses_err,
        yerr=cool_gas_err[0],
        label="Cool",
        **plot_config,
    )
    axes[0][0].hist2d(
        halo_masses,
        gas_data[:, 0, 0],
        cmap=util.custom_cmap(colors["cool"]),
        range=(xlims, frac_lims),
        bins=(xbins, frac_bins),
    )
    # cool gas mass
    axes[0][1].errorbar(
        binned_halo_masses,
        cool_gas[1],
        xerr=binned_halo_masses_err,
        yerr=cool_gas_err[1],
        label="Cool",
        **plot_config,
    )
    axes[0][1].hist2d(
        halo_masses,
        gas_data[:, 1, 0],
        cmap=util.custom_cmap(colors["cool"]),
        range=(xlims, mass_lims),
        bins=(xbins, mass_bins),
    )
    # warm gas fraction
    axes[1][0].errorbar(
        binned_halo_masses,
        warm_gas[0],
        xerr=binned_halo_masses_err,
        yerr=warm_gas_err[0],
        label="Warm",
        **plot_config,
    )
    axes[1][0].hist2d(
        halo_masses,
        gas_data[:, 0, 1],
        cmap=util.custom_cmap(colors["warm"]),
        range=(xlims, frac_lims),
        bins=(xbins, frac_bins),
    )
    # warm gas mass
    axes[1][1].errorbar(
        binned_halo_masses,
        warm_gas[1],
        xerr=binned_halo_masses_err,
        yerr=warm_gas_err[1],
        label="Warm",
        **plot_config,
    )
    axes[1][1].hist2d(
        halo_masses,
        gas_data[:, 1, 1],
        cmap=util.custom_cmap(colors["warm"]),
        range=(xlims, mass_lims),
        bins=(xbins, mass_bins),
    )
    # hot gas fraction
    axes[2][0].errorbar(
        binned_halo_masses,
        hot_gas[0],
        xerr=binned_halo_masses_err,
        yerr=hot_gas_err[0],
        label="Hot",
        **plot_config,
    )
    axes[2][0].hist2d(
        halo_masses,
        gas_data[:, 0, 2],
        cmap=util.custom_cmap(colors["hot"]),
        range=(xlims, frac_lims),
        bins=(xbins, frac_bins),
    )
    # hot gas mass
    axes[2][1].errorbar(
        binned_halo_masses,
        hot_gas[1],
        xerr=binned_halo_masses_err,
        yerr=hot_gas_err[1],
        label="Hot",
        **plot_config,
    )
    axes[2][1].hist2d(
        halo_masses,
        gas_data[:, 1, 2],
        cmap=util.custom_cmap(colors["hot"]),
        range=(xlims, mass_lims),
        bins=(xbins, mass_bins),
    )

    return fig, axes
