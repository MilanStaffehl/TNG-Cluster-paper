"""
Plotting tools for mass trends.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt

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
