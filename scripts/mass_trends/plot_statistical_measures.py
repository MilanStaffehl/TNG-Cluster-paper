from __future__ import annotations

import argparse
import logging
import logging.config
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.lines import Line2D

root_dir = Path(__file__).parents[2].resolve()
sys.path.insert(0, str(root_dir / "src"))

from library.config import config, logging_config
from library.plotting import colormaps

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from numpy.typing import NDArray


def main(args: argparse.Namespace) -> None:
    """Create plot of gas mass trends for individual halos"""
    log_cfg = logging_config.get_logging_config(logging.INFO)
    logging.config.dictConfig(log_cfg)
    where = "cluster core" if args.core else "full cluster"
    logging.info(f"Plotting {args.what} for {where}.")

    # get a default config
    cfg = config.get_default_config("TNG-Cluster")

    # load config file
    config_file = root_dir / "src/pipelines/mass_trends/plot_config.yaml"
    with open(config_file, "r") as f:
        stream = f.read()
    configuration = yaml.full_load(stream)

    # create a figure
    fig, axes = plt.subplots(figsize=(10, 2.5))
    if args.what in ["pcc", "pcc-log", "pcc-loglog"]:
        key = args.what.replace("-", "_")
        axes.set_ylabel("R (Pearson CC)")
        if args.core:
            ylims = [-0.9, 1.05]
        else:
            ylims = [-.55, .55]
        axes.set_ylim(ylims)
        # add lines
        axes.hlines(
            [-1, -.5, .5, 1], -0.02, 1.02, linestyles="dotted", colors="black"
        )
        axes.hlines(0, -0.02, 1.02, linestyles="solid", colors="black")
        inset_axes = None
    elif args.what == "ratio":
        key = "ratios"
        axes.set_ylabel(r"$\alpha$ (Upper-to-lower ratio)")
        # axes.set_ylim([0, 2])
        axes.set_yscale("log")
        axes.hlines(1, -0.02, 1.02, linestyles="solid", colors="black")
        # add a secondary axes to better display the scale
        inset_axes = axes.inset_axes([0.5, 0.42, 0.48, 0.5])
        inset_axes.set_ylim([0.5, 1.5])
        inset_axes.hlines(1, -0.02, 1.02, linestyles="solid", colors="black")
    else:
        logging.error(f"Unknown plot type: {args.what}")
        return

    # determine file add-in if needed
    addin = "_core" if args.core else ""

    # load the y-data and create axes ticks
    y_data = np.zeros((len(configuration), 7))
    x_tick_labels = []
    for i, field in enumerate(configuration.keys()):
        # load y-data
        filepath = cfg.data_home / "mass_trends" / "statistical_measures"
        fieldname = field.replace('-', '_')
        filename = f"mass_trends_clusters_{fieldname}{addin}_statistics.npz"
        if args.use_mass:
            filename = filename.replace(
                "statistics", "statistics_absolute_gas_mass"
            )
        with np.load(filepath / filename) as data_file:
            y_data[i] = data_file[key]

        # get the human-readable tick label
        x_tick_labels.append(rf"${configuration[field]['label']['dev'][0]}$")

    # set axes ticks
    xs = np.linspace(0, 1, num=len(x_tick_labels))
    axes.set_xticks(xs, x_tick_labels, rotation=270)
    if inset_axes is not None:
        inset_axes.set_xticks(xs, [], rotation=270)

    # add grid to better associate points to a quantity
    grid_config = {
        "which": "major",
        "axis": "x",
        "linestyle": "solid",
        "color": "lightgrey",
        "linewidth": 1,
        "alpha": 0.9
    }
    axes.grid(**grid_config)
    if inset_axes is not None:
        inset_axes.grid(**grid_config)

    # plot the data
    offsets = 0.004 * np.array([1, -1])
    marker_config = {
        "marker": "o",
        "markersize": 8,
        "alpha": 0.8,
    }
    errbar_config = {
        "marker": "x",
        "markersize": 10,
        "alpha": 1,
        "capsize": 3,
        "elinewidth": 2,
        "capthick": 2,
        "markeredgewidth": 2,
    }

    # overplot points onto main axes
    _plot_points(axes, xs, y_data, offsets, marker_config, errbar_config)

    # create custom legend handles:
    handles = _create_legend_handles(marker_config, errbar_config)
    axes.legend(
        handles=handles,
        ncols=len(handles) // 2,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.35),
    )

    # if an inset axes exists, overplot points there with updated config
    if inset_axes is not None:
        marker_config.update({"markersize": 6})
        errbar_config.update({"markersize": 8, "markeredgewidth": 1.5})
        _plot_points(
            inset_axes, xs, y_data, offsets, marker_config, errbar_config
        )

    figurepath = cfg.figures_home / "mass_trends" / "clusters"
    mass_suffix = "_absolute_gas_mass" if args.use_mass else ""
    sfx = f"{key}{addin}{mass_suffix}"
    filename = f"mass_trends_statistical_measure_{sfx}.{args.fig_ext}"
    if not figurepath.exists():
        logging.info(f"Creating missing figures directory {figurepath}.")
        figurepath.mkdir(parents=True)
    fig.savefig(figurepath / filename, bbox_inches="tight")
    logging.info(f"Saved figure to file: {figurepath / filename}.")


def _plot_points(
    axes: Axes,
    xs: NDArray,
    y_data: NDArray,
    offsets: NDArray,
    marker_config: dict[str, Any],
    errbar_config: dict[str, Any]
) -> None:
    """
    Plot the data points onto the given axes with the given config.

    :param axes: Axes onto which to plot the points.
    :param xs: The x-values for the data points. Shape (X, )
    :param y_data: The y-values for the data points. Shape (X, N) where
        N is the number of data points per quantity.
    :param offsets: x-offsets for points. Ever point with an even index
        is offset by ``offsets[0]``, every point with an odd index is
        offset by ``offsets[1]``.
    :param marker_config: Keyword args for plotting points.
    :param errbar_config: Keyword args for plotting mean and std as
        points with errorbars.
    :return: None
    """
    for i, x in enumerate(xs):
        # Don't ask... we need the indices that each entry of the y-data
        # would have in a sorted y_data array, and this does the job...
        offset_indices = np.argsort(np.argsort(y_data[i]))
        for j in range(7):
            # sample color
            color = colormaps.sample_cmap("jet", 7, j)
            offset = offsets[offset_indices[j] % 2]
            axes.plot(x + offset, y_data[i][j], **marker_config, color=color)

        # add mean as a black errorbar
        mean = np.nanmean(y_data[i])
        std = np.nanstd(y_data[i])
        axes.errorbar(x, mean, yerr=std, **errbar_config, color="black")


def _create_legend_handles(
    marker_options: dict[str, Any], ebar_options: dict[str, Any]
) -> list[Line2D]:
    """
    Return a list of legend handles required for the summary plot.

    :param marker_options: A keyword dict for marker options.
    :param ebar_options: A keyword dict for the errorbar options.
    :return: A list of 8 handles to pass to the legend drawing function.
    """
    handles = []
    for i in range(7):
        min_mass = 14.0 + 0.2 * i
        max_mass = min_mass + 0.2
        label = rf"$10^{{{min_mass:.1f}}} - 10^{{{max_mass:.1f}}} M_\odot$"
        handles.append(
            Line2D(
                [], [],
                color=colormaps.sample_cmap("jet", 7, i),
                ls="",
                label=label,
                **marker_options
            )
        )
    # append a marker for the total
    handles.append(
        Line2D(
            [],
            [],
            color="black",
            ls="",
            label="Mean and std",
            marker=ebar_options["marker"],
            markersize=ebar_options["markersize"],
            markeredgewidth=ebar_options["markeredgewidth"],
        )
    )
    return handles


DESCRIPTION = """Plot statistical measures of the cool gas in clusters.

Script plots the selected statistical measure (Pearson correlation
coefficient or a ratio, see below) of every selected cluster property
with either cool gas mass fraction or absolute cool gas mass in seven
mass bins of 0.2 dex from 10^14 to 10^15.4 solar masses, i.e. it shows
how much each of the selected properties correlates with the cool gas
mass or cool gas fraction. The plot consists of a single panel showing,
for every one of the cluster properties described in
`src/pipelines/mass_trends/plot_config.yaml`, how the cool gas mass/fraction
in each of the cluster mass bins correlates with that property.

The options are the normal linear Pearson correlation coefficient
(`--what pcc`), the Pearson correlation coefficient taken on a semi-log
scale, i.e. with the cool gas fraction/mass in log scale, but the cluster
property in linear scale (`--what pcc-log`), the Pearson correlation
coefficient taken on a double-log scale, i.e. with both cool gas
mass/fraction and cluster property in loga scale (`--what pcc-loglog`),
or the ratio of mean value above bin median to mean value below bin
median (`--what ratio`).

This script requires that the statistical measure under scrutiny has
been written to file previously by running the `plot_cool_gas_mass_trends.py`
script for the corresponding property at least once, using the `--to-file`
option.
"""

if __name__ == "__main__":
    # construct parser
    parser = argparse.ArgumentParser(
        prog=f"python {Path(__file__).name}",
        description=DESCRIPTION,
    )
    parser.add_argument(
        "-w",
        "--what",
        help=(
            "What to plot: Pearson correlation coefficient or upper-to-lower "
            "mean color ratio. Defaults to Pearson correlation coefficient."
        ),
        dest="what",
        choices=["pcc", "pcc-log", "pcc-loglog", "ratio"],
        default="pcc",
    )
    parser.add_argument(
        "--absolute-mass",
        help=(
            "Plot the statistical measure for absolute gas mass instead of "
            "cool gas fraction."
        ),
        dest="use_mass",
        action="store_true",
    )
    parser.add_argument(
        "-cc",
        "--cluster-core",
        help=(
            "Plot the statistical quantity of choice for the cluster-core "
            "region instead of the full cluster."
        ),
        action="store_true",
        dest="core",
    )
    parser.add_argument(
        "--ext",
        help="File extension for the plot files. Defaults to pdf.",
        dest="fig_ext",
        type=str,
        default="pdf",
        choices=["pdf", "png"]
    )

    # parse arguments
    try:
        args = parser.parse_args()
        main(args)
    except KeyboardInterrupt:
        print("Execution forcefully stopped.")
        sys.exit(1)
