"""
Plot how different subhalo properties develop along the MPB.
"""
from __future__ import annotations

import dataclasses
import logging
from typing import TYPE_CHECKING

import illustris_python as il
import matplotlib.pyplot as plt
import numpy as np
import yaml

from library import units
from library.data_acquisition import halos_daq
from pipelines import base

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclasses.dataclass
class PlotMergerTreePropertiesPipeline(base.Pipeline):
    """
    Plot properties of the primary subhalo of every cluster.

    Follow the main progenitor branch of every cluster primary subhalo
    and plot a property of it with time. Properties can include mass,
    position, velocity, etc.
    """

    field_name: str

    def __post_init__(self):
        with open("mpb_plot_config.yaml", "r") as file:
            full_cfg = yaml.full_load(file.read())
            try:
                self.plot_config = full_cfg[self.field_name]
            except KeyError:
                msg = f"Unsupported field: {self.field_name}."
                logging.fatal(msg)
                raise NotImplementedError(msg)
            logging.debug(
                f"Loaded plot config for field {self.field_name}: "
                f"{self.plot_config}"
            )

    def run(self) -> int:
        """
        Create a plot for the current property.

        :return: Exit code
        """
        # Step 0: verify directories
        self._verify_directories()
        if self.config.sim_name != "TNG-Cluster":
            logging.fatal(
                f"Currently unsupported simulation: {self.config.sim_name}."
            )
            return 1

        # Step 1: Load IDs of primary subhalos
        cluster_data = halos_daq.get_halo_properties(
            self.config.base_path,
            self.config.snap_num,
            fields=["GroupFirstSub"],
            cluster_restrict=True,
        )

        # Step 2: allocate memory for result
        n_clusters = cluster_data["GroupFirstSub"].size
        if self.plot_config["dims"] > 1:
            result = np.empty((100, n_clusters, self.plot_config["dims"]))
        else:
            result = np.empty((100, n_clusters))
        # fill with NaNs
        result[:] = np.nan

        # Step 3: Load the main progenitor branch for every primary
        logging.info("Loading main progenitor branch for all clusters.")
        for i, primary_subhalo_id in enumerate(cluster_data["GroupFirstSub"]):
            logging.debug(f"Loading MPB for zoom-in {i}/{n_clusters}.")
            # Step 3.1: Load main progenitor branch
            subhalo_data = il.sublink.loadTree(
                self.config.base_path,
                self.config.snap_num,
                primary_subhalo_id,
                fields=[self.plot_config["field"], "SnapNum"],
                onlyMPB=True,
            )

            # Step 3.2: convert units, append to data
            result[subhalo_data["SnapNum"], i] = units.UnitConverter.convert(
                subhalo_data[self.plot_config["field"]],
                self.plot_config["field"],
            )

        # Step 4: plot data
        logging.info(f"Plotting {self.field_name} plot.")
        self._plot(result)

        return 0

    def _plot(self, data: NDArray) -> None:
        """
        Plot the quantity vs time.

        :param data: An array of shape (100, N) or (100, N, 3), where N
            is the number of clusters and 3 is the dimensionality of the
            quantity.
        :return: None, figure is saved to file.
        """
        if len(data.shape) > 2:
            for k in range(data.shape[-1]):
                self._plot_single_panel(data[:, :, k], k)
        else:
            self._plot_single_panel(data, None)

    def _plot_single_panel(
        self, data: NDArray, axis_index: int | None
    ) -> None:
        """
        Plot a single panel of the quantity with time.

        :param data: Array of the data of shape (100, N) where N is the
            number of clusters.
        :param axis_index: The index of the current axis for spacial
            quantities. Must be either 0, 1, or 2. Set to None for non-
            spatial quantities.
        :return: None, figure is saved to file.
        """
        if axis_index is not None:
            cur_axis = ["x", "y", "z"][axis_index]
        else:
            cur_axis = ""

        fig, (ax1, ax2) = plt.subplots(
            nrows=2, figsize=(4, 5), sharex=True, height_ratios=[4, 1]
        )

        # add labels, configure axes
        y_label = self.plot_config["label"].format(cur_axis)
        ax1.set_ylabel(y_label)
        ymin = self.plot_config["ymin"]
        ymax = self.plot_config["ymax"]
        if ymin is not None:
            ymin = float(ymin)
        if ymax is not None:
            ymax = float(ymax)
        if any([ymin, ymax]):
            ax1.set_ylim((ymin, ymax))
        if self.plot_config["log"]:
            ax1.set_yscale("log")
            ax2.set_yscale("log")

        # axis for diff
        ax2.set_ylabel(r"$\Delta$ ($\log_{10}$)")
        ax2.set_xlabel("Snapshot")

        # plot data
        xs = np.arange(0, 100, step=1)
        plot_config = {
            "alpha": 0.05,
            "color": "black",
            "linestyle": "solid",
            "marker": "none",
        }
        ax1.plot(xs, data, **plot_config)

        # plot difference between current and previous
        diff = data[1:] - data[:-1]
        ax2.hlines(0, xs[0], xs[-1], colors="grey", linestyles="solid")
        ax2.plot(xs[:-1], diff, **plot_config)

        # save figure
        if axis_index is None:
            suffix = ""
        else:
            suffix = f"_{cur_axis}"
        self._save_fig(
            fig, ident_flag=f"{self.field_name}{suffix}", tight_layout=True
        )
