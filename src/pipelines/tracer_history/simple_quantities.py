"""
Trace back some simple quantities of the tracer particles.
"""
from __future__ import annotations

import dataclasses
import logging
from typing import TYPE_CHECKING, ClassVar

import h5py
import matplotlib.pyplot as plt
import numpy as np

from library import constants
from library.plotting import common
from pipelines import base

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclasses.dataclass
class PlotSimpleQuantityWithTimePipeline(base.Pipeline):
    """Load data from hdf5 archive and plot it in various ways"""

    quantity: str  # name of the dataset in the archive
    quantity_label: str  # y-axis label for quantity
    color: str  # color for faint lines
    make_ridgeline: bool = False

    n_clusters: ClassVar[int] = 352
    n_snaps: ClassVar[int] = 100 - constants.MIN_SNAP

    def run(self) -> int:
        """Load and plot data"""
        # Step 0: check archive exists
        if not self.config.cool_gas_history.exists():
            logging.fatal(
                f"Did not find cool gas archive file "
                f"{self.config.cool_gas_history}."
            )
            return 1

        # Step 1: allocate memory for quantities to trace
        quantity_mean = np.zeros((self.n_clusters, self.n_snaps))
        quantity_median = np.zeros_like(quantity_mean)
        quantity_min = np.zeros_like(quantity_mean)
        quantity_max = np.zeros_like(quantity_mean)
        # TODO: allocate memory for 2D hists

        # Step 2: open the archive
        f = h5py.File(self.config.cool_gas_history, "r")

        # Step 3: load quantity and process for plotting
        for zoom_id in range(self.n_clusters):
            group = f"ZoomRegion_{zoom_id:03d}"
            # check field exists
            try:
                quantity = f[group][self.quantity]
            except KeyError:
                logging.fatal(
                    f"Zoom-in {zoom_id} is missing dataset {self.quantity}. "
                    f"Cannot proceed with plotting."
                )
                return 2

            quantity_mean[zoom_id] = np.nanmean(quantity, axis=1)
            quantity_median[zoom_id] = np.nanmedian(quantity, axis=1)
            quantity_max[zoom_id] = np.nanmax(quantity, axis=1)
            quantity_min[zoom_id] = np.nanmin(quantity, axis=1)

        # Step 4: plot line plots
        self._plot_and_save_lineplots(
            quantity_mean, quantity_median, quantity_min, quantity_max
        )

        # Step 5: plot 2D histograms
        self._plot_and_save_2dhistograms()

        # Step 6: plot ridgeline plots
        self._plot_and_save_ridgelineplots()

        return 0

    def _plot_and_save_lineplots(
        self, means: NDArray, medians: NDArray, mins: NDArray, maxs: NDArray
    ) -> None:
        """
        Plot the development of the quantity with time.

        Function plots the mean and median lines for individual clusters
        plus the mean and median over all clusters in a line plot over
        redshift. It saves the plots to file.

        :param means: Array of mean of the quantity for every cluster,
            i.e. an array of shape (100, 352) where every entry is the
            mean of the gas quantity at that snapshot for that cluster.
        :param medians: Array of medians of the quantity for every
            cluster, i.e. an array of shape (100, 352) where every entry
            is the median of the gas quantity at that snapshot for that
            cluster.
        :param mins: Array of minimum of the quantity for every cluster,
            i.e. an array of shape (100, 352) where every entry is the
            min of the gas quantity at that snapshot for that cluster.
        :param maxs: Array of maximums of the quantity for every cluster,
            i.e. an array of shape (100, 352) where every entry is the
            max of the gas quantity at that snapshot for that cluster.
        :return: None
        """
        plot_types = {
            "Mean": means,
            "Median": medians,
            "Minimum": mins,
            "Maximum": maxs,
        }
        for label_prefix, plot_quantity in plot_types.items():
            logging.info(
                f"Creating line plot for {label_prefix} {self.quantity_label}"
            )
            # create figure and configure axes
            fig, axes = plt.subplots(figsize=(4, 4))
            xs = common.make_redshift_plot(axes, start=constants.MIN_SNAP)
            axes.set_ylabel(f"{label_prefix} {self.quantity_label}")
            axes.set_yscale("log")

            # plot mean, median, etc.
            plot_config = {
                "marker": "none",
                "linestyle": "solid",
                "alpha": 0.1,
                "color": self.color,  # TODO: color by cluster mass
            }
            axes.plot(xs, plot_quantity.transpose(), **plot_config)

            # plot mean and median
            m_config = {"marker": "none", "color": "black"}
            mean = np.mean(plot_quantity, axis=0)
            axes.plot(xs, mean, ls="solid", **m_config)
            median = np.median(plot_quantity, axis=0)
            axes.plot(xs, median, ls="dashed", **m_config)

            # save figure
            self._save_fig(
                fig, ident_flag=label_prefix.lower(), subdir="./1d_plots"
            )

    def _plot_and_save_ridgelineplots(self):
        # TODO: implement
        pass

    def _plot_and_save_2dhistograms(self):
        # TODO: implement
        pass
