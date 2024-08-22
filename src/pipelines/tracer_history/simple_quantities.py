"""
Trace back some simple quantities of the tracer particles.
"""
from __future__ import annotations

import dataclasses
import logging
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

import h5py
import matplotlib.cm
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import yaml

from library import constants
from library.plotting import common
from library.processing import statistics
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
    n_bins: ClassVar[int] = 50  # number of bins

    def run(self) -> int:
        """Load and plot data"""
        # Step 0: check archive exists
        if not self.config.cool_gas_history.exists():
            logging.fatal(
                f"Did not find cool gas archive file "
                f"{self.config.cool_gas_history}."
            )
            return 1

        # Step 1: Find plot config
        cfg = Path(__file__).parent / "simple_quantities_plot_config.yaml"
        with open(cfg, "r") as cfg_file:
            stream = cfg_file.read()
        try:
            plot_config = yaml.full_load(stream)[self.quantity]
            hist_range = plot_config["min"], plot_config["max"]
            log = plot_config["log"]
        except KeyError:
            logging.warning(
                f"Found no plot config for quantity {self.quantity}, will set "
                f"no boundaries for histograms."
            )
            hist_range = None
            log = False

        # Step 2: allocate memory for quantities to trace
        quantity_mean = np.zeros((self.n_clusters, self.n_snaps))
        quantity_median = np.zeros_like(quantity_mean)
        quantity_min = np.zeros_like(quantity_mean)
        quantity_max = np.zeros_like(quantity_mean)
        quantity_hists = np.zeros((self.n_clusters, self.n_snaps, self.n_bins))

        # Step 3: open the archive
        f = h5py.File(self.config.cool_gas_history, "r")

        # Step 4: load quantity and process for plotting
        logging.info(
            "Loading particle data from file and processing. May take a while."
        )
        for zoom_id in range(self.n_clusters):
            logging.debug(
                f"Loading and processing {self.quantity} for zoom-in "
                f"{zoom_id}."
            )
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

            quantity_mean[zoom_id] = np.nanmean(
                quantity[constants.MIN_SNAP:], axis=1
            )
            quantity_median[zoom_id] = np.nanmedian(
                quantity[constants.MIN_SNAP:], axis=1
            )
            quantity_max[zoom_id] = np.nanmax(
                quantity[constants.MIN_SNAP:], axis=1
            )
            quantity_min[zoom_id] = np.nanmin(
                quantity[constants.MIN_SNAP:], axis=1
            )
            for i, snap in enumerate(range(constants.MIN_SNAP, 100, 1)):
                if log:
                    q = np.log10(quantity[snap])
                else:
                    q = quantity[snap]
                quantity_hists[zoom_id, i] = np.histogram(
                    q, self.n_bins, range=hist_range
                )[0]

        # Step 4: plot line plots
        self._plot_and_save_lineplots(
            quantity_mean, quantity_median, quantity_min, quantity_max
        )

        if hist_range is None:
            logging.warning(
                "Plot config was incomplete, cannot plot ridgeline and 2D "
                "histogram plots."
            )
            return 0

        # Step 5: plot 2D histograms
        self._plot_and_save_2dhistograms(quantity_hists)

        # Step 6: plot ridgeline plots
        self._plot_and_save_ridgelineplots(
            quantity_hists, hist_range[0], hist_range[1]
        )

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

    def _plot_and_save_ridgelineplots(
        self, hists: NDArray, minimum: float, maximum: float
    ) -> None:
        """
        Plot a ridgeline plot of the development of the quantity.

        Plot shows the 1D distribution of the quantity at different
        redshifts. It plots the mean and median distribution over all
        clusters in every snapshot.

        :param hists: Array of histograms, of shape (352, 92, N) where
            N is the number of bins.
        :param minimum: Leftmost edge of the histogram bins.
        :param maximum: Rightmost edge of the histogram bins.
        :return: None, figure saved to file.
        """
        # color map
        cmap = matplotlib.cm.get_cmap("gist_heat")
        norm = matplotlib.colors.Normalize(vmin=constants.MIN_SNAP, vmax=120)

        for method in ["mean", "median"]:
            # Step 1: stack histograms
            stacked_hist = statistics.stack_histograms(
                hists, method, axis=0
            )[0]

            # Step 2: set up figure
            fig, axes = plt.subplots(figsize=(5, 5))
            q_label = self.quantity_label[0].lower() + self.quantity_label[1:]
            axes.set_xlabel(f"{method.capitalize()} {q_label}")
            axes.set_ylabel("Snap num")

            # Step 3: set up x-values
            xs = np.linspace(minimum, maximum, num=self.n_bins)

            # Step 4: set up y-values and y-axis ticks and labels
            y_base = np.arange(constants.MIN_SNAP, 100, step=1)

            # Step 5: plot baselines
            axes.hlines(y_base, minimum, maximum, color="grey", linewidths=1)

            # Step 6: plot ridgelines
            for i in range(self.n_snaps):
                ys = y_base[i] + stacked_hist[i] / np.sum(stacked_hist[i]) * 30
                color = cmap(norm(i + constants.MIN_SNAP))
                axes.plot(xs, ys, color=color)

            # Step 6: save figure
            self._save_fig(fig, ident_flag=method, subdir="2d_plots")
            logging.info(f"Saved {method} ridgeline plot to file.")

    def _plot_and_save_2dhistograms(self, hists: NDArray) -> None:
        # TODO: implement
        pass
