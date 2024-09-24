"""
Trace back some simple quantities of the tracer particles.
"""
from __future__ import annotations

import copy
import dataclasses
import logging
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

import h5py
import matplotlib.cm
import matplotlib.collections
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import yaml

from library import compute, constants
from library.data_acquisition import halos_daq, sublink_daq
from library.plotting import common
from library.plotting import plot_radial_profiles as plot_hists
from library.processing import statistics
from pipelines import base

if TYPE_CHECKING:
    from numpy.typing import NDArray

PLOT_TYPES = [
    "lineplot",
    "global2dhist",
    "globalridgeline",
    "zoomedridgeline",
]


class HistogramMixin:
    """Mixin to provide useful common methods."""


@dataclasses.dataclass
class PlotSimpleQuantityWithTimePipeline(base.Pipeline):
    """Load data from hdf5 archive and plot it in various ways"""

    quantity: str  # name of the dataset in the archive
    quantity_label: str  # y-axis label for quantity
    color: str  # color for faint lines
    plot_types: list[str] | None = None  # what to plot

    n_clusters: ClassVar[int] = 352
    n_snaps: ClassVar[int] = 100 - constants.MIN_SNAP
    n_bins: ClassVar[int] = 50  # number of bins

    def __post_init__(self):
        super().__post_init__()
        self.hist_range: tuple[float, float] | None = None
        self.log: bool | None = None
        if self.plot_types is None:
            self.plot_types = PLOT_TYPES

    def run(self) -> int:
        """Load and plot data"""
        logging.info(
            f"Starting pipeline to plot {', '.join(self.plot_types)} for "
            f"{self.quantity}."
        )

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
            self.hist_range = plot_config["min"], plot_config["max"]
            self.log = plot_config["log"]
        except KeyError:
            logging.warning(
                f"Found no plot config for quantity {self.quantity}, will set "
                f"no boundaries for histograms; 2D plot creation will be "
                f"skipped."
            )

        # Step 2: open the archive, ensure quantity exists
        f = h5py.File(self.config.cool_gas_history, "r")
        logging.info("Checking archive for all required datasets.")
        quantity_archived = True
        for zoom_id in range(self.n_clusters):
            grp = f"ZoomRegion_{zoom_id:03d}"
            if self.quantity not in f[grp].keys():
                logging.error(
                    f"Zoom-in {zoom_id} missing dataset {self.quantity}."
                )
                quantity_archived = False
            if "uniqueness_flags" not in f[grp].keys():
                logging.error(f"Zoom-in {zoom_id} missing uniqueness flags.")
                quantity_archived = False
        if not quantity_archived:
            logging.fatal(
                f"Quantity {self.quantity} is not archived for all zoom-ins. "
                f"Cannot proceed with plotting."
            )
            return 2
        logging.info("Archive OK. Continuing with plotting.")

        # Step 3: check for lineplots and plot
        if "lineplot" in self.plot_types:
            logging.info("Plotting line plots.")
            self._plot_and_save_lineplots(f)
            logging.info("Finished line plots, saved to file.")

        # Step 4: plot 2D histograms
        if "global2dhist" in self.plot_types:
            logging.info("Plotting global 2D histograms.")
            self._plot_and_save_2dhistograms(f)
            logging.info("Finished global 2D histograms, saved to file.")

        # Step 5: plot global ridgeline plot
        if "globalridgeline" in self.plot_types:
            logging.info("Plotting global ridgeline plots.")
            self._plot_and_save_ridgelineplots(f)
            logging.info("Finished global ridgeline plots, saved to file.")

        # Step 6: plot zoomed-in ridgeline plot
        if "zoomedridgeline" in self.plot_types:
            logging.info("Plotting zoomed-in ridgeline plots.")
            # see if zoomed-in plot it supported
            try:
                plot_config = yaml.full_load(stream)[self.quantity]
                zoomed_config = plot_config["zoomed"]
                hist_range = zoomed_config["min"], zoomed_config["max"]
            except KeyError:
                logging.warning(
                    "None or incomplete config for zoomed ridgeline plot. "
                    "Skipping."
                )
            else:
                # change ranges
                old_range = copy.copy(self.hist_range)
                self.hist_range = hist_range
                self._plot_and_save_ridgelineplots(f, True, "_zoomed_in")
                logging.info(
                    "Finished zoomed-in ridgeline plots, saved to file."
                )
                self.hist_range = old_range  # reset to old value

        f.close()
        logging.info("Done plotting! Saved all plots to file.")
        return 0

    def _get_quantity_hists(self, archive_file: h5py.File) -> NDArray:
        """
        Create and return 2D histograms for all clusters.

        Function loads the quantity from the given archive file, and
        creates histograms of it at every snapshot. It does so taking
        into account possibly duplicated particles, counting every
        particle only once. The histograms are normalized before being
        saved into an array of shape (N, S, B) where N is the number of
        clusters, S is the number of snapshots analyzed, and B is the
        number of bins.

        Normalizations are:

        - Distance to MP: normalized to the volume of the respective
          radial shell.
        - All other: normalized such that the histogram at every snapshot
          sums to unity.

        :param archive_file: Opened cool gas history archive file.
        :return: Array of histograms of the distribution of the quantity
            at all snapshots analyzed for all clusters. Suitably
            normalized for the given quantity.
        """
        quantity_hists = np.zeros((self.n_clusters, self.n_snaps, self.n_bins))
        for zoom_id in range(self.n_clusters):
            logging.debug(f"Creating histogram for zoom-in {zoom_id}.")
            group = f"ZoomRegion_{zoom_id:03d}"
            quantity = archive_file[group][self.quantity]
            uniqueness_flags = archive_file[group]["uniqueness_flags"]
            for i, snap in enumerate(range(constants.MIN_SNAP, 100, 1)):
                unique_q = quantity[snap][uniqueness_flags[snap] == 1]
                if self.log:
                    q = np.log10(unique_q)
                else:
                    q = unique_q
                # different normalizations
                if self.quantity == "DistanceToMP":
                    hist = statistics.volume_normalized_radial_profile(
                        q,
                        np.ones_like(q),
                        self.n_bins,
                        radial_range=self.hist_range,
                    )[0]
                else:
                    hist = np.histogram(
                        q, self.n_bins, range=self.hist_range
                    )[0]
                    hist = hist / np.sum(hist)  # normalize to unity
                quantity_hists[zoom_id, i] = hist
        return quantity_hists

    def _plot_and_save_lineplots(self, archive_file: h5py.File) -> None:
        """
        Plot the development of the quantity with time.

        Function plots the mean and median lines for individual clusters
        plus the mean and median over all clusters in a line plot over
        redshift. It saves the plots to file.

        Note: Below, S is the number of snaps from the minimum snap
        considered to snap 99.

        :param archive_file: The opened archive file containing the
            particle data and uniqueness flags.
        :return: None
        """
        # load quantities and find max, min, mean, and median
        quantity_mean = np.zeros((self.n_clusters, self.n_snaps))
        quantity_median = np.zeros_like(quantity_mean)
        quantity_min = np.zeros_like(quantity_mean)
        quantity_max = np.zeros_like(quantity_mean)

        for zoom_id in range(self.n_clusters):
            group = f"ZoomRegion_{zoom_id:03d}"
            quantity = archive_file[group][self.quantity]
            uniqueness_flags = archive_file[group]["uniqueness_flags"]

            quantity_max[zoom_id] = np.nanmax(
                quantity[constants.MIN_SNAP:], axis=1
            )
            quantity_min[zoom_id] = np.nanmin(
                quantity[constants.MIN_SNAP:], axis=1
            )
            for i, snap in enumerate(range(constants.MIN_SNAP, 100, 1)):
                # make sure particles are not counted twice
                unique_q = quantity[snap][uniqueness_flags[snap] == 1]
                quantity_mean[zoom_id, i] = np.nanmean(unique_q)
                quantity_median[zoom_id, i] = np.nanmedian(unique_q)

        # plot
        plot_types = {
            "Mean": quantity_mean,
            "Median": quantity_median,
            "Minimum": quantity_min,
            "Maximum": quantity_max,
        }
        # create a colormap for the current mass range
        cmap = matplotlib.cm.get_cmap("plasma")
        norm = matplotlib.colors.Normalize(vmin=14.0, vmax=15.4)
        # load masses to color plots by them
        cluster_data = halos_daq.get_halo_properties(
            self.config.base_path,
            self.config.snap_num,
            [self.config.mass_field],
            cluster_restrict=True,
        )
        masses = np.log10(cluster_data[self.config.mass_field])
        colors = [cmap(norm(mass)) for mass in masses]

        for label_prefix, plot_quantity in plot_types.items():
            logging.info(
                f"Creating line plot for {label_prefix} {self.quantity_label}"
            )
            # create figure and configure axes
            fig, axes = plt.subplots(figsize=(5, 4))
            xs = common.make_redshift_plot(axes, start=constants.MIN_SNAP)
            label = self.quantity_label[0].lower() + self.quantity_label[1:]
            axes.set_ylabel(f"{label_prefix} {label}")
            axes.set_yscale("log")

            # plot mean, median, etc.
            plot_config = {
                "marker": "none",
                "linestyle": "solid",
                "alpha": 0.1,
            }
            for i in range(self.n_clusters):
                axes.plot(
                    xs,
                    plot_quantity[i],
                    color=colors[i],
                    **plot_config,
                )
            fig.colorbar(
                matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap),
                ax=axes,
                location="right",
                label="$log_{10} M_{200c}$ at z = 0",
            )

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
        self,
        archive_file: h5py.File,
        log_height: bool = False,
        suffix: str = "",
    ) -> None:
        """
        Plot a ridgeline plot of the development of the quantity.

        Plot shows the 1D distribution of the quantity at different
        redshifts. It plots the mean and median distribution over all
        clusters in every snapshot.

        :param archive_file: The opened cool gas history archive file.
        :param log_height: Whether to plot the height of the line in log
            space.
        :param suffix: Suffix for the file name, to distinguish different
            ridgeline plot types.
        :return: None, figure saved to file.
        """
        # Check if plotting is possible
        if self.hist_range is None or self.log is None:
            logging.error("Cannot plot global ridgeline; missing plot config.")
            return

        # Load data
        quantity_hists = self._get_quantity_hists(archive_file)
        minimum, maximum = self.hist_range

        # color map
        cmap = matplotlib.cm.get_cmap("gist_heat")
        norm = matplotlib.colors.Normalize(vmin=constants.MIN_SNAP, vmax=120)

        for method in ["mean", "median"]:
            # Step 1: stack histograms
            stacked_hist = statistics.stack_histograms(
                quantity_hists, method, axis=0
            )[0]

            # Step 2: set up figure
            fig, axes = plt.subplots(figsize=(5, 4))
            q_label = self.quantity_label[0].lower() + self.quantity_label[1:]
            if self.log:
                q_label.replace("[", r"[$\log_{10}$")
            axes.set_xlabel(f"{method.capitalize()} {q_label}")

            # Step 3: set up x-values
            xs = np.linspace(minimum, maximum, num=self.n_bins)

            # Step 4: set up y-values and y-axis ticks and labels
            y_base = common.label_snapshots_with_redshift(
                axes, constants.MIN_SNAP, 99, (5, 115), "y"
            )

            # Step 5: plot baselines
            axes.hlines(y_base, minimum, maximum, color="grey", linewidths=1)

            # Step 6: plot ridgelines
            if log_height:
                stacked_hist = np.log10(stacked_hist)
                stacked_hist[stacked_hist == -np.inf] = 0
                # shift up negative values
                min_value = np.min(stacked_hist[stacked_hist != 0])
                if min_value < 0:
                    stacked_hist[stacked_hist != 0] -= min_value
            m = 150 if log_height else 30
            for i in range(self.n_snaps):
                ys = y_base[i] + stacked_hist[i] / np.sum(stacked_hist[i]) * m
                color = cmap(norm(i + constants.MIN_SNAP))
                axes.plot(xs, ys, color=color, zorder=120 - i)

            # Step 6: save figure
            ident_flag = f"ridgeline_{method}{suffix}"
            self._save_fig(fig, ident_flag=ident_flag, subdir="2d_plots")
            logging.info(f"Saved {method} ridgeline plot to file.")

    def _plot_and_save_2dhistograms(self, archive_file: h5py.File) -> None:
        """
        Plot a 2D histogram plot of the development of the quantity.

        Plot shows the 1D distribution of the quantity at different
        redshifts using a 2D histogram. It plots the mean and median
        distribution over all clusters in every snapshot.

        :param archive_file: The opened cool gas history archive file.
        :return: None, figure saved to file.
        """
        # Check plotting is possible
        if self.hist_range is None or self.log is None:
            logging.error("Cannot plot 2D histogram; missing plot config.")
            return

        # Load data and create a histogram
        quantity_hists = self._get_quantity_hists(archive_file)

        # Plot the histograms
        for method in ["mean", "median"]:
            # Step 1: stack histograms
            stacked_hist = statistics.stack_histograms(
                quantity_hists, method, axis=0
            )[0]

            # Step 2: set up figure
            fig, axes = plt.subplots(figsize=(5.5, 4))
            q_label = self.quantity_label[0].lower() + self.quantity_label[1:]
            if self.log:
                q_label.replace("[", r"[$\log_{10}$")

            # Step 3: plot 2D histograms
            ranges = [
                constants.MIN_SNAP, 99, self.hist_range[0], self.hist_range[1]
            ]
            plot_hists.plot_2d_radial_profile(
                fig,
                axes,
                stacked_hist.transpose(),
                ranges=ranges,
                xlabel="Snap num",
                ylabel=f"{method.capitalize()} {q_label}",
                colormap="inferno",
                cbar_label="Normalized count [log]",
                scale="log",
                cbar_limits=(-4, None),
            )

            # Step 4: label x-axis appropriately
            common.label_snapshots_with_redshift(
                axes,
                constants.MIN_SNAP,
                99,
                tick_positions_z=np.array([0, 0.1, 0.5, 1, 2, 5]),
                tick_positions_t=np.array([0, 1, 5, 8, 11, 13]),
            )

            # Step 5: save figure
            ident_flag = f"2dhist_{method}"
            self._save_fig(fig, ident_flag=ident_flag, subdir="2d_plots")
            logging.info(f"Saved {method} 2D histogram plot to file.")


@dataclasses.dataclass
class PlotSimpleQuantitiesForSingleClusters(base.Pipeline):
    """
    Plot simple particle quantities, but for individual clusters.

    This includes plots for the development of the quantity for individual
    particles instead of means and medians.
    """

    quantity: str  # name of the dataset in the archive
    quantity_label: str  # y-axis label for quantity
    zoom_in: int  # the zoom-in region to plot
    part_limit: int | None = None  # limit plots to this many particles

    n_clusters: ClassVar[int] = 352
    n_snaps: ClassVar[int] = 100 - constants.MIN_SNAP
    n_bins: ClassVar[int] = 50  # number of bins

    def __post_init__(self):
        super().__post_init__()
        self.hist_range: tuple[float, float] | None = None
        self.log: bool = False
        self.individual_log: bool = False

    def run(self) -> int:
        """Load and plot data"""
        # Step 0: check archive exists, create paths
        self._verify_directories()
        if not self.config.cool_gas_history.exists():
            logging.fatal(
                f"Did not find cool gas archive file "
                f"{self.config.cool_gas_history}."
            )
            return 1

        # Step 1: get plot config
        cfg = Path(__file__).parent / "simple_quantities_plot_config.yaml"
        with open(cfg, "r") as cfg_file:
            stream = cfg_file.read()
        try:
            plot_config = yaml.full_load(stream)[self.quantity]
            self.hist_range = plot_config["min"], plot_config["max"]
            self.log = plot_config["log"]
            self.individual_log = plot_config["individual-log"]
        except KeyError:
            logging.warning(
                f"Found no plot config for quantity {self.quantity}, will set "
                f"scale to linear."
            )

        # Step 2: open the archive
        f = h5py.File(self.config.cool_gas_history, "r")

        # Step 3: extract the data required
        particle_data = f[f"ZoomRegion_{self.zoom_in:03d}/{self.quantity}"][()]
        uniqueness = f[f"ZoomRegion_{self.zoom_in:03d}/uniqueness_flags"][()]

        # Step 4: plot the data
        self._plot_time_development(particle_data)
        self._plot_2dhistogram(particle_data, uniqueness)

        f.close()
        return 0

    def _plot_time_development(self, particle_data: NDArray) -> None:
        """
        Plot, for every gas cell, the development of the quantity.

        The plot will contain a single line for every tracer. This means
        that it is entirely possible that lines will overlap whenever two
        or more tracers occupy the same particle cell, and also that
        these lines can be interrupted for quantities that exist only
        for gas particles, if the tracer is transferred to a star or BH
        particle. Therefore, these plots may not be useful for every
        quantity.

        :param particle_data: Array of shape (100, N) where N is the
            number oif cells (and therefore the number of lines the
            plot will have), and the first axis orders the data by snap
            number. The first axis must be ordered such that index i
            points to snap num i.
        :return: None, plots are saved to file.
        """
        logging.info(
            f"Plotting development of {self.quantity} for all particles of "
            f"zoom-in {self.zoom_in}."
        )

        if self.part_limit is not None:
            logging.info(
                f"Limiting particle data to only the first {self.part_limit} "
                f"particles."
            )
            particle_data = particle_data[:, :self.part_limit]

        if self.individual_log:
            particle_data = np.log10(particle_data)

        # set up figure and axes
        fig, axes = plt.subplots(figsize=(15, 15))  # must be LARGE!
        axes.set_ylabel(self.quantity_label)
        xs = common.make_redshift_plot(axes, start=constants.MIN_SNAP)

        # plot data
        logging.info("Plotting a line for every tracer. May take a while...")
        n_part = particle_data.shape[1]
        cmap = matplotlib.cm.get_cmap("hsv")
        norm = matplotlib.colors.Normalize(vmin=0, vmax=n_part)
        colors = cmap(norm(np.arange(0, n_part, step=1)))
        # BEHOLD: the absolute clusterfuck that matplotlib requires, just
        # to make LineCollection work. Whatever the developers are on, I
        # want some of that. Must be good stuff...
        ys = particle_data[constants.MIN_SNAP:, :]
        lines = [np.column_stack([xs, ys[:, i]]) for i in range(ys.shape[1])]
        lc = matplotlib.collections.LineCollection(
            lines, colors=colors, alpha=0.2
        )
        axes.add_collection(lc)
        axes.autoscale_view()
        axes.set_rasterization_zorder(5)

        # add characteristic cluster property as line
        logging.info("Overplotting characteristic cluster property.")
        if self.quantity == "Temperature":
            label = "Virial temperature at z = 0"
            cluster_data = halos_daq.get_halo_properties(
                self.config.base_path,
                self.config.snap_num,
                fields=[self.config.radius_field, self.config.mass_field],
                cluster_restrict=True
            )
            cluster_cq = compute.get_virial_temperature(
                cluster_data[self.config.mass_field][self.zoom_in],
                cluster_data[self.config.radius_field][self.zoom_in],
            )
        elif self.quantity == "DistanceToMP":
            label = r"$2R_{200c}$"
            primary_id = halos_daq.get_halo_properties(
                self.config.base_path,
                self.config.snap_num,
                ["GroupFirstSub"],
                cluster_restrict=True,
            )["GroupFirstSub"][self.zoom_in]
            mpb_data = sublink_daq.get_mpb_properties(
                self.config.base_path,
                self.config.snap_num,
                primary_id,
                fields=[self.config.radius_field],
                start_snap=constants.MIN_SNAP,
                log_warning=True,
            )
            cluster_cq = mpb_data[self.config.radius_field]
        else:
            logging.info(
                f"No characteristic property to plot for {self.quantity}."
            )
            label = None
            cluster_cq = np.NaN
        axes.plot(
            xs,
            cluster_cq if not self.individual_log else np.log10(cluster_cq),
            ls="dashed",
            color="black",
            label=label,
            zorder=10,
        )

        # add labels
        if label:
            axes.legend()

        # save fig
        logging.info("Saving plot to file, may take a while...")
        if self.part_limit is None:
            ident_flag = "all_particles"
        else:
            ident_flag = f"first_{self.part_limit:d}_particles"
        self._save_fig(
            fig, ident_flag=ident_flag, subdir=f"zoom_in_{self.zoom_in}"
        )

    def _plot_2dhistogram(
        self, particle_data: NDArray, uniqueness_flags: NDArray
    ) -> None:
        """
        Plot a 2D histogram plot of the development of the quantity.

        Plot shows the 1D distribution of the quantity at different
        redshifts using a 2D histogram.

        :param particle_data: The array of the quantity to plot of shape
            (S, N) where S is the number of snapshots and N the number of
            particles.
        :param uniqueness_flags: Uniqueness flags of the particles, must
            be of shape (S, N).
        :return: None, figure saved to file.
        """
        logging.info(f"Plotting 2D histogram of {self.quantity}.")
        # Check plotting is possible
        if self.hist_range is None or self.log is None:
            logging.error("Cannot plot 2D histogram; missing plot config.")
            return

        # Calculate the histogram
        quantity_hist = np.zeros((self.n_snaps, self.n_bins))
        for i, snap in enumerate(range(constants.MIN_SNAP, 100, 1)):
            unique_q = particle_data[snap][uniqueness_flags[snap] == 1]
            if self.log:
                q = np.log10(unique_q)
            else:
                q = unique_q
            # different normalizations
            if self.quantity == "DistanceToMP":
                hist = statistics.volume_normalized_radial_profile(
                    q,
                    np.ones_like(q),
                    self.n_bins,
                    radial_range=self.hist_range,
                )[0]
            else:
                hist = np.histogram(q, self.n_bins, range=self.hist_range)[0]
                hist = hist / np.sum(hist)  # normalize to unity
            quantity_hist[i] = hist

        # Step 2: set up figure
        fig, axes = plt.subplots(figsize=(5.5, 4))
        q_label = self.quantity_label[0].lower() + self.quantity_label[1:]
        if self.log:
            q_label.replace("[", r"[$\log_{10}$")

        # Step 3: plot 2D histograms
        ranges = [
            constants.MIN_SNAP, 99, self.hist_range[0], self.hist_range[1]
        ]
        plot_hists.plot_2d_radial_profile(
            fig,
            axes,
            quantity_hist.transpose(),
            ranges=ranges,
            xlabel="Snap num",
            ylabel=q_label,
            colormap="inferno",
            cbar_label="Normalized count [log]",
            scale="log",
            cbar_limits=(-4, None),
        )

        # Step 4: label x-axis appropriately
        common.label_snapshots_with_redshift(
            axes,
            constants.MIN_SNAP,
            99,
            tick_positions_z=np.array([0, 0.1, 0.5, 1, 2, 5]),
            tick_positions_t=np.array([0, 1, 5, 8, 11, 13]),
        )

        # Step 5: save figure
        ident_flag = f"z{self.zoom_in:03d}_2dhist"
        self._save_fig(
            fig, ident_flag=ident_flag, subdir=f"zoom_in_{self.zoom_in}"
        )
        logging.info("Saved 2D histogram plot to file.")
