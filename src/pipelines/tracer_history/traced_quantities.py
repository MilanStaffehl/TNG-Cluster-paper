"""
Trace back some simple quantities of the tracer particles.
"""
from __future__ import annotations

import copy
import dataclasses
import enum
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Protocol

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
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from numpy.typing import NDArray


class PlotType(enum.IntEnum):
    LINEPLOT = 0
    GLOBAL_2DHIST = 1
    GLOBAL_RIDGELINE = 2
    ZOOMED_RIDGELINE = 3


class PlotPipelineProtocol(Protocol):
    """Dummy protocol to make mixin work."""

    @property
    def log(self) -> bool:
        ...

    @property
    def hist_range(self) -> tuple[float, float]:
        ...

    @property
    def quantity(self) -> str:
        ...

    @property
    def volume_normalize(self) -> bool:
        ...


class HistogramMixin:
    """Mixin to provide common plotting utils."""

    def _plot_histogram(
        self: PlotPipelineProtocol,
        q_label: str,
        fig: Figure,
        axes: Axes,
        quantity_hist: NDArray
    ) -> NDArray:
        """
        Plot the given ``quantity_hist`` onto the given axes.

        Method plots the given histogram, assuming it is a physical
        quantity plotted vs snapshot numbers from ``MIN_SNAP`` to
        snapshot 99. The snapshots are automatically labeled with
        redshift and lookback time.

        :param q_label: The label for the y-axis, describing the
            quantity plotted vs. redshift. Must **not** contain any
            mention of log scaling, this is added automatically.
        :param fig: The figure object to plot on.
        :param axes: The axes object to plot on.
        :param quantity_hist: The histogram as an array of shape
            (S, N), i.e. the way it is returned from numpy's
            ``histogram2d`` as-is. Transformation for plotting is done
            by the method.
        :return: The x-values needed for overplotting other data,
            corresponding to redshifts of the 92 snapshots plotted.
        """
        if self.log:
            q_label = q_label.replace("[", r"[$\log_{10}$")

        # colorbar label
        if self.volume_normalize:
            cbar_lims = (-5, None)
            cbar_label = r"Tracer density [$\log_{10}(M_\odot / ckpc^3)$]"
        else:
            cbar_lims = (-4, None)
            cbar_label = "Normalized count [log]"

        # plot 2D histograms
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
            cbar_label=cbar_label,
            scale="log",
            cbar_limits=cbar_lims,
        )

        # label x-axis appropriately
        xs = common.label_snapshots_with_redshift(
            axes,
            constants.MIN_SNAP,
            99,
            tick_positions_z=np.array([0, 0.1, 0.5, 1, 2, 5]),
            tick_positions_t=np.array([0, 1, 5, 8, 11, 13]),
        )
        return xs


@dataclasses.dataclass
class PlotSimpleQuantityWithTimePipeline(HistogramMixin, base.Pipeline):
    """Load data from hdf5 archive and plot it in various ways"""

    quantity: str  # name of the dataset in the archive
    color: str  # color for faint lines
    volume_normalize: bool = False  # normalize by volume?
    normalize: bool = False  # whether to normalize to characteristic property
    plot_types: list[int] | None = None  # what to plot

    n_clusters: ClassVar[int] = 352
    n_snaps: ClassVar[int] = 100 - constants.MIN_SNAP
    n_bins: ClassVar[int] = 50  # number of bins

    def __post_init__(self):
        super().__post_init__()

        # plot types
        if self.plot_types is None:
            self.plot_types = [e.value for e in PlotType]

        # turn off volume normalization if it is not a distance plot
        if self.quantity != "DistanceToMP":
            self.volume_normalize = False

        # update config for plot
        category = "normalized" if self.normalize else "standard"
        cfg = Path(__file__).parent / "simple_quantities_plot_config.yaml"
        with open(cfg, "r") as cfg_file:
            stream = cfg_file.read()
        try:
            plot_config = yaml.full_load(stream)[self.quantity][category]
            self.hist_range = plot_config["min"], plot_config["max"]
            self.log = plot_config["log"]
            self.quantity_label = rf"{plot_config['label']}"
        except KeyError:
            logging.warning(
                f"Found no plot config for quantity {self.quantity}, or the"
                f"config is incomplete. Will set no boundaries for histograms;"
                f" 2D plot creation will be skipped."
            )
            self.hist_range = None
            self.log = None
            self.quantity_label = "no label"
        try:
            zoomed_config = yaml.full_load(stream)[self.quantity]["zoomed"]
            self.zoomed_range = zoomed_config["min"], zoomed_config["max"]
        except KeyError:
            self.zoomed_range = None
        logging.debug(
            f"Set the following plot config:\nhist_range: {self.hist_range}"
            f"\nlog: {self.log}\nquantity_label: {self.quantity_label}\n"
            f"zoomed_range: {self.zoomed_range}"
        )

    def run(self) -> int:
        """Load and plot data"""
        logging.info(
            f"Starting pipeline to plot {', '.join(self.plot_types)} for "
            f"{self.quantity}."
        )
        if self.volume_normalize:
            logging.info("Will normalize 2D histograms by shell volume.")

        # Step 0: check archive exists
        if not self.config.cool_gas_history.exists():
            logging.fatal(
                f"Did not find cool gas archive file "
                f"{self.config.cool_gas_history}."
            )
            return 1

        # Step 1: open the archive, ensure quantity exists
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

        # Step 2: check for lineplots and plot
        if PlotType.LINEPLOT in self.plot_types:
            logging.info("Plotting line plots.")
            self._plot_and_save_lineplots(f)
            logging.info("Finished line plots, saved to file.")

        # Step 3: plot 2D histograms
        if PlotType.GLOBAL_2DHIST in self.plot_types:
            logging.info("Plotting global 2D histograms.")
            self._plot_and_save_2dhistograms(f)
            logging.info("Finished global 2D histograms, saved to file.")

        # Step 4: plot global ridgeline plot
        if PlotType.GLOBAL_RIDGELINE in self.plot_types:
            logging.info("Plotting global ridgeline plots.")
            self._plot_and_save_ridgelineplots(f)
            logging.info("Finished global ridgeline plots, saved to file.")

        # Step 5: plot zoomed-in ridgeline plot
        if PlotType.ZOOMED_RIDGELINE in self.plot_types:
            logging.info("Plotting zoomed-in ridgeline plots.")
            # see if zoomed-in plot it supported
            if self.zoomed_range is None:
                logging.warning(
                    "None or incomplete config for zoomed ridgeline plot. "
                    "Skipping."
                )
            else:
                # change ranges
                old_range = copy.copy(self.hist_range)
                self.hist_range = self.zoomed_range
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
        logging.info(
            "Creating histogram data for all clusters. This can take a while."
        )
        quantity_hists = np.zeros((self.n_clusters, self.n_snaps, self.n_bins))
        normalization_factor = self._get_normalization()
        for zoom_id in range(self.n_clusters):
            logging.debug(f"Creating histogram for zoom-in {zoom_id}.")
            group = f"ZoomRegion_{zoom_id:03d}"
            quantity = archive_file[group][self.quantity]
            uniqueness_flags = archive_file[group]["uniqueness_flags"]
            for i, snap in enumerate(range(constants.MIN_SNAP, 100, 1)):
                unique_q = quantity[snap][uniqueness_flags[snap] == 1]
                normed_q = unique_q / normalization_factor[zoom_id, i]
                if self.log:
                    q = np.log10(normed_q)
                else:
                    q = normed_q
                # different normalizations
                if self.volume_normalize:
                    weights = np.ones_like(q) * constants.TRACER_MASS
                    hist = statistics.volume_normalized_radial_profile(
                        q,
                        weights,
                        self.n_bins,
                        radial_range=self.hist_range,
                        virial_radius=normalization_factor[zoom_id, i],
                        distances_are_log=self.log,
                    )[0]
                else:
                    hist = np.histogram(
                        q, self.n_bins, range=self.hist_range
                    )[0]
                    hist = hist / np.sum(hist)  # normalize to unity
                quantity_hists[zoom_id, i] = hist
        return quantity_hists

    def _get_normalization(self) -> NDArray:
        """
        Get a characteristic cluster property for the current cluster.

        Examples for characteristic property are virial temperature for
        the particle temperature plots or the virial radius for the
        distance plots.

        Method returns an array of shape (C, S) where C is the number of
        zoom-ins and S is the number of snapshots analyzed. Each entry
        is the characteristic property of that zoom-in at that snap for
        the current quantity (e.g. virial radius for distance etc.).

        :return: Array of characteristic property.
        """
        normalization = np.ones((self.n_clusters, self.n_snaps))
        if not self.normalize:
            return normalization  # no normalization

        logging.info("Loading characteristic cluster property.")
        primaries_ids = halos_daq.get_halo_properties(
            self.config.base_path,
            self.config.snap_num,
            ["GroupFirstSub"],
            cluster_restrict=True,
        )["GroupFirstSub"]
        for zoom_in in range(self.n_clusters):
            mpb_data = sublink_daq.get_mpb_properties(
                self.config.base_path,
                self.config.snap_num,
                primaries_ids[zoom_in],
                fields=[self.config.radius_field, self.config.mass_field],
                start_snap=constants.MIN_SNAP,
                log_warning=False,
            )
            if self.quantity == "Temperature":
                normalization[zoom_in] = compute.get_virial_temperature(
                    mpb_data[self.config.mass_field],
                    mpb_data[self.config.radius_field],
                )
            elif self.quantity == "DistanceToMP":
                normalization[zoom_in] = mpb_data[self.config.radius_field]
            else:
                logging.info(
                    f"No characteristic property for normalization available "
                    f"for {self.quantity}."
                )
                break
        return normalization

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
        # load masses to color plots by them
        cluster_data = halos_daq.get_halo_properties(
            self.config.base_path,
            self.config.snap_num,
            [self.config.mass_field],
            cluster_restrict=True,
        )
        masses = np.log10(cluster_data[self.config.mass_field])

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
            common.plot_cluster_line_plot(fig, axes, xs, plot_quantity, masses)

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
            logging.info(f"Plotting {method} ridgeline plot.")
            # Step 1: stack histograms
            stacked_hist = statistics.stack_histograms(
                quantity_hists, method, axis=0
            )[0]

            # Step 2: set up figure
            fig, axes = plt.subplots(figsize=(5, 4))
            q_label = self.quantity_label[0].lower() + self.quantity_label[1:]
            if self.log:
                q_label = q_label.replace("[", r"[$\log_{10}$")
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
            norm_flag = "_normalized" if self.normalize else ""
            volnorm_flag = "_volumenormalized" if self.volume_normalize else ""
            ident_flag = f"ridgeline_{method}{norm_flag}{volnorm_flag}{suffix}"
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
            logging.info(f"Plotting {method} 2D histogram plot.")
            # Step 1: stack histograms
            stacked_hist = statistics.stack_histograms(
                quantity_hists, method, axis=0
            )[0]

            # Step 2: set up figure
            fig, axes = plt.subplots(figsize=(5.5, 4))
            self._plot_histogram(self.quantity_label, fig, axes, stacked_hist)

            # Step 5: save figure
            norm_flag = "_normalized" if self.normalize else ""
            volnorm_flag = "_volumenormalized" if self.volume_normalize else ""
            ident_flag = f"2dhist_{method}{volnorm_flag}{norm_flag}"
            self._save_fig(fig, ident_flag=ident_flag, subdir="2d_plots")
            logging.info(f"Saved {method} 2D histogram plot to file.")


@dataclasses.dataclass
class PlotSimpleQuantitiesForSingleClusters(HistogramMixin, base.Pipeline):
    """
    Plot simple particle quantities, but for individual clusters.

    This includes plots for the development of the quantity for individual
    particles instead of means and medians.
    """

    quantity: str  # name of the dataset in the archive
    zoom_in: int  # the zoom-in region to plot
    part_limit: int | None = None  # limit plots to this many particles
    volume_normalize: bool = False
    plot_types: list[int] | None = None  # what to plot
    color_by: str | None = None  # what to color lines by

    n_clusters: ClassVar[int] = 352
    n_snaps: ClassVar[int] = 100 - constants.MIN_SNAP
    n_bins: ClassVar[int] = 50  # number of bins

    def __post_init__(self):
        super().__post_init__()

        if self.plot_types is None:
            self.plot_types = [e.value for e in PlotType]

        if self.quantity != "DistanceToMP":
            self.volume_normalize = False

        cfg = Path(__file__).parent / "simple_quantities_plot_config.yaml"
        with open(cfg, "r") as cfg_file:
            stream = cfg_file.read()
        try:
            plot_config = yaml.full_load(stream)[self.quantity]
            standard_config = plot_config["standard"]
            self.hist_range = standard_config["min"], standard_config["max"]
            self.log = standard_config["log"]
            self.quantity_label = standard_config["label"]
            self.individual_log = plot_config["individual-log"]
        except KeyError as e:
            logging.fatal(
                f"Plot config for quantity {self.quantity} incomplete: {e}."
            )
            raise RuntimeError("Incomplete plot config.")

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

        # Step 1: open the archive
        f = h5py.File(self.config.cool_gas_history, "r")

        # Step 2: extract the data required
        particle_data = f[f"ZoomRegion_{self.zoom_in:03d}/{self.quantity}"][()]
        uniqueness = f[f"ZoomRegion_{self.zoom_in:03d}/uniqueness_flags"][()]

        # Step 3: get characteristic cluster property
        logging.info("Loading characteristic cluster property.")
        primary_id = halos_daq.get_halo_properties(
            self.config.base_path,
            self.config.snap_num,
            ["GroupFirstSub"],
            cluster_restrict=True,
        )["GroupFirstSub"][self.zoom_in]
        if self.quantity == "Temperature":
            label = "Virial temperature"
            mpb_data = sublink_daq.get_mpb_properties(
                self.config.base_path,
                self.config.snap_num,
                primary_id,
                fields=[self.config.radius_field, self.config.mass_field],
                start_snap=constants.MIN_SNAP,
                log_warning=False,
            )
            cluster_cq = compute.get_virial_temperature(
                mpb_data[self.config.mass_field],
                mpb_data[self.config.radius_field],
            )
        elif self.quantity == "DistanceToMP":
            label = r"$R_{200c}$"
            mpb_data = sublink_daq.get_mpb_properties(
                self.config.base_path,
                self.config.snap_num,
                primary_id,
                fields=[self.config.radius_field],
                start_snap=constants.MIN_SNAP,
                log_warning=False,
            )
            cluster_cq = mpb_data[self.config.radius_field]
        else:
            logging.info(
                f"No characteristic property to plot for {self.quantity}."
            )
            label = None
            cluster_cq = np.empty(self.n_snaps)
            cluster_cq[:] = np.nan

        # Step 4: load color quantity if set
        if self.color_by == "parent-category":
            colors = f[f"ZoomRegion_{self.zoom_in:03d}/ParentCategory"][()]
        else:
            colors = None

        f.close()

        # Step 5: plot the data
        if PlotType.LINEPLOT in self.plot_types:
            logging.info(
                f"Plotting development of {self.quantity} for all particles "
                f"of zoom-in {self.zoom_in}."
            )
            self._plot_time_development(
                particle_data, cluster_cq, colors, label
            )
        if PlotType.GLOBAL_2DHIST in self.plot_types:
            logging.info(
                f"Plotting 2D histogram of {self.quantity} of zoom-in "
                f"{self.zoom_in}."
            )
            self._plot_2dhistogram(
                particle_data, uniqueness, cluster_cq, label
            )

        return 0

    def _plot_time_development(
        self,
        particle_data: NDArray,
        cluster_cq: NDArray,
        color_quantity: NDArray | None,
        label: str | None,
    ) -> None:
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
            number of cells (and therefore the number of lines the
            plot will have), and the first axis orders the data by snap
            number. The first axis must be ordered such that index i
            points to snap num i.
        :param cluster_cq: Characteristic property of the cluster,
            matching the current quantity to plot (e.g. virial temperature
            for plotting particle temperature). Must be an array of shape
            (S, ) where S is the number of snaps plotted. If no suitable
            cluster property exists, pass an array of NaNs instead.
        :param color_quantity: Either an array of data by which to color
            the individual tracer tracks, or None to color them by
            uniformly sampling a cyclic colormap.
        :param label: If a characteristic cluster quantity is provided,
            then this is the label to place in the legend. If none is
            provided or no legend shall be created, set this to None.
        :return: None, plots are saved to file.
        """
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
        if color_quantity is None:
            self._plot_rainbow_line_collection(axes, particle_data, xs)
        else:
            self._plot_categorized_line_collection(
                axes, particle_data, xs, color_quantity
            )

        # add characteristic cluster property as line
        if np.all(~np.isnan(cluster_cq)):
            logging.info(
                "Overplotting characteristic cluster property onto line plot."
            )
            handle_ccq, = axes.plot(
                xs,
                cluster_cq
                if not self.individual_log else np.log10(cluster_cq),
                ls="dashed",
                color="black",
                label=label,
                zorder=10,
            )
        else:
            handle_ccq = None

        # construct and add legend
        handles = []
        if handle_ccq is not None:
            handles.append(handle_ccq)
        if self.color_by is not None:
            self._add_additional_handles(handles)
        if label:
            axes.legend(handles=handles)

        # save fig
        logging.info("Saving plot to file, may take a while...")
        ident_flag = f"z{self.zoom_in:03d}_tracks_"
        if self.color_by is not None:
            split_by = self.color_by.replace("-", "_")
            ident_flag += f"{split_by}_"
        if self.part_limit is None:
            ident_flag += "all_particles"
        else:
            ident_flag += f"first_{self.part_limit:d}_particles"
        self._save_fig(
            fig, ident_flag=ident_flag, subdir=f"zoom_in_{self.zoom_in}"
        )
        logging.info("Done! Saved individual line plot to file!")

    def _plot_rainbow_line_collection(
        self, axes: Axes, particle_data: NDArray, xs: NDArray
    ) -> None:
        """
        Plot a line collection, with each line colored a different color.

        The lines are the tracer tracks given by ``particle_data``, and
        they are colored by sampling a cyclic colormap so that every
        line has its own color. Lines are directly added to the given
        axes, which is then auto-scaled to display all lines correctly.

        :param axes: The axes onto which to plot the line collection.
        :param particle_data: The array of particle properties. Must be
            of shape (100, N).
        :param xs: The array of x-values of shape (S, ) where S is the
            number of snaps from ``constants.MIN_SNAP`` to z = 0.
        :return: None
        """
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

    def _plot_categorized_line_collection(
        self,
        axes: Axes,
        particle_data: NDArray,
        xs: NDArray,
        color_quantity: NDArray,
    ) -> None:
        """
        Plot a line collection, colored by the given quantity.

        The lines are the tracer tracks given by ``particle_data``, and
        each segment is colored according to the data given as the
        ``color_quantity`` array, by sampling a colormap from the max
        value range. The lines are directly added to the given axes,
        which is then auto-scaled.

        :param axes: The axes onto which to plot the line collection.
        :param particle_data: The array of particle properties. Must be
            of shape (100, N).
        :param xs: The array of x-values of shape (S, ) where S is the
            number of snaps from ``constants.MIN_SNAP`` to z = 0.
        :param color_quantity: The quantity which determines the color
            of each line segment. Must be of shape (100, N).
        :return: None
        """
        # setup
        n_part = particle_data.shape[1]
        cmap, norm = self._get_norm_and_cmap(color_quantity)

        # Oh, you thought that stuff up there was bad? HA! HAHAHAHA!
        # Yeah, try plotting multiple lines with colored segments.
        # Impossible to do with LineCollection - except with the slowest
        # for-loop the world has ever seen. Enjoy...
        log_level = logging.getLogger("root").level
        for i in range(n_part):
            if log_level <= 15:
                perc = i / n_part * 100
                print(f"Plotting line {i}/{n_part} ({perc:.1f}%)", end="\r")
            colors = color_quantity[constants.MIN_SNAP + 1:, i]
            ys = particle_data[constants.MIN_SNAP:, i]
            start_points = np.stack((xs[:-1], ys[:-1]), axis=1)
            end_points = np.stack((xs[1:], ys[1:]), axis=1)
            lines = np.stack((start_points, end_points), axis=1)
            lc = matplotlib.collections.LineCollection(
                lines, array=colors, alpha=0.2, cmap=cmap, norm=norm
            )
            axes.add_collection(lc)

        axes.autoscale_view()
        axes.set_rasterization_zorder(5)

    def _get_norm_and_cmap(
        self, color_quantity: NDArray
    ) -> tuple[matplotlib.colors.Colormap, matplotlib.colors.Normalize]:
        """
        Return a cmap and norm for the current color quantity.

        :return: Tuple of appropriate cmap and norm objects.
        """
        if self.color_by == "parent-category":
            logging.info("Setting cmap and norm for parent category.")
            cmap = matplotlib.cm.get_cmap("turbo")
            norm = matplotlib.colors.Normalize(vmin=0, vmax=4)
        else:
            cmap = matplotlib.cm.get_cmap("hsv")
            vmin = np.nanmin(color_quantity[constants.MIN_SNAP:, :])
            vmax = np.nanmax(color_quantity[constants.MIN_SNAP:, :])
            norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
        return cmap, norm

    def _add_additional_handles(self, handles: list[Any]) -> None:
        """
        Add handles to the list of handles for plot legend creation.

        :param handles: A list of handles for the legend. Will be appended
            by new, appropriate handles.
        :return: None, list is altered in place.
        """
        pass

    def _plot_2dhistogram(
        self,
        particle_data: NDArray,
        uniqueness_flags: NDArray,
        cluster_cq: NDArray,
        label: str | None,
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
        :param cluster_cq: Characteristic property of the cluster,
            matching the current quantity to plot (e.g. virial temperature
            for plotting particle temperature). Must be an array of shape
            (S, ) where S is the number of snaps plotted. If no suitable
            cluster property exists, pass an array of NaNs instead.
        :param label: If a characteristic cluster quantity is provided,
            then this is the label to place in the legend. If none is
            provided or no legend shall be created, set this to None.
        :return: None, figure saved to file.
        """
        # Step 0: Check plotting is possible
        if self.hist_range is None or self.log is None:
            logging.error("Cannot plot 2D histogram; missing plot config.")
            return
        if self.volume_normalize:
            logging.info("2D histogram will be volume-normalized.")

        # Step 1: Calculate the histogram
        quantity_hist = np.zeros((self.n_snaps, self.n_bins))
        for i, snap in enumerate(range(constants.MIN_SNAP, 100, 1)):
            unique_q = particle_data[snap][uniqueness_flags[snap] == 1]
            if self.log:
                q = np.log10(unique_q)
            else:
                q = unique_q
            # different normalizations
            if self.volume_normalize:
                weights = np.ones_like(q) * constants.TRACER_MASS
                hist = statistics.volume_normalized_radial_profile(
                    q,
                    weights,
                    self.n_bins,
                    radial_range=self.hist_range,
                    distances_are_log=self.log,
                )[0]
            else:
                hist = np.histogram(q, self.n_bins, range=self.hist_range)[0]
                hist = hist / np.sum(hist)  # normalize to unity
            quantity_hist[i] = hist

        # Step 2: set up figure
        fig, axes = plt.subplots(figsize=(5.5, 4))

        # Step 3: overplot histogram
        xs = self._plot_histogram(
            self.quantity_label, fig, axes, quantity_hist
        )

        # Step 4: plot characteristic property
        if np.all(~np.isnan(cluster_cq)):
            logging.info(
                "Overplotting characteristic cluster property onto 2D "
                "histogram."
            )
            plot_config = {
                "linestyle": "dashed",
                "color": "white",
                "label": label,
                "marker": "none",
            }
            if self.log:
                cluster_cq = np.log10(cluster_cq)
            axes.plot(xs, cluster_cq, **plot_config)

        # Step 5: overplot example tracer tracks
        track_config = {
            "linestyle": "solid",
            "color": "white",
            "marker": "none",
            "alpha": 0.5,
        }
        rand_index = [0, 100, 1000, 10000]
        for idx in rand_index:
            if self.log:
                track = np.log10(particle_data[constants.MIN_SNAP:, idx])
            else:
                track = particle_data[constants.MIN_SNAP:, idx]
            axes.plot(xs, track, **track_config)

        # Step 6: legend
        if label:
            axes.legend()

        # Step 7: save figure
        volnorm_flag = "_volumenormalized" if self.volume_normalize else ""
        ident_flag = f"z{self.zoom_in:03d}_2dhist{volnorm_flag}"
        self._save_fig(
            fig, ident_flag=ident_flag, subdir=f"zoom_in_{self.zoom_in}"
        )
        logging.info("Saved 2D histogram plot to file.")
