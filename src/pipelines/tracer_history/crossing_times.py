"""
Pipeline to plot crossing time plots.
"""
from __future__ import annotations

import dataclasses
import enum
import logging
from typing import TYPE_CHECKING, ClassVar

import cmasher  # noqa: F401
import h5py
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np

from library import constants
from library.data_acquisition import halos_daq
from library.plotting import common, plot_radial_profiles
from library.processing import statistics
from pipelines import base

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray


class PlotType(enum.IntEnum):
    """Enum for different plot types"""
    DISTRIBUTION = 0
    DISTRIBUTION_RVIR = 1
    HIGH_Z_DISTANCE = 2
    CURRENT_DISTANCE = 3
    HIGH_Z_DISTANCE_STACKED = 4
    CURRENT_DISTANCE_STACKED = 5
    MEAN_CROSSING_TIME = 6


class CommonPlotMixin:
    """Mixin for common plotting methods"""

    def _save_distribution_plot(
        self: base.Pipeline,
        first_crossing: NDArray,
        last_crossing: NDArray,
        legend_title: str,
        ident_flag: str,
        subdir: str | None,
        color_first: str = "teal",
        color_last: str = "purple",
    ) -> None:
        """
        Plot the distribution of first and last crossing time and save it.

        :param first_crossing: Array of first crossing redshifts.
        :param last_crossing: Array of last crossing redshifts.
        :param legend_title: Title for the legend.
        :param ident_flag: Ident flag for the figure.
        :param subdir: Subdir for the figure.
        :param color_first: Color for first crossing lines.
        :param color_last: Color for last crossing lines.
        :return: None, figure is saved to file.
        """
        # create and configure a figure
        fig, axes = plt.subplots(figsize=(4, 4))
        zs = common.make_redshift_plot(axes, start=constants.MIN_SNAP)
        axes.set_xlabel("Estimated crossing redshift [z]")
        axes.set_ylabel(r"Tracer mass [$\log_{10} M_\odot$]")
        axes.set_yscale("log")

        # set all crossing times close to zero to a sentinel value (as
        # the range will be limited to 1e-3 to 8 later, and we need the
        # outliers to still fall into the last bin)
        first_crossing[first_crossing <= zs[-1]] = 1.01 * zs[-1]
        last_crossing[last_crossing <= zs[-1]] = 1.01 * zs[-1]

        # plot histograms
        logbins = np.geomspace(zs[-1], zs[0], 21)
        weights = constants.TRACER_MASS * np.ones_like(first_crossing)
        axes.hist(
            first_crossing,
            bins=logbins,
            weights=weights,
            histtype="step",
            color=color_first,
            label="First crossing",
            zorder=10,
        )
        axes.hist(
            last_crossing,
            bins=logbins,
            weights=weights,
            histtype="step",
            color=color_last,
            label="Last crossing",
            zorder=9,
        )

        # plot mean and median
        axes.axvline(
            np.nanmean(first_crossing),
            linestyle="solid",
            color=color_first,
            zorder=12,
        )
        axes.axvline(
            np.nanmedian(first_crossing),
            linestyle="dashed",
            color=color_first,
            zorder=12,
        )
        axes.axvline(
            np.nanmean(last_crossing),
            linestyle="solid",
            color=color_last,
            zorder=11,
        )
        axes.axvline(
            np.nanmedian(last_crossing),
            linestyle="dashed",
            color=color_last,
            zorder=11,
        )

        # plot label showing how many particles cross more than once
        legend = axes.legend(
            title=legend_title,
            title_fontsize="small",
            alignment="left",
            loc="best",
        )
        legend.set_zorder(20)

        self._save_fig(
            fig,
            ident_flag=ident_flag,
            subdir=subdir,
        )


@dataclasses.dataclass
class PlotCrossingTimesPlots(CommonPlotMixin, base.Pipeline):
    """
    Pipeline to plot various plots involving the first and last crossing
    time of particles that eventually end up in cool gas.
    """

    plot_types: list[int] | None = None  # what to plot

    n_clusters: ClassVar[int] = 352
    n_snaps: ClassVar[int] = 100 - constants.MIN_SNAP

    def __post_init__(self):
        super().__post_init__()
        if self.plot_types is None:
            self.plot_types = [e.value for e in PlotType]

    def run(self) -> int:
        """
        Plot crossing times plots and save them to file.

        :return: Exit code.
        """
        logging.info("Starting pipeline to plot crossing times plots.")
        # Step 0: check directories and archive
        if not self.config.cool_gas_history.exists():
            logging.fatal(
                f"Did not find cool gas archive file "
                f"{self.config.cool_gas_history}."
            )
            return 1

        # Step 1: open the archive file
        archive_file = h5py.File(self.config.cool_gas_history, "r")

        # Step 2: plot the different plot types
        if PlotType.DISTRIBUTION in self.plot_types:
            self._plot_distribution(archive_file)

        if PlotType.DISTRIBUTION_RVIR in self.plot_types:
            self._plot_distribution(archive_file, "1Rvir")

        if PlotType.HIGH_Z_DISTANCE in self.plot_types:
            self._plot_distance_dependence_overall(
                constants.MIN_SNAP, archive_file
            )

        if PlotType.CURRENT_DISTANCE in self.plot_types:
            self._plot_distance_dependence_overall(99, archive_file)

        if PlotType.HIGH_Z_DISTANCE_STACKED in self.plot_types:
            self._plot_distance_dependence_stacked_per_zoom(
                constants.MIN_SNAP, archive_file
            )

        if PlotType.CURRENT_DISTANCE_STACKED in self.plot_types:
            self._plot_distance_dependence_stacked_per_zoom(99, archive_file)

        if PlotType.MEAN_CROSSING_TIME in self.plot_types:
            self._plot_mean_crossing_time(archive_file)

        archive_file.close()
        logging.info("Done! Successfully plotted all requested plots!")
        return 0

    def _plot_distribution(
        self, archive_file: h5py.File, sfx: str = ""
    ) -> None:
        """
        Plot the distribution of crossing times.

        Method plots for every cluster the distribution of crossing times
        for the traced gas cells, with first and last crossing times in
        the same plot as two separate distributions.

        It also creates a final plot of all crossing times, similarly
        with two distributions for first and last crossing.

        :param archive_file: Opened cool gas history archive file.
        :param sfx: Suffix to the field name of the crossing time. Since
            crossing times for other spheres than the 2Rvir case can
            exist, and since these are suffixed by the radius in the
            form ``NRvir``, this allows for plotting the distirbution
            also for other crossing time definitions.
        :return: None, plots are saved to file.
        """
        logging.info(
            "Plotting distributions of crossing times for all clusters."
        )
        file_suffix = f"_{sfx}" if sfx else ""
        # Loop over all clusters and count number of total particles
        total_part_num = 0
        for zoom_id in range(self.n_clusters):
            logging.debug(f"Plotting distribution for zoom-in {zoom_id}.")
            grp = f"ZoomRegion_{zoom_id:03d}"
            first_crossing = archive_file[grp][f"FirstCrossingRedshift{sfx}"][(
            )]
            last_crossing = archive_file[grp][f"LastCrossingRedshift{sfx}"][()]
            total_part_num += last_crossing.size  # increment counter

            # find legend title
            diff = first_crossing - last_crossing
            mc = np.count_nonzero(diff) / first_crossing.size
            title = f"Multiple crossings: {mc * 100:.2f}%"

            self._save_distribution_plot(
                first_crossing,
                last_crossing,
                title,
                f"distribution_z{zoom_id:03d}{file_suffix}",
                f"individuals/zoom_in_{zoom_id}"
            )
            logging.debug(f"Saved distribution for zoom-in {zoom_id}.")

        logging.info("Plotting distribution over all particles.")
        logging.debug(f"Total number of particles: {total_part_num}.")
        # allocate memory for total particle number
        all_first_crossings = np.zeros(total_part_num)
        all_last_crossings = np.zeros(total_part_num)
        i = 0
        for zoom_id in range(self.n_clusters):
            grp = f"ZoomRegion_{zoom_id:03d}"
            first_crossing = archive_file[grp][f"FirstCrossingRedshift{sfx}"][(
            )]
            last_crossing = archive_file[grp][f"LastCrossingRedshift{sfx}"][()]
            n = first_crossing.size
            all_first_crossings[i:i + n] = first_crossing
            all_last_crossings[i:i + n] = last_crossing
            i += n

        # find legend title
        diff = all_first_crossings - all_last_crossings
        mc = np.count_nonzero(diff) / total_part_num
        title = f"Multiple crossings: {mc * 100:.2f}%"

        # plot distribution of all particles
        self._save_distribution_plot(
            all_first_crossings,
            all_last_crossings,
            title,
            f"distribution_unweighted{file_suffix}",
            None,
        )

    def _plot_distance_dependence_overall(
        self, snap: int, archive_file: h5py.File
    ) -> None:
        """
        Plot distance at given snap vs. crossing time for all particles.

        :param snap: The snapshot at which to take the distance.
        :param archive_file: The opened cool gas history archive file.
        :return: None, plots are saved to file.
        """
        logging.info(
            f"Plotting 2D histogram of distance at snap {snap} vs. crossing "
            f"times."
        )

        # Step 1: allocate memory
        n_part = archive_file["Header"].attrs["TotalPartNum"]
        crossing_times = np.zeros(n_part)
        distances = np.zeros(n_part)

        # Step 2: load the data
        i = 0
        for zoom_in in range(self.n_clusters):
            grp = f"ZoomRegion_{zoom_in:03d}"
            ct = archive_file[grp]["FirstCrossingRedshift"][()]
            n = ct.size
            crossing_times[i:i + n] = ct
            dist = archive_file[grp]["DistanceToMP"][snap, :]
            distances[i:i + n] = dist
            i += n  # increment counter

        # Step 3: adjust scales
        # set very small crossing times to smalles crossing time
        crossing_times[crossing_times <= 1e-3] = 1e-3
        crossing_times = np.log10(crossing_times)
        distances = np.log10(distances)

        # Step 4: create a 2D histogram of the two quantities
        # x_min = np.min(high_z_distance)
        x_max = np.max(distances)
        y_min = np.nanmin(crossing_times)
        y_max = np.nanmax(crossing_times)
        ranges = np.array([[1.5, x_max], [y_min, y_max]])

        hist, _, _ = np.histogram2d(
            distances,
            crossing_times,
            weights=np.ones_like(distances) * constants.TRACER_MASS,
            bins=50,
            range=ranges,
        )
        # normalize to tracer mass PER CLUSTER
        hist = hist / self.n_clusters

        # Step 5: Plot the histogram
        if snap == constants.MIN_SNAP:
            xlabel = r"Distance to cluster center at z = 8 [$\log_{10}$ ckpc]"
        elif snap == 99:
            xlabel = r"Distance to cluster center at z = 0 [$\log_{10}$ ckpc]"
        else:
            xlabel = (
                rf"Distance to cluster center at snap {snap} "
                rf"[$\log_{{10}}$ ckpc]"
            )

        # Step 6: save plot to file
        self._save_distance_dep_plot(
            hist,
            ranges,
            xlabel,
            ident_flag=f"distance_dependence_snap_{snap}",
        )

    def _plot_distance_dependence_stacked_per_zoom(
        self, snap: int, archive_file: h5py.File
    ) -> None:
        """
        Plot distance at given snap vs. crossing time.

        The difference to :meth:`_plot_distance_dependence_overall` is
        that this function first creates a 2D histogram for every cluster
        and then takes the mean of these histograms over all clusters.

        :param snap: The snapshot at which to take the distance.
        :param archive_file: The opened cool gas history archive file.
        :return: None, plots are saved to file.
        """
        logging.info(
            f"Plotting 2D histogram of distance at snap {snap} vs. crossing "
            f"times, stacked for clusters."
        )
        # Step 1: set plot config
        n_bins = 50
        dist_2dhists = np.zeros((self.n_clusters, n_bins, n_bins))
        if snap == constants.MIN_SNAP:
            xlabel = r"Distance to cluster center at z = 8 [$\log_{10}$ ckpc]"
        elif snap == 99:
            xlabel = r"Distance to cluster center at z = 0 [$\log_{10}$ ckpc]"
        else:
            xlabel = (
                rf"Distance to cluster center at snap {snap} "
                rf"[$\log_{{10}}$ ckpc]"
            )

        # Step 2: load and process the data
        ranges = np.array([[1.5, 4.8], [-3.0, np.log10(8)]])
        for zoom_in in range(self.n_clusters):
            grp = f"ZoomRegion_{zoom_in:03d}"
            crossing_times = archive_file[grp]["FirstCrossingRedshift"][()]
            distances = archive_file[grp]["DistanceToMP"][snap, :]

            # set very small crossing times to smalles crossing time
            crossing_times[crossing_times <= 1e-3] = 1e-3
            crossing_times = np.log10(crossing_times)
            distances = np.log10(distances)
            hist, _, _ = np.histogram2d(
                distances,
                crossing_times,
                weights=np.ones_like(distances) * constants.TRACER_MASS,
                bins=n_bins,
                range=ranges,
            )
            dist_2dhists[zoom_in] = hist

            self._save_distance_dep_plot(
                hist,
                ranges,
                xlabel,
                ident_flag=f"distance_dependence_snap_{snap}",
                subdir=f"individuals/zoom_in_{zoom_in}",
            )

        # Step 3: find and plot the total stacked hist
        mean_hist = np.nanmean(dist_2dhists, axis=0)
        self._save_distance_dep_plot(
            mean_hist,
            ranges,
            xlabel,
            ident_flag=f"distance_dependence_snap_{snap}_stack",
        )

    def _save_distance_dep_plot(
        self,
        hist: NDArray,
        ranges: NDArray,
        xlabel: str,
        ident_flag: str,
        subdir: str | Path | None = None,
    ) -> None:
        """
        Helper function; plots the given 2D histogram and saves it to file.

        :param hist: The 2D histogram as returned by ``np.histogram2d``.
        :param ranges: The ranges as a sequence of min and max value:
            ``[[xmin, xmax], [ymin, ymax]]``.
        :param xlabel: The label for the x-axis, i.e. the distance axis.
        :param ident_flag: The ident flag for the file.
        :param subdir: The fgures subdirectory under which to save the
            plot.
        :return: None, figures are saved to file.
        """
        # create figure
        fig, axes = plt.subplots(
            figsize=(5, 5.5),
            nrows=2,
            sharex=True,
            gridspec_kw={"height_ratios": [1, 4], "hspace": 0},
            constrained_layout=True,
        )
        with np.errstate(divide="ignore"):
            plot_radial_profiles.plot_2d_radial_profile(
                fig,
                axes[1],
                hist.transpose(),
                ranges.flatten(),
                xlabel=xlabel,
                ylabel="Estimated crossing redshift [log z]",
                scale="log",
                cbar_label=r"Tracer mass [$\log_{10} M_\odot$]",
            )

        # choose appropriate redshift labels
        z_ticks = np.array([-3, -2, -1, 0, np.log10(5)])
        z_labels = ["0", "0.01", "0.1", "1", "5"]
        axes[1].set_yticks(z_ticks, labels=z_labels)
        minor_ticks = np.concatenate(
            [
                np.arange(0.001, 0.01, 0.001),
                np.arange(0.01, 0.1, 0.01),
                np.arange(0.1, 1, 0.1),
                np.arange(1, 8, 1),
            ]
        )
        minor_ticks = np.log10(minor_ticks)
        axes[1].set_yticks(minor_ticks, minor=True)

        # add mean line
        xs = np.linspace(ranges[0, 0], ranges[0, 1], num=50)
        with np.errstate(invalid="ignore"):
            running_mean = statistics.get_2d_histogram_running_average(
                hist.transpose(), ranges[1]
            )
        axes[1].plot(xs, running_mean, color="white", linestyle="dashed")

        # add 1D distribution at top
        with np.errstate(divide="ignore"):
            hist_1d_log = np.log10(np.sum(hist, axis=1))
        axes[0].step(
            xs,
            hist_1d_log,
            where="post",
            color="blue",
        )

        # adjust secondary axes
        axes[0].set_ylabel(r"M [$\log_{10} M_\odot$]")
        axes[0].set_yticks(np.arange(5, 12.5, step=2.5))

        # save figure
        self._save_fig(fig, subdir=subdir, ident_flag=ident_flag)

    def _plot_mean_crossing_time(self, archive_file: h5py.File) -> None:
        """
        Plot the mean crossing time of each cluster vs. its mass at z = 0.

        :param archive_file: The opened archive file.
        :return: None, figure is saved to file.
        """
        logging.info("Plotting mean crossing time vs. cluster mass.")
        # allocate memory
        mean_crossing_times = np.zeros(self.n_clusters)

        # find mean crossing times
        for zoom_in in range(self.n_clusters):
            ds = f"ZoomRegion_{zoom_in:03d}/FirstCrossingRedshift"
            mean_crossing_times[zoom_in] = np.nanmean(archive_file[ds][()])

        # load cluster masses
        cluster_data = halos_daq.get_halo_properties(
            self.config.base_path,
            self.config.snap_num,
            [self.config.mass_field],
            cluster_restrict=True,
        )
        cluster_masses = np.log10(cluster_data[self.config.mass_field])

        # load cool gas fraction as color value
        base_file = (
            self.config.data_home / "mass_trends"
            / "mass_trends_clusters_base_data.npz"
        )
        with np.load(base_file) as base_data:
            cool_gas_fractions = base_data["cool_gas_fracs"][-self.n_clusters:]

        # find median crossing time in 7 mass bins
        left_bin_edges = np.array([14.2, 14.4, 14.6, 14.8, 15.0, 15.2])
        logging.info(left_bin_edges)
        medians = np.zeros_like(left_bin_edges, dtype=np.float64)
        for i, left_bin_edge in enumerate(left_bin_edges):
            right_bin_edge = left_bin_edge + 0.2
            if i == len(left_bin_edges):
                right_bin_edge += 0.01  # catch the once cluster to right
            where = np.logical_and(
                cluster_masses >= left_bin_edge,
                cluster_masses < right_bin_edge
            )
            medians[i] = np.nanmedian(mean_crossing_times[where])

        # create and configure fig and axes
        fig, axes = plt.subplots(figsize=(5, 4))
        axes.set_xlabel(r"Cluster mass at $z = 0$ [$\log_{10} M_\odot$]")
        axes.set_ylabel("Mean crossing redshift")
        axes.set_yscale("log")
        axes.set_yticks([1, 0.5, 0.2, 0.1], labels=["1", "0.5", "0.2", "0.1"])
        axes.get_xaxis().set_major_formatter(
            matplotlib.ticker.ScalarFormatter()
        )

        # plot scatterplot and median line
        common.plot_scatterplot(
            fig,
            axes,
            cluster_masses,
            mean_crossing_times,
            marker_style="D",
            color_quantity=np.log10(cool_gas_fractions),
            cmap="cmr.cosmic",
            cbar_label=r"Cool gas fraction at $z = 0$ [$\log_{10}$]",
        )
        axes.plot(
            left_bin_edges + 0.1,
            medians,
            linestyle="dashed",
            color="black",
            zorder=20,
            label="Median",
        )
        axes.legend()

        # save figure
        self._save_fig(fig, ident_flag="mean_vs_mass")


@dataclasses.dataclass
class PlotCoolingTimesPlots(CommonPlotMixin, base.Pipeline):
    """
    Pipeline to plot various plots involving the first and last time
    when particles that eventually end up in cool gas cooled below the
    temperature threshold of logT < 4.5.
    """

    plot_types: list[int] | None = None  # what to plot

    n_clusters: ClassVar[int] = 352
    n_snaps: ClassVar[int] = 100 - constants.MIN_SNAP

    def __post_init__(self):
        super().__post_init__()
        if self.plot_types is None:
            self.plot_types = [e.value for e in PlotType]

    def run(self) -> int:
        """
        Plot crossing times plots and save them to file.

        :return: Exit code.
        """
        logging.info("Starting pipeline to plot cooling times plots.")
        # Step 0: check directories and archive
        if not self.config.cool_gas_history.exists():
            logging.fatal(
                f"Did not find cool gas archive file "
                f"{self.config.cool_gas_history}."
            )
            return 1

        # Step 1: open the archive file
        archive_file = h5py.File(self.config.cool_gas_history, "r")

        # Step 2: plot the different plot types
        if PlotType.DISTRIBUTION in self.plot_types:
            self._plot_distribution(archive_file)

        # Step 3: plot mean cooling time vs. cluster mass
        if PlotType.MEAN_CROSSING_TIME in self.plot_types:
            self._plot_mean_last_cooling_time(archive_file)

        archive_file.close()
        logging.info("Done! Successfully plotted all requested plots!")
        return 0

    def _plot_distribution(self, archive_file: h5py.File) -> None:
        """
        Plot the distribution of cooling times.

        Method plots for every cluster the distribution of cooling times
        for the traced gas cells, with first and last cooling times in
        the same plot as two separate distributions.

        It also creates a final plot of all cooling times, similarly
        with two distributions for first and last crossing.

        :param archive_file: Opened cool gas history archive file.
        :return: None, plots are saved to file.
        """
        logging.info(
            "Plotting distributions of cooling times for all clusters."
        )
        # Loop over all clusters and count number of total particles
        total_part_num = 0
        for zoom_id in range(self.n_clusters):
            logging.debug(f"Plotting distribution for zoom-in {zoom_id}.")
            grp = f"ZoomRegion_{zoom_id:03d}"
            first_cooling = archive_file[grp]["FirstCoolingRedshift"][()]
            last_cooling = archive_file[grp]["LastCoolingRedshift"][()]
            total_part_num += last_cooling.size  # increment counter

            # find legend title
            n_nans = np.count_nonzero(np.isnan(first_cooling))
            title = f"Always cool: {n_nans / total_part_num * 1000:.3f}‰"

            self._save_distribution_plot(
                first_cooling,
                last_cooling,
                title,
                f"distribution_z{zoom_id:03d}",
                f"individuals/zoom_in_{zoom_id}",
                color_first="orange",
                color_last="navy",
            )
            logging.debug(f"Saved distribution for zoom-in {zoom_id}.")

        logging.info("Plotting distribution over all particles.")
        logging.debug(f"Total number of particles: {total_part_num}.")
        # allocate memory for total particle number
        all_first_coolings = np.zeros(total_part_num)
        all_last_coolings = np.zeros(total_part_num)
        i = 0
        for zoom_id in range(self.n_clusters):
            grp = f"ZoomRegion_{zoom_id:03d}"
            first_cooling = archive_file[grp]["FirstCoolingRedshift"][()]
            last_cooling = archive_file[grp]["LastCoolingRedshift"][()]
            n = first_cooling.size
            all_first_coolings[i:i + n] = first_cooling
            all_last_coolings[i:i + n] = last_cooling
            i += n

        # find legend title
        n_nans = np.count_nonzero(np.isnan(all_first_coolings))
        title = f"Always cool: {n_nans / total_part_num * 1000:.3f}‰"

        # plot distribution of all particles
        self._save_distribution_plot(
            all_first_coolings,
            all_last_coolings,
            title,
            "distribution_unweighted",
            None,
            color_first="orange",
            color_last="navy",
        )

    def _plot_mean_last_cooling_time(self, archive_file: h5py.File) -> None:
        """
        Plot the mean last cooling time of each cluster vs. its mass.

        Mass is taken at redshift zero.

        :param archive_file: The opened archive file.
        :return: None, figure is saved to file.
        """
        logging.info("Plotting mean last cooling time vs. cluster mass.")
        # allocate memory
        mean_last_cooling_times = np.zeros(self.n_clusters)

        # find mean crossing times
        for zoom_in in range(self.n_clusters):
            ds = f"ZoomRegion_{zoom_in:03d}/LastCoolingRedshift"
            mean_last_cooling_times[zoom_in] = np.nanmean(archive_file[ds][()])

        # load cluster masses
        cluster_data = halos_daq.get_halo_properties(
            self.config.base_path,
            self.config.snap_num,
            [self.config.mass_field],
            cluster_restrict=True,
        )
        cluster_masses = np.log10(cluster_data[self.config.mass_field])

        # load cool gas fraction as color value
        base_file = (
            self.config.data_home / "mass_trends"
            / "mass_trends_clusters_base_data.npz"
        )
        with np.load(base_file) as base_data:
            cool_gas_fractions = base_data["cool_gas_fracs"][-self.n_clusters:]

        # find median crossing time in 7 mass bins
        left_bin_edges = np.array([14.2, 14.4, 14.6, 14.8, 15.0, 15.2])
        medians = np.zeros_like(left_bin_edges, dtype=np.float64)
        for i, left_bin_edge in enumerate(left_bin_edges):
            right_bin_edge = left_bin_edge + 0.2
            if i == len(left_bin_edges):
                right_bin_edge += 0.01  # catch the once cluster to right
            where = np.logical_and(
                cluster_masses >= left_bin_edge,
                cluster_masses < right_bin_edge
            )
            medians[i] = np.nanmedian(mean_last_cooling_times[where])

        # create and configure fig and axes
        fig, axes = plt.subplots(figsize=(5, 4))
        axes.set_xlabel(r"Cluster mass at $z = 0$ [$\log_{10} M_\odot$]")
        axes.set_ylabel("Redshift of final cooling")
        axes.set_yscale("log")
        axes.set_yticks(
            [0.2, 0.1, 0.06, 0.05, 0.04],
            labels=["0.2", "0.1", "", "0.05", ""],
        )
        axes.get_xaxis().set_major_formatter(
            matplotlib.ticker.ScalarFormatter()
        )

        # plot scatterplot and median line
        common.plot_scatterplot(
            fig,
            axes,
            cluster_masses,
            mean_last_cooling_times,
            marker_style="D",
            color_quantity=np.log10(cool_gas_fractions),
            cmap="cmr.cosmic",
            cbar_label=r"Cool gas fraction at $z = 0$ [$\log_{10}$]",
        )
        axes.plot(
            left_bin_edges + 0.1,
            medians,
            linestyle="dashed",
            color="black",
            zorder=20,
            label="Median",
        )
        axes.legend()

        # save figure
        self._save_fig(fig, ident_flag="mean_vs_mass")
