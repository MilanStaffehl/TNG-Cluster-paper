"""
Pipeline to plot crossing time plots.
"""
from __future__ import annotations

import dataclasses
import logging
from typing import TYPE_CHECKING, ClassVar

import h5py
import matplotlib.pyplot as plt
import numpy as np

from library import constants
from library.plotting import common, plot_radial_profiles
from library.processing import statistics
from pipelines import base

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray

PLOT_TYPES = [
    "distribution",
    "high-z-distance",
    "z-zero-distance",
    "distance-stack-high-z",
    "distance-stack-z-zero",
    "mean-crossing-time",
]


@dataclasses.dataclass
class PlotCrossingTimesPlots(base.Pipeline):
    """
    Pipeline to plot various plots involving the first and last crossing
    time of particles that eventually end up in cool gas.
    """

    plot_types: list[str] | None = None  # what to plot

    n_clusters: ClassVar[int] = 352
    n_snaps: ClassVar[int] = 100 - constants.MIN_SNAP

    def __post_init__(self):
        super().__post_init__()
        if self.plot_types is None:
            self.plot_types = PLOT_TYPES

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
        if "distribution" in self.plot_types:
            self._plot_distribution(archive_file)

        if "high-z-distance" in self.plot_types:
            self._plot_distance_dependence_overall(
                constants.MIN_SNAP, archive_file
            )

        if "z-zero-distance" in self.plot_types:
            self._plot_distance_dependence_overall(99, archive_file)

        if "distance-stack-high-z" in self.plot_types:
            self._plot_distance_dependence_stacked_per_zoom(
                constants.MIN_SNAP, archive_file
            )

        if "distance-stack-z-zero" in self.plot_types:
            self._plot_distance_dependence_stacked_per_zoom(99, archive_file)

        if "mean-crossing-time" in self.plot_types:
            self._plot_mean_crossing_time(archive_file)

        archive_file.close()
        logging.info("Done! Successfully plotted all requested plots!")
        return 0

    def _plot_distribution(self, archive_file: h5py.File) -> None:
        """
        Plot the distribution of crossing times.

        Method plots for every cluster the distribution of crossing times
        for the traced gas cells, with first and last crossing times in
        the same plot as two separate distributions.

        It also creates a final plot of all crossing times, similarly
        with two distributions for first and last crossing.

        :param archive_file: Opened cool gas history archive file.
        :return: None, plots are saved to file.
        """
        logging.info(
            "Plotting distributions of crossing times for all clusters."
        )
        # Loop over all clusters and count number of total particles
        total_part_num = 0
        for zoom_id in range(self.n_clusters):
            logging.debug(f"Plotting distribution for zoom-in {zoom_id}.")
            grp = f"ZoomRegion_{zoom_id:03d}"
            first_crossing = archive_file[grp]["FirstCrossingRedshift"][()]
            last_crossing = archive_file[grp]["LastCrossingRedshift"][()]
            total_part_num += last_crossing.size  # increment counter
            self._save_distribution_plot(
                first_crossing,
                last_crossing,
                f"distribution_z{zoom_id:03d}",
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
            first_crossing = archive_file[grp]["FirstCrossingRedshift"][()]
            last_crossing = archive_file[grp]["LastCrossingRedshift"][()]
            n = first_crossing.size
            all_first_crossings[i:i + n] = first_crossing
            all_last_crossings[i:i + n] = last_crossing
            i += n

        # plot distribution of all particles
        self._save_distribution_plot(
            all_first_crossings,
            all_last_crossings,
            "distribution_unweighted",
            None,
        )

    def _save_distribution_plot(
        self,
        first_crossing: NDArray,
        last_crossing: NDArray,
        ident_flag: str,
        subdir: str | None,
    ) -> None:
        """
        Plot the distribution of first and last crossing time and save it.

        :param first_crossing: Array of first crossing redshifts.
        :param last_crossing: Array of last crossing redshifts.
        :param ident_flag: Ident flag for the figure.
        :param subdir: Subdir for the figure.
        :return: None, figure is saved to file.
        """
        # create and configure a figure
        fig, axes = plt.subplots(figsize=(4, 4))
        zs = common.make_redshift_plot(axes, start=constants.MIN_SNAP)
        axes.set_xlabel("Estimated crossing redshift [z]")
        axes.set_ylabel("Count")
        axes.set_yscale("log")

        # set all crossing times close to zero to a sentinel value (as
        # the range will be limited to 1e-3 to 8 later, and we need the
        # outliers to still fall into the last bin)
        first_crossing[first_crossing <= zs[-1]] = 1.01 * zs[-1]
        last_crossing[last_crossing <= zs[-1]] = 1.01 * zs[-1]

        # plot histograms
        logbins = np.geomspace(zs[-1], zs[0], 21)
        axes.hist(
            first_crossing,
            bins=logbins,
            histtype="step",
            color="teal",
            label="First crossing",
            zorder=10,
        )
        axes.hist(
            last_crossing,
            bins=logbins,
            histtype="step",
            color="purple",
            label="Last crossing",
            zorder=9,
        )

        # plot mean and median
        axes.axvline(
            np.nanmean(first_crossing),
            linestyle="solid",
            color="teal",
            zorder=12,
        )
        axes.axvline(
            np.nanmedian(first_crossing),
            linestyle="dashed",
            color="teal",
            zorder=12,
        )
        axes.axvline(
            np.nanmean(last_crossing),
            linestyle="solid",
            color="purple",
            zorder=11,
        )
        axes.axvline(
            np.nanmedian(last_crossing),
            linestyle="dashed",
            color="purple",
            zorder=11,
        )

        # plot label showing how many particles cross more than once
        diff = first_crossing - last_crossing
        mc = np.count_nonzero(diff) / first_crossing.size
        legend = axes.legend(
            title=f"Multiple crossings: {mc * 100:.2f}%",
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
            "Plotting 2D histogram of distance at high z vs. crossing times."
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
            bins=50,
            range=ranges,
        )

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
            "Plotting 2D histogram of distance at high z vs. crossing times,"
            "stacked for clusters."
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
        z_labels = ["0.001", "0.01", "0.1", "1", "5"]
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
        running_mean = statistics.get_2d_histogram_running_average(
            hist.transpose(), ranges[1]
        )
        axes[1].plot(xs, running_mean, color="white", linestyle="dashed")

        # add 1D distribution at top
        hist_1d = np.sum(hist, axis=1)
        axes[0].step(
            xs,
            np.log10(hist_1d),
            where="post",
            color="blue",
        )

        # adjust secondary axes
        axes[0].set_ylabel("Count [log]")

        # save figure
        self._save_fig(fig, subdir=subdir, ident_flag=ident_flag)

    def _plot_mean_crossing_time(self, archive_file: h5py.File) -> None:
        pass
