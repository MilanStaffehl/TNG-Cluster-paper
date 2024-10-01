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
from pipelines import base

if TYPE_CHECKING:
    from numpy.typing import NDArray

PLOT_TYPES = [
    "distribution",
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
        axes.set_xlabel("Estimated crossing redshift [z]")
        axes.set_xscale("log")
        axes.set_ylabel("Count")
        axes.set_yscale("log")

        # plot histograms
        min_ = np.nanmin([np.nanmin(first_crossing), np.nanmin(last_crossing)])
        max_ = np.nanmax([np.nanmax(first_crossing), np.nanmax(last_crossing)])
        logbins = np.geomspace(min_, max_, 21)
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
        axes.legend(
            title=f"Multiple crossings: {mc * 100:.2f}%",
            title_fontsize="small",
            alignment="left"
        )

        self._save_fig(
            fig,
            ident_flag=ident_flag,
            subdir=subdir,
        )
