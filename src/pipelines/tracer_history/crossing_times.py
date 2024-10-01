"""
Pipeline to plot crossing time plots.
"""
from __future__ import annotations

import dataclasses
import logging
from typing import ClassVar

import h5py
import matplotlib.pyplot as plt
import numpy as np

from library import constants
from pipelines import base

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

            # create and configure a figure
            fig, axes = plt.subplots(figsize=(4, 4))
            axes.set_xlabel("Estimated crossing redshift [z]")
            axes.set_xscale("log")
            axes.set_ylabel("Count")
            axes.set_yscale("log")

            # plot histograms
            logbins = np.geomspace(
                first_crossing.min(), first_crossing.max(), 21
            )
            axes.hist(
                first_crossing,
                bins=logbins,
                histtype="step",
                color="darkslategray",
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
            axes.legend()

            # plot mean and median
            axes.axvline(
                np.mean(first_crossing),
                linestyle="solid",
                color="darkslategrey",
                zorder=12,
            )
            axes.axvline(
                np.median(first_crossing),
                linestyle="dashed",
                color="darkslategrey",
                zorder=12,
            )
            axes.axvline(
                np.mean(last_crossing),
                linestyle="solid",
                color="purple",
                zorder=11,
            )
            axes.axvline(
                np.median(last_crossing),
                linestyle="dashed",
                color="purple",
                zorder=11,
            )

            self._save_fig(
                fig,
                ident_flag=f"distribution_z{zoom_id:03d}",
                subdir=f"individuals/zoom_in_{zoom_id}",
            )
            logging.debug(f"Saved distribution for zoom-in {zoom_id}.")

        logging.debug(f"Total number of particles: {total_part_num}.")
