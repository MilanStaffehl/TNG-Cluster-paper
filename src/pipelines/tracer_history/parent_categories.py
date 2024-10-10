"""
Plot plots for the parent categories of clusters.
"""
from __future__ import annotations

import dataclasses
import logging
from typing import ClassVar

import h5py
import matplotlib.cm
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np

from library import constants
from library.data_acquisition import halos_daq
from library.plotting import common
from pipelines import base


@dataclasses.dataclass
class PlotParentCategoryPlots(base.Pipeline):
    """
    Plot various plots related to the parent category of tracers.

    REMINDERS:

    - Remember to make particles unique before analyzing!
    - Remember to exclude snaps with category 255!
    """

    n_clusters: ClassVar[int] = 352
    n_snaps: ClassVar[int] = 100 - constants.MIN_SNAP

    def run(self) -> int:
        """
        Plot various parent category plots.

        :return: Exit code.
        """
        # Step 1: open archive file
        archive_file = h5py.File(self.config.cool_gas_history, "r")

        # Step 2: plot fraction of tracers in satellites
        logging.info("Plotting fraction of tracers in satellites.")
        self._plot_satellite_fraction(archive_file)

        return 0

    def _plot_satellite_fraction(self, archive_file: h5py.File) -> None:
        """
        Plot the time development of the numer of tracers in satellites.

        :param archive_file: The opened archive file.
        :return: None, plot saved to file.
        """
        # Step 1: allocate memory
        current_fraction = np.zeros((self.n_clusters, self.n_snaps))

        # Step 2: find fractions
        for zoom_id in range(self.n_clusters):
            grp = f"ZoomRegion_{zoom_id:03d}"
            parent_categories = archive_file[grp]["ParentCategory"][()]
            fractions = np.count_nonzero(parent_categories == 4, axis=1)
            fractions = fractions / parent_categories.shape[1]
            # exclude incorrect snaps
            where_faulty = np.any(parent_categories == 255, axis=1)
            fractions[where_faulty] = np.nan
            current_fraction[zoom_id] = fractions[constants.MIN_SNAP:]

        # Step 3: figure and axis setup
        fig, axes = plt.subplots(figsize=(5, 4))
        axes.set_ylabel("Fraction of tracers in satellites")
        xs = common.make_redshift_plot(axes, start=constants.MIN_SNAP)

        # Step 4: create colorscale for lines
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

        # Step 5: plot the datapoints
        plot_config = {
            "marker": "none",
            "linestyle": "solid",
            "alpha": 0.1,
        }
        for i in range(self.n_clusters):
            axes.plot(
                xs,
                current_fraction[i],
                color=colors[i],
                **plot_config,
            )
        fig.colorbar(
            matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap),
            ax=axes,
            location="right",
            label="$log_{10} M_{200c}$ at z = 0",
        )

        # Step 6: plot mean and median
        m_config = {"marker": "none", "color": "black"}
        mean = np.mean(current_fraction, axis=0)
        axes.plot(xs, mean, ls="solid", **m_config)
        median = np.median(current_fraction, axis=0)
        axes.plot(xs, median, ls="dashed", **m_config)

        # Step 7: save figure
        self._save_fig(fig, ident_flag="current_fraction")
        logging.info("Finished saving plot for satellite fraction to file.")
