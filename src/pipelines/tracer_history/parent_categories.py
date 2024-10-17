"""
Plot plots for the parent categories of clusters.
"""
from __future__ import annotations

import dataclasses
import logging
from typing import TYPE_CHECKING, ClassVar

import h5py
import matplotlib.cm
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np

from library import constants
from library.data_acquisition import halos_daq
from library.plotting import common
from pipelines import base

if TYPE_CHECKING:
    from numpy.typing import NDArray


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

        # Step 2: Load cluster data
        cluster_data = halos_daq.get_halo_properties(
            self.config.base_path,
            self.config.snap_num,
            [self.config.mass_field],
            cluster_restrict=True,
        )
        masses = np.log10(cluster_data[self.config.mass_field])

        # Step 2: plot fraction of tracers in satellites
        logging.info("Plotting fraction of tracers in satellites.")
        self._plot_category_fractions(4, "satellites", masses, archive_file)

        # Step 3: plot fraction of tracers in primaries
        logging.info("Plotting fraction of tracers in primaries.")
        self._plot_category_fractions(3, "primaries", masses, archive_file)

        # Step 4: plot fraction of tracers in other halos
        logging.info("Plotting fraction of tracers in other halos.")
        self._plot_category_fractions(1, "other halos", masses, archive_file)

        # Step 4: plot fraction of tracers in other halos
        logging.info("Plotting fraction of unbound tracers.")
        self._plot_category_fractions(0, "no host", masses, archive_file)

        return 0

    def _plot_category_fractions(
        self,
        category: int,
        category_name: str,
        masses: NDArray,
        archive_file: h5py.File
    ) -> None:
        """
        Plot the time development of the fraction in the given category.

        :param category: The index of the category. Can be 0, 1, 2, 3, 4.
        :param category_name: The name of the category as it should appear
            in the axes labels and file ident flag.
        :param archive_file: The opened archive file.
        :return: None, plot saved to file.
        """
        # Step 1: allocate memory
        current_fraction = np.zeros((self.n_clusters, self.n_snaps))

        # Step 2: find fractions
        for zoom_id in range(self.n_clusters):
            grp = f"ZoomRegion_{zoom_id:03d}"
            parent_categories = archive_file[grp]["ParentCategory"][()]
            fractions = np.count_nonzero(parent_categories == category, axis=1)
            fractions = fractions / parent_categories.shape[1]
            # exclude incorrect snaps
            where_faulty = np.any(parent_categories == 255, axis=1)
            fractions[where_faulty] = np.nan
            current_fraction[zoom_id] = fractions[constants.MIN_SNAP:]

        # Step 3: figure and axis setup
        fig, axes = plt.subplots(figsize=(5, 4))
        axes.set_ylabel(f"Fraction of tracers in {category_name}")
        xs = common.make_redshift_plot(axes, start=constants.MIN_SNAP)

        # Step 4: create colorscale for lines
        cmap = matplotlib.cm.get_cmap("plasma")
        norm = matplotlib.colors.Normalize(vmin=14.0, vmax=15.4)
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
        mean = np.nanmean(current_fraction, axis=0)
        axes.plot(xs, mean, ls="solid", **m_config)
        median = np.nanmedian(current_fraction, axis=0)
        axes.plot(xs, median, ls="dashed", **m_config)

        # Step 7: save figure
        category_file = category_name.replace(" ", "_")
        self._save_fig(fig, ident_flag=f"current_{category_file}_fraction")
        logging.info(
            f"Finished saving plot for {category_name} fraction to file."
        )
