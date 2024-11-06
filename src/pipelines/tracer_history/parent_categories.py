"""
Plot plots for the parent categories of clusters.
"""
from __future__ import annotations

import dataclasses
import enum
import logging
from typing import TYPE_CHECKING, ClassVar

import h5py
import matplotlib.cm
import matplotlib.pyplot as plt
import numpy as np

from library import constants
from library.data_acquisition import halos_daq
from library.plotting import common
from pipelines import base

if TYPE_CHECKING:
    from numpy.typing import NDArray


class PlotType(enum.IntEnum):
    FRACTION_PLOT = 0
    FRACTION_PLOT_RVIR = 1
    CUMULATIVE_FRACTION_PLOT = 2
    INDIVIDUAL_FRACTION_PLOT = 3
    INDIVIDUAL_FRACTION_RVIR = 4
    OTHER_HALO_FRACTION_PLOT = 5


@dataclasses.dataclass
class PlotParentCategoryPlots(base.Pipeline):
    """
    Plot various plots related to the parent category of tracers.

    REMINDERS:

    - Remember to make particles unique before analyzing!
    - Remember to exclude snaps with category 255!
    """

    plot_types: list[int] | None = None

    n_clusters: ClassVar[int] = 352
    n_snaps: ClassVar[int] = 100 - constants.MIN_SNAP

    def __post_init__(self):
        super().__post_init__()
        if self.plot_types is None:
            self.plot_types = [e.value for e in PlotType]

    def run(self) -> int:
        """
        Plot various parent category plots.

        :return: Exit code.
        """
        logging.info(
            f"Starting pipeline for plot types "
            f"{', '.join([PlotType(e).name for e in self.plot_types])}"
        )

        # Step 1: open archive file
        archive_file = h5py.File(self.config.cool_gas_history, "r")

        # Step 2: Load cluster data
        cluster_data = halos_daq.get_halo_properties(
            self.config.base_path,
            self.config.snap_num,
            [self.config.mass_field, self.config.radius_field],
            cluster_restrict=True,
        )
        masses = np.log10(cluster_data[self.config.mass_field])
        radii = cluster_data[self.config.radius_field]

        category_map = {
            "no host": 0,
            "other halos": 1,
            "inner fuzz": 2,
            "primaries": 3,
            "satellites": 4,
        }

        # Step 3: plot fraction of tracers at current redshift
        if PlotType.FRACTION_PLOT in self.plot_types:
            for name, num in category_map.items():
                logging.info(f"Plotting fraction of tracers in {name}.")
                self._plot_category_fractions(num, name, masses, archive_file)
            # additional plots not containing a single category
            self._plot_primary_halo_fraction(masses, archive_file)

        # Step 4: plot fraction of tracers within R_vir
        if PlotType.FRACTION_PLOT_RVIR in self.plot_types:
            for name, num in category_map.items():
                logging.info(
                    f"Plotting fraction of tracers within R_vir in {name}."
                )
                self._plot_category_fractions(
                    num, name, masses, archive_file, radii
                )
            # additional plots not containing a single category
            self._plot_primary_halo_fraction(masses, archive_file, radii)

        # Step 5: plot cumulative fraction of tracers in satellites
        if PlotType.CUMULATIVE_FRACTION_PLOT in self.plot_types:
            logging.info(
                "Plotting cumulative fraction of tracers in satellites."
            )
            self._plot_cumulative_satellite_fraction(masses, archive_file)

        # Step 6: plot fractions for individual clusters
        if PlotType.INDIVIDUAL_FRACTION_PLOT in self.plot_types:
            logging.info(
                "Plotting fraction of tracers by parent category for all "
                "clusters individually."
            )
            self._plot_individual_fractions(archive_file, None)

        # step 7: plot fractions for individuals, only within virial radius
        if PlotType.INDIVIDUAL_FRACTION_RVIR in self.plot_types:
            logging.info(
                "Plotting fraction of tracers by parent category for all "
                "clusters individually, limited to virial radius at z = 0."
            )
            self._plot_individual_fractions(archive_file, radii)

        # Step 8: plot about how the "other halo" category is comprised
        if PlotType.OTHER_HALO_FRACTION_PLOT in self.plot_types:
            logging.info(
                "Plotting fraction of \"other halo\" tracers bound to a "
                "subhalo."
            )
            self._plot_other_halo_fraction(masses, archive_file)

        return 0

    def _plot_category_fractions(
        self,
        category: int,
        category_name: str,
        masses: NDArray,
        archive_file: h5py.File,
        virial_radii: NDArray | None = None,
    ) -> None:
        """
        Plot the time development of the fraction in the given category.

        :param category: The index of the category. Can be 0, 1, 2, 3, 4.
        :param category_name: The name of the category as it should appear
            in the axes labels and file ident flag.
        :param archive_file: The opened archive file.
        :param virial_radii: Either an array of radii of shape (352, ),
            giving a (virial) radius for every cluster in ckpc, or None.
            When given a list of radii, only particles that end up within
            that radius at z = 0 are considered.
        :return: None, plot saved to file.
        """
        # Step 1: allocate memory
        current_fraction = np.zeros((self.n_clusters, self.n_snaps))

        # Step 2: find fractions
        for zoom_id in range(self.n_clusters):
            grp = f"ZoomRegion_{zoom_id:03d}"
            parent_categories = archive_file[grp]["ParentCategory"][()]
            if virial_radii is not None:
                dataset = f"ZoomRegion_{zoom_id:03d}/DistanceToMP"
                distances_z0 = archive_file[dataset][99, :]
                mask = distances_z0 <= virial_radii[zoom_id]
                parent_categories = parent_categories[:, mask]
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

        # Step 4: plot lines and mean/median
        common.plot_cluster_line_plot(fig, axes, xs, current_fraction, masses)

        # Step 5: save figure
        category_file = category_name.replace(" ", "_")
        sfx = "" if virial_radii is None else "_within_1Rvir"
        self._save_fig(
            fig, ident_flag=f"current_{category_file}_fraction{sfx}"
        )
        logging.info(
            f"Finished saving plot for {category_name} fraction to file."
        )

    def _plot_primary_halo_fraction(
        self,
        masses: NDArray,
        archive_file: h5py.File,
        virial_radii: NDArray | None = None,
    ) -> None:
        """
        Plot the time development of the fraction in the primary halo.

        :param archive_file: The opened archive file.
        :param virial_radii: Either an array of radii of shape (352, ),
            giving a (virial) radius for every cluster in ckpc, or None.
            When given a list of radii, only particles that end up within
            that radius at z = 0 are considered.
        :return: None, plot saved to file.
        """
        # Step 1: allocate memory
        current_fraction = np.zeros((self.n_clusters, self.n_snaps))

        # Step 2: find fractions
        for zoom_id in range(self.n_clusters):
            grp = f"ZoomRegion_{zoom_id:03d}"
            parent_categories = archive_file[grp]["ParentCategory"][()]
            if virial_radii is not None:
                dataset = f"ZoomRegion_{zoom_id:03d}/DistanceToMP"
                distances_z0 = archive_file[dataset][99, :]
                mask = distances_z0 <= virial_radii[zoom_id]
                parent_categories = parent_categories[:, mask]
            in_halo = np.logical_or(
                (parent_categories == 2 | parent_categories == 3),
                parent_categories == 4
            )
            fractions = np.count_nonzero(in_halo, axis=1)
            fractions = fractions / parent_categories.shape[1]
            # exclude incorrect snaps
            where_faulty = np.any(parent_categories == 255, axis=1)
            fractions[where_faulty] = np.nan
            current_fraction[zoom_id] = fractions[constants.MIN_SNAP:]

        # Step 3: figure and axis setup
        fig, axes = plt.subplots(figsize=(5, 4))
        axes.set_ylabel("Fraction of tracers in primary halo")
        xs = common.make_redshift_plot(axes, start=constants.MIN_SNAP)

        # Step 4: plot lines and mean/median
        common.plot_cluster_line_plot(fig, axes, xs, current_fraction, masses)

        # Step 5: save figure
        sfx = "" if virial_radii is None else "_within_1Rvir"
        self._save_fig(fig, ident_flag=f"current_primary_halo_fraction{sfx}")
        logging.info("Finished saving plot for primary halo fraction to file.")

    def _plot_cumulative_satellite_fraction(
        self, masses: NDArray, archive_file: h5py.File
    ) -> None:
        """
        Plot the fraction of tracers that have ever been in satellites.

        :param masses: The cluster masses in log scale.
        :param archive_file: The opened cool gas archive file.
        :return: None, figure is saved to file.
        """
        # Step 1: find fraction of tracers that have ever been in a satellite
        cum_fractions = np.empty((self.n_clusters, self.n_snaps))
        for zoom_in in range(self.n_clusters):
            # Step 2.1: get parent categories
            grp = f"ZoomRegion_{zoom_in:03d}"
            categories = archive_file[grp]["ParentCategory"][()]
            n_particles = categories.shape[1]

            # Step 2.2: find the first time that every particle is in a
            # satellite using argmax on a boolean array
            first_snap_in_satellite = np.argmax(categories == 4, axis=0)
            # Set all zeros to a high value, so we don't accidentally count
            # those entries that are never in a satellite, which argmax sets
            # to zero, as "has been in a satellite for the first time at snap
            # zero". We can get away with this, since we KNOW that not a single
            # particle can be category 4 at snapshot zero: everything before
            # MIN_SNAP is set to category 255, always and everywhere.
            first_snap_in_satellite[first_snap_in_satellite == 0] = 101
            assert np.all(first_snap_in_satellite >= constants.MIN_SNAP)
            for snap_num in range(constants.MIN_SNAP, 100):
                i = snap_num - constants.MIN_SNAP
                n_ever_in_satellite = np.count_nonzero(
                    first_snap_in_satellite <= snap_num
                )
                cum_fractions[zoom_in][i] = n_ever_in_satellite / n_particles

        # Step 2: create a figure
        fig, axes = plt.subplots(figsize=(5, 4))
        xs = common.make_redshift_plot(axes, start=constants.MIN_SNAP)
        axes.set_ylabel("Cum. fraction of tracers in satellites")

        # Step 3: plot the fraction
        common.plot_cluster_line_plot(fig, axes, xs, cum_fractions, masses)

        # Step 4: save plot to file
        self._save_fig(fig, ident_flag="cumulative_satellite_frac")
        logging.info(
            "Finished saving plot for cumulative satellite fraction to file."
        )

    def _plot_individual_fractions(
        self,
        archive_file: h5py.File,
        virial_radii: NDArray | None = None
    ) -> None:
        """
        Plot the fraction of every category vs redshift per cluster.

        Method creates a plot for every cluster, showing five lines
        where each line shows the fraction of tracers belonging to a
        parent category. Plot is saved to file.

        Optionally, the plot can be limited to only take into account
        particles that end up within aspecific radius, usually the
        virial radius. To achieve this, the ``virial_radii`` argument
        must be given, defining this radius for every cluster.

        :param virial_radii: Either an array of virial radii in ckpc for
            all clusters or None. If given an array of radii, the plots
            will be limited to only include particles that end up within
            this radius at z = 0.
        :param archive_file: The opened cool gas archive file.
        :return: None, figure is saved to file.
        """
        # color setup
        cmap = matplotlib.cm.get_cmap("turbo_r")
        norm = matplotlib.colors.Normalize(vmin=0, vmax=4.2)
        colors = cmap(norm(np.arange(0, 5, step=1)))
        for zoom_in in range(self.n_clusters):
            logging.debug(
                f"Plotting parent fraction plot for zoom-in {zoom_in}."
            )
            # create figure
            fig, axes = plt.subplots(figsize=(4, 4))
            axes.set_ylim([-0.1, 1.1])
            axes.set_ylabel("Fraction of tracers")
            xs = common.make_redshift_plot(axes, start=constants.MIN_SNAP)

            # load parent categories
            dataset = f"ZoomRegion_{zoom_in:03d}/ParentCategory"
            parent_categories = archive_file[dataset][constants.MIN_SNAP:, :]
            if virial_radii is not None:
                dataset = f"ZoomRegion_{zoom_in:03d}/DistanceToMP"
                distances_z0 = archive_file[dataset][99, :]
                mask = distances_z0 <= virial_radii[zoom_in]
                parent_categories = parent_categories[:, mask]
            n_particles = parent_categories.shape[1]

            # plot lines for categories
            categories = [
                "unbound",
                "other halo",
                "inner fuzz",
                "central",
                "satellite",
            ]
            for i, category in enumerate(categories):
                n_current = np.count_nonzero(parent_categories == i, axis=1)
                fraction = n_current / n_particles
                # exclude incorrect snaps
                where_faulty = np.any(parent_categories == 255, axis=1)
                fraction[where_faulty] = np.nan
                # plot
                plot_config = {
                    "color": colors[i],
                    "linestyle": "solid",
                    "marker": "none",
                    "label": category,
                }
                axes.plot(xs, fraction, **plot_config)

            axes.legend(fontsize="small")

            # save figure to file
            sfx = "" if virial_radii is None else "_within_Rvir"
            self._save_fig(
                fig,
                ident_flag=f"parent_category_fractions_z{zoom_in:03d}{sfx}",
                subdir=f"individuals/zoom_in_{zoom_in:03d}",
            )
        logging.info(
            "Finished saving plot for parent fractions for individual "
            "clusters to file."
        )

    def _plot_other_halo_fraction(
        self, masses: NDArray, archive_file: h5py.File
    ) -> None:
        """
        Plot the fraction of tracer in other halos that are in a subhalo.

        Method plots for every cluster a line that shows the fraction of
        particles that are in category 2 "other halos", and simultaneously
        also are bound to a subhalo, i.e. are NOT part of the inner fuzz
        of other halos.

        :param masses: Array of cluster masses.
        :param archive_file: Opened cool gas archive file.
        :return: None, figure is saved to file.
        """
        # Step 1: allocate memory
        bound_fraction = np.zeros((self.n_clusters, self.n_snaps))

        # Step 2: find fractions
        for zoom_id in range(self.n_clusters):
            grp = f"ZoomRegion_{zoom_id:03d}"
            parent_categories = archive_file[grp]["ParentCategory"][()]
            parent_subhalos = archive_file[grp]["ParentSubhaloIndex"][()]
            in_other_halo = parent_categories == 1
            in_subhalo = parent_subhalos != -1
            fractions = np.count_nonzero(
                np.logical_and(in_other_halo, in_subhalo), axis=1
            )
            fractions = fractions / np.count_nonzero(in_other_halo, axis=1)
            # exclude incorrect snaps
            where_faulty = np.any(parent_categories == 255, axis=1)
            fractions[where_faulty] = np.nan
            bound_fraction[zoom_id] = fractions[constants.MIN_SNAP:]

        # Step 3: figure and axis setup
        fig, axes = plt.subplots(figsize=(5, 4))
        axes.set_ylabel("Fraction of bound tracers in other halos")
        xs = common.make_redshift_plot(axes, start=constants.MIN_SNAP)

        # Step 4: plot lines and mean/median
        common.plot_cluster_line_plot(fig, axes, xs, bound_fraction, masses)

        # Step 5: save figure
        self._save_fig(fig, ident_flag="other_halo_bound_fraction")
        logging.info(
            "Finished saving plot for fraction of other halo tracers bound "
            "to satellites to file."
        )
