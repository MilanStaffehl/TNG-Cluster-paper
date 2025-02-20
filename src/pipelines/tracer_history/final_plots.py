"""
Pipeline for some final plots, utilizing multiple fields of the archive.
"""
from __future__ import annotations

import dataclasses
import logging
from typing import TYPE_CHECKING, ClassVar

import h5py
import matplotlib.cm
import matplotlib.colors
import matplotlib.lines
import matplotlib.patches
import matplotlib.pyplot as plt
import numpy as np

from library import constants
from library.data_acquisition import halos_daq, sublink_daq
from library.plotting import common
from pipelines import base

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclasses.dataclass
class ParentCategoryBarPlotPipeline(base.Pipeline):
    """
    Plot the parent category fraction of particles at crossing.
    """

    fractions: bool = False

    n_clusters: ClassVar[int] = 352

    def run(self) -> int:
        """
        Plot the fraction of parents at crossing as a bar chart.

        :return: Exit code
        """
        logging.info("Starting pipeline for bar plot of parent categories.")

        # Step 1: open archive file
        archive = h5py.File(self.config.cool_gas_history, "r")

        # Step 2: allocate memory
        total_n_part = archive["Header"].attrs["TotalPartNum"]
        parent_category_2Rvir = np.zeros(total_n_part)
        parent_category_1Rvir = np.zeros(total_n_part)
        parent_category_z0 = np.zeros(total_n_part)

        # Step 3: fill with values
        seen = 0
        logging.info(
            "Loading parent categories and crossing times for all clusters."
        )
        for zoom_in in range(self.n_clusters):
            logging.debug(
                f"Loading parent category and crossing times for zoom-in {zoom_in}."
            )
            grp = f"ZoomRegion_{zoom_in:03d}"
            parent_category = archive[grp]["ParentCategory"][()]
            crossing_snap_2 = archive[grp]["FirstCrossingSnapshot"][()]
            crossing_snap_1 = archive[grp]["FirstCrossingSnapshot1Rvir"][()]
            n = crossing_snap_2.size

            pc_at_crossing_2 = np.array(
                [parent_category[crossing_snap_2[i], i] for i in range(n)]
            )
            pc_at_crossing_1 = np.array(
                [parent_category[crossing_snap_1[i], i] for i in range(n)]
            )
            # remove particles that never cross
            pc_at_crossing_2[crossing_snap_2 == -1] = 255
            pc_at_crossing_1[crossing_snap_1 == -1] = 255

            # assign to allocated memory
            parent_category_2Rvir[seen:seen + n] = pc_at_crossing_2
            parent_category_1Rvir[seen:seen + n] = pc_at_crossing_1
            parent_category_z0[seen:seen + n] = parent_category[99]

            # increment counter of particles
            seen += n
        archive.close()

        # Step 4: count the number of parents that appear
        logging.info("Calculating counts per parent category.")
        _, pc2Rvir_c = np.unique(parent_category_2Rvir, return_counts=True)
        _, pc1Rvir_c = np.unique(parent_category_1Rvir, return_counts=True)
        _, pcz0_c = np.unique(parent_category_z0, return_counts=True)
        # sum together particles in inner fuzz and in primary halo
        pc2Rvir_counts = np.zeros(5, dtype=np.uint32)
        pc2Rvir_counts[0:2] = pc2Rvir_c[0:2]
        pc2Rvir_counts[2] = pc2Rvir_c[2] + pc2Rvir_c[3]
        pc2Rvir_counts[3:] = pc2Rvir_c[4:]
        pc1Rvir_counts = np.zeros(5, dtype=np.uint32)
        pc1Rvir_counts[0:2] = pc1Rvir_c[0:2]
        pc1Rvir_counts[2] = pc1Rvir_c[2] + pc1Rvir_c[3]
        pc1Rvir_counts[3:] = pc1Rvir_c[4:]
        pcz0_counts = np.zeros(5, dtype=np.uint32)
        pcz0_counts[0:2] = pcz0_c[0:2]
        pcz0_counts[2] = pcz0_c[2] + pcz0_c[3]
        pcz0_counts[3] = pcz0_c[4]
        # normalize to a fraction if required, tracer mass otherwise
        if self.fractions:
            pc2Rvir_counts = pc2Rvir_counts / total_n_part
            pc1Rvir_counts = pc1Rvir_counts / total_n_part
            pcz0_counts = pcz0_counts / total_n_part
        else:
            n = 352
            pc2Rvir_counts = pc2Rvir_counts * constants.TRACER_MASS / n
            pc1Rvir_counts = pc1Rvir_counts * constants.TRACER_MASS / n
            pcz0_counts = pcz0_counts * constants.TRACER_MASS / n

        # Step 5: set up figure
        logging.info("Start plotting bar chart.")
        fig, axes = plt.subplots(figsize=(4, 4))
        # axes.set_xlabel("Category")
        if self.fractions:
            axes.set_ylabel("Fraction")
        else:
            axes.set_ylabel(r"Mean tracer mass [$\log_{10} M_\odot$]")
        axes.set_yscale("log")
        axes.set_xticks(
            [0, 1, 2, 3, 4],
            labels=[
                "Unbound",
                "Other\nhalo",
                "Primary\nhalo",
                "Satellite",
                "Never\ncrossed",
            ],
            rotation=25,
        )

        # Step 6: plot the data
        midpoints = np.arange(0, 5, step=1)
        offset = 0.3
        cmap = matplotlib.cm.get_cmap("turbo_r")
        norm = matplotlib.colors.Normalize(vmin=0, vmax=4.2)
        colors = cmap(norm(np.arange(0, 5, step=1)))

        bar_config = {
            "edgecolor": colors,
            "color": "none",
            "width": 0.3,
        }
        axes.bar(midpoints, pc1Rvir_counts, hatch="......", **bar_config)
        axes.bar(
            midpoints - offset, pc2Rvir_counts, hatch=r"\\\\\\", **bar_config
        )
        axes.bar(midpoints + offset, pcz0_counts, color=colors, width=0.3)

        # Step 7: add a legend
        patch_config = {
            "edgecolor": "grey",
            "linestyle": "solid",
        }
        handles = [
            matplotlib.patches.Patch(
                **patch_config,
                facecolor="none",
                hatch=r"\\\\\\",
                label=r"At crossing $2 R_{200c}$",
            ),
            matplotlib.patches.Patch(
                **patch_config,
                facecolor="none",
                hatch="......",
                label=r"At crossing $1 R_{200c}$",
            ),
            matplotlib.patches.Patch(
                **patch_config,
                facecolor="grey",
                label=r"At redshift $z = 0$",
            )
        ]
        axes.legend(
            handles=handles,
            fontsize="x-small",
            ncols=3,
            bbox_to_anchor=(1.1, 1.1)
        )

        # Step 8: save figure
        ident_flag = "bar_graph"
        if self.fractions:
            ident_flag += "_fractions"
        self._save_fig(fig, ident_flag=ident_flag)
        logging.info(
            "Finshed plotting bar chart for parent category fractions!"
        )

        return 0


@dataclasses.dataclass
class PlotTracerFractionInRadius(base.Pipeline):
    """Plot the fraction of tracers in 1 and 2 R_vir."""

    n_clusters: ClassVar[int] = 352
    n_snaps: ClassVar[int] = 100 - constants.MIN_SNAP

    def run(self) -> int:
        """
        Plot the fraction of tracers within 1 and 2 Rvir.

        :return: Exit code.
        """
        logging.info(
            "Starting pipeline to plot tracer fraction within 1 and 2 Rvir."
        )
        # Step 1: open archive
        archive = h5py.File(self.config.cool_gas_history, "r")

        # Step 2: allocate memory
        fractions_1Rvir = np.zeros((self.n_clusters, self.n_snaps))
        fractions_2Rvir = np.zeros_like(fractions_1Rvir)

        # Step 3: load primaries and masses
        cluster_data = halos_daq.get_halo_properties(
            self.config.base_path,
            self.config.snap_num,
            fields=["GroupFirstSub", self.config.mass_field],
            cluster_restrict=True,
        )
        cluster_masses = np.log10(cluster_data[self.config.mass_field])
        primaries = cluster_data["GroupFirstSub"]

        # Step 4: Find fractions
        logging.info("Starting to calculate fractions. Can take a while...")
        for zoom_in in range(self.n_clusters):
            logging.debug(f"Finding fractions for zoom-in {zoom_in:03d}.")
            # Step 4.1: load virial radius
            mpb = sublink_daq.get_mpb_properties(
                self.config.base_path,
                self.config.snap_num,
                primaries[zoom_in],
                fields=[self.config.radius_field],
                start_snap=constants.MIN_SNAP,
                interpolate=True,
            )
            virial_radii = mpb[self.config.radius_field]

            # Step 4.2: load distances
            grp = f"ZoomRegion_{zoom_in:03d}"
            distances = archive[grp]["DistanceToMP"][constants.MIN_SNAP:, :]
            n_particles = distances.shape[1]

            # Step 4.3: find fraction
            for i in range(self.n_snaps):
                snap_num = i + constants.MIN_SNAP
                vr = virial_radii[mpb["SnapNum"] == snap_num]
                within_1rvir = np.count_nonzero(distances[i] <= vr)
                fractions_1Rvir[zoom_in, i] = within_1rvir / n_particles
                within_2rvir = np.count_nonzero(distances[i] <= 2 * vr)
                fractions_2Rvir[zoom_in, i] = within_2rvir / n_particles

        # Step 5: plot both fractions and save to file
        mapping = {
            "within_1Rvir": fractions_1Rvir,
            "within_2Rvir": fractions_2Rvir,
        }
        for ident_flag, fractions in mapping.items():
            logging.info(f"Plotting fraction {ident_flag.replace('_', ' ')}.")
            fig, axes = plt.subplots(figsize=(5, 4))
            threshold = r"$R_{200c}$"
            if ident_flag == "within_2Rvir":
                threshold = threshold.replace("$R", "$2R")
            axes.set_ylabel(f"Tracer fraction within {threshold}")
            xs = common.make_redshift_plot(axes, start=constants.MIN_SNAP)
            common.plot_cluster_line_plot(
                fig, axes, xs, fractions, cluster_masses
            )
            self._save_fig(fig, ident_flag=ident_flag)

        logging.info(
            "Successfully plotted fraction of tracers within 1 and 2 Rvir."
        )
        return 0


@dataclasses.dataclass
class ParentCategoryWithClusterMass(base.Pipeline):
    """
    Plot the parent category gas mass with cluster mass.
    """

    combine_panels: bool = False

    n_clusters: ClassVar[int] = 352

    def run(self) -> int:
        """
        Plot the mass of tracers in each category vs. cluster mass.

        :return: Exit code.
        """
        logging.info(
            "Starting pipeline to plot parent gas mass vs cluster mass."
        )

        # Step 1: Open archive
        archive = h5py.File(self.config.cool_gas_history)

        # Step 2: allocate memory
        pc_mass_1Rvir = np.zeros((self.n_clusters, 5))
        pc_mass_2Rvir = np.zeros_like(pc_mass_1Rvir)
        pc_mass_z0 = np.zeros_like(pc_mass_1Rvir)

        # Step 3: load cluster masses
        masses = halos_daq.get_halo_properties(
            self.config.base_path,
            self.config.snap_num,
            fields=[self.config.mass_field],
            cluster_restrict=True,
        )[self.config.mass_field]

        # Step 4: find and save parent fractions
        logging.info(
            "Loading parent categories at crossing times for all clusters. "
            "May take a while..."
        )
        for zoom_in in range(self.n_clusters):
            logging.debug(
                f"Loading parent category and crossing times for zoom-in {zoom_in}."
            )
            grp = f"ZoomRegion_{zoom_in:03d}"
            parent_category = archive[grp]["ParentCategory"][()]
            crossing_snap_2 = archive[grp]["FirstCrossingSnapshot"][()]
            crossing_snap_1 = archive[grp]["FirstCrossingSnapshot1Rvir"][()]
            n = crossing_snap_2.size

            pc_at_crossing_2 = np.array(
                [parent_category[crossing_snap_2[i], i] for i in range(n)]
            )
            pc_at_crossing_1 = np.array(
                [parent_category[crossing_snap_1[i], i] for i in range(n)]
            )
            # remove particles that never cross
            pc_at_crossing_2[crossing_snap_2 == -1] = 255
            pc_at_crossing_1[crossing_snap_1 == -1] = 255

            # count parent category
            pc_1Rvir, ct_1Rvir = np.unique(pc_at_crossing_1, return_counts=True)
            pc_2Rvir, ct_2Rvir = np.unique(pc_at_crossing_2, return_counts=True)
            pc_z0, ct_z0 = np.unique(parent_category[99], return_counts=True)

            # assign results to allocated memory
            pc_mass_1Rvir[zoom_in] = self._sum_masses(pc_1Rvir, ct_1Rvir)
            pc_mass_2Rvir[zoom_in] = self._sum_masses(pc_2Rvir, ct_2Rvir)
            pc_mass_z0[zoom_in] = self._sum_masses(pc_z0, ct_z0)
        archive.close()

        # Step 5: plot results
        logging.info("Start plotting mass plots.")
        parent_categories = [
            "Unbound",
            "Other halo",
            "Primary halo",
            "Satellite",
            "Never crossed",
        ]
        plot_categories = {
            "At crossing $2 R_{200c}$": pc_mass_2Rvir,
            "At crossing $1 R_{200c}$": pc_mass_1Rvir,
            "At redshift $z = 0$": pc_mass_z0,
        }
        if self.combine_panels:
            self._plot_combined_panels(
                masses, plot_categories, parent_categories
            )
        else:
            self._plot_individual_panels(
                masses, plot_categories, parent_categories
            )

        logging.info(
            "Done! Successfully plotted mass dependence of parent category!"
        )
        return 0

    def _plot_individual_panels(
        self,
        masses: NDArray,
        plot_categories: dict[str:NDArray],
        parent_categories: list[str],
    ) -> None:
        """
        Plot the mass dependence as three separate figures.

        :param masses: List of cluster masses in solar masses (not log!).
            Shape (N, ).
        :param plot_categories: A dictionary, mapping the title of each
            panel to the corresponding array of mean tracer mass per
            category in each halo of shape (N, 5).
        :param parent_categories: A list of names for each of the five
            parent categories, used to label the data points.
        :return: None, figures are saved to file.
        """
        ident_flags = {
            "At crossing $2 R_{200c}$": "at_crossing_2Rvir",
            "At crossing $1 R_{200c}$": "at_crossing_1Rvir",
            "At redshift $z = 0$": "at_redshift_zero",
        }
        cmap = matplotlib.cm.get_cmap("turbo_r")
        norm = matplotlib.colors.Normalize(vmin=0, vmax=4.2)
        colors = cmap(norm(np.arange(0, 5, step=1)))

        for plot_type, data in plot_categories.items():
            logging.debug(f"Plotting mass plot for {plot_type}.")
            fig, axes = plt.subplots(figsize=(4, 4))
            axes.set_xlabel(r"Halo mass $M_{200c}$ [$\log_{10} M_\odot$]")
            axes.set_ylabel(r"Tracer mass [$M_\odot$]")
            axes.set_yscale("log")

            plot_config = {
                "linestyle": "none",
                "markersize": 2,
                "marker": "D",
                "alpha": 0.8,
            }
            for i, pc in enumerate(parent_categories):
                axes.plot(
                    np.log10(masses),
                    data[:, i],
                    color=colors[i],
                    label=pc,
                    **plot_config,
                )
            axes.legend(
                ncols=3,
                loc="upper center",
                fontsize="small",
                title_fontsize="small",
                title=plot_type,
                bbox_to_anchor=(0.5, 1.25),
            )

            # save figure
            self._save_fig(fig, ident_flag=ident_flags[plot_type])

    def _plot_combined_panels(
        self,
        masses: NDArray,
        plot_categories: dict[str, NDArray],
        parent_categories: list[str],
    ) -> None:
        """
        Plot the mass dependence as a single three-panel figure.

        :param masses: List of cluster masses in solar masses (not log!).
            Shape (N, ).
        :param plot_categories: A dictionary, mapping the title of each
            panel to the corresponding array of mean tracer mass per
            category in each halo of shape (N, 5).
        :param parent_categories: A list of names for each of the five
            parent categories, used to label the data points.
        :return: None, figures are saved to file.
        """
        cmap = matplotlib.cm.get_cmap("turbo_r")
        norm = matplotlib.colors.Normalize(vmin=0, vmax=4.2)
        colors = cmap(norm(np.arange(0, 5, step=1)))

        fig, axes = plt.subplots(ncols=3, figsize=(8, 2.8))
        fig.set_tight_layout(True)
        for ax in axes:
            ax.set_xlabel(r"Halo mass $M_{200c}$ [$\log_{10} M_\odot$]")
            ax.set_ylabel(r"Tracer mass [$M_\odot$]")
            ax.set_yscale("log")

        ax_index = 0
        handles = []
        for plot_type, data in plot_categories.items():
            logging.debug(f"Plotting mass plot for {plot_type}.")
            plot_config = {
                "linestyle": "none",
                "markersize": 2.7,
                "markeredgewidth": 0,
                "marker": "D",
                "alpha": 0.8,
            }
            axes[ax_index].set_title(plot_type, fontsize=10)
            handles = []  # reset to empty
            for i, pc in enumerate(parent_categories):
                line, = axes[ax_index].plot(
                    np.log10(masses),
                    data[:, i],
                    color=colors[i],
                    label=pc,
                    **plot_config,
                )
                handles.append(line)
            ax_index += 1
        fig.legend(
            handles=handles,
            ncols=5,
            loc="upper center",
            # fontsize="small",
            bbox_to_anchor=(0.5, 1.1),
        )

        # save figure
        self._save_fig(fig, ident_flag="combined", tight_layout=True)

    @staticmethod
    def _sum_masses(category_index: NDArray, counts: NDArray) -> NDArray:
        """
        Turn the cunts of each parent category into a mass.

        Archived parent categories are turned into reduced parent category,
        i.e. inner fuzz and primary halo categories are summed together.
        Return an array of size (5, ) with the entries being:

        0. The total tracer mass in unbound state
        1. The total tracer mass in other halos
        2. The total tracer mass in the primary
        3. The total tracer mass in satellites
        4. The total tracer mass that never crossed

        Faulty entries are ignored.

        :param category_index: Array of all parent categories found.
            Must be archived parent categories, i.e. integers from
            0 to 4, or 255 for faulty entries.
        :param counts: The count belonging to each parent category.
        :return: Array containing the total tracer mass of each
            _reduced_ parent category.
        """
        result = np.zeros(5)
        pc_index_list = list(category_index)
        # map archived categories to reduced categories/result array index
        category_mapping = {0: 0, 1: 1, 2: 2, 3: 2, 4: 3, 255: 4}
        # go through parent categories
        for pc, target_pc in category_mapping.items():
            if pc not in pc_index_list:
                continue
            i = pc_index_list.index(pc)
            mass = counts[i] * constants.TRACER_MASS
            result[target_pc] = result[target_pc] + mass
        return result
