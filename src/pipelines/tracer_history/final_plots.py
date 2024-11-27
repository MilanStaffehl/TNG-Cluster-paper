"""
Pipeline for some final plots, utilizing multiple fields of the archive.
"""
from __future__ import annotations

import dataclasses
import logging
from typing import ClassVar

import h5py
import matplotlib.cm
import matplotlib.colors
import matplotlib.patches
import matplotlib.pyplot as plt
import numpy as np

from library import constants
from library.data_acquisition import halos_daq, sublink_daq
from library.plotting import common
from pipelines import base


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
        logging.info(f"{pc2Rvir_c} {pc1Rvir_c} {pcz0_c}")
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
        logging.info(f"{pc2Rvir_counts} {pc1Rvir_counts} {pcz0_counts}")
        # normalize to a fraction if required, tracer mass otherwise
        if self.fractions:
            pc2Rvir_counts = pc2Rvir_counts / total_n_part
            pc1Rvir_counts = pc1Rvir_counts / total_n_part
            pcz0_counts = pcz0_counts / total_n_part
        else:
            pc2Rvir_counts = pc2Rvir_counts * constants.TRACER_MASS
            pc1Rvir_counts = pc1Rvir_counts * constants.TRACER_MASS
            pcz0_counts = pcz0_counts * constants.TRACER_MASS
        logging.info(f"{pc2Rvir_counts} {pc1Rvir_counts} {pcz0_counts}")

        # Step 5: set up figure
        logging.info("Start plotting bar chart.")
        fig, axes = plt.subplots(figsize=(4, 4))
        # axes.set_xlabel("Category")
        if self.fractions:
            axes.set_ylabel("Fraction")
        else:
            axes.set_ylabel(r"Tracer mass [$\log_{10} M_\odot$]")
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
                label=r"At crossing $2 R_{200}$",
            ),
            matplotlib.patches.Patch(
                **patch_config,
                facecolor="none",
                hatch="......",
                label=r"At crossing $1 R_{200}$",
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
            threshold = r"$R_{200}$"
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
