"""
Pipeline to plot images of the cool gas distribution in clusters.
"""
from __future__ import annotations

import dataclasses
import logging
from typing import TYPE_CHECKING, ClassVar

import h5py
import matplotlib.pyplot as plt
import numpy as np

from library.config import config
from library.data_acquisition import halos_daq, particle_daq
from library.plotting import plot_radial_profiles
from pipelines import base

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray


@dataclasses.dataclass
class PlotCoolGasDistribution(base.Pipeline):
    """
    Plot the distribution of cool gas in clusters.
    """

    n_bins: int = 80
    z_threshold: float = 0.5  # half slice thickness in virial radii

    ranges: ClassVar[NDArray] = np.array([[-2, 2], [-2, 2]])
    n_clusters: ClassVar[int] = 352
    n_tngclstr: ClassVar[int] = 352
    n_tng300_1: ClassVar[int] = 280

    def __post_init__(self):
        self.tngclstr_basepath = config.get_simulation_base_path("TNG-Cluster")
        self.tng300_1_basepath = config.get_simulation_base_path("TNG300-1")

    def run(self) -> int:
        """
        Create a 2D histogram of cool gas distribution.

        :return: Exit code.
        """
        logging.info(
            "Started pipeline to plot cool gas distribution at redshift zero."
        )
        # Step 1: allocate memory
        distribution = np.zeros((self.n_clusters, self.n_bins, self.n_bins))

        # Step 2: load virial radii of TNG-Cluster
        cluster_data = halos_daq.get_halo_properties(
            self.tngclstr_basepath,
            self.config.snap_num,
            fields=[self.config.radius_field, "GroupPos"],
            cluster_restrict=True,
        )
        virial_radii = cluster_data[self.config.radius_field]
        cluster_positions = cluster_data["GroupPos"]

        # Step 3: loop over TNG-Cluster clusters
        archive = h5py.File(self.config.cool_gas_history, "r")
        for zoom_in in range(self.n_tngclstr):
            logging.info(
                f"Plotting distribution for TNG-Cluster zoom-in {zoom_in:03d}."
            )
            # load indices and masses
            grp = f"ZoomRegion_{zoom_in:03d}"
            s = self.config.snap_num
            all_particle_indices = archive[grp]["particle_indices"][s, :]
            all_masses = archive[grp]["Mass"][s, :]
            uniqueness = archive[grp]["uniqueness_flags"][s, :]

            # make indices and masses unique
            particle_indices = all_particle_indices[uniqueness == 1]
            masses = all_masses[uniqueness == 1]

            # load the particle positions
            all_positions = particle_daq.get_particle_properties(
                self.tngclstr_basepath,
                self.config.snap_num,
                part_type=0,  # gas only
                fields=["Coordinates"],
                zoom_id=zoom_in,
            )["Coordinates"]

            # extract traced positions
            positions = all_positions[particle_indices]
            # center positions on cluster center
            positions_centered = positions - cluster_positions[zoom_in]
            # normalize positions to virial radius
            positions_normed = positions_centered / virial_radii[zoom_in]
            # limit projection depth by cutting out large z-values
            mask = np.abs(positions_normed[:, 2]) <= self.z_threshold
            positions_sliced = positions_normed[mask]
            masses_masked = masses[mask]

            # create a 2D-histogram
            hist = np.histogram2d(
                positions_sliced[:, 0],
                positions_sliced[:, 1],
                self.n_bins,
                range=self.ranges,
                weights=masses_masked,
            )

            # assign to global array
            distribution[zoom_in] = hist[0]

            # plot the histogram
            self._plot_distribution_hist(
                hist[0], f"individuals/zoom_in_{zoom_in}", f"z{zoom_in:03d}"
            )

        # Step 4: create total histogram
        logging.info("Creating overall mean distribution plot.")
        mean_distribution = np.nanmean(distribution, axis=0)
        self._plot_distribution_hist(mean_distribution, None, "overall_mean")

        logging.info("Done! Successfully saved plots to file.")
        return 0

    def _plot_distribution_hist(
        self,
        hist2d: NDArray,
        figure_subdir: str | Path | None,
        ident_flag: str
    ) -> None:
        """
        Save the given 2D histogram to file.

        :param hist2d: The 2D histogram of the cool gas distribution, as
            returned by ``numpy.histogram2d``.
        :param figure_subdir: The subdirectory for the figure to be
            saved under, or None to save it in the base figure directory
            of the milestone.
        :param ident_flag: The ident flag for the file name.
        :return: None, figure is saved to file.
        """
        # Create figure
        fig, axes = plt.subplots(figsize=(12, 10))
        axes.set_aspect("equal")

        # plot 2D hist on top
        plot_radial_profiles.plot_2d_radial_profile(
            fig,
            axes,
            hist2d.transpose(),
            ranges=self.ranges.flatten(),
            xlabel=r"x [$R_{200c}$]",
            ylabel=r"y [$R_{200c}$]",
            colormap="binary",
            cbar_label="Column density [???]",  # TODO: units
            scale="log",
        )

        # add circles for virial radius and 2Rvir
        circle_config = {
            "facecolor": "none",
            "edgecolor": "black",
            "alpha": 0.8,
        }
        inner_circle = plt.Circle(
            (0, 0), radius=1, linestyle="solid", **circle_config
        )
        outer_circle = plt.Circle(
            (0, 0), radius=2, linestyle="dotted", **circle_config
        )
        axes.add_patch(inner_circle)
        axes.add_patch(outer_circle)

        # save figure
        self._save_fig(fig, ident_flag=ident_flag, subdir=figure_subdir)
