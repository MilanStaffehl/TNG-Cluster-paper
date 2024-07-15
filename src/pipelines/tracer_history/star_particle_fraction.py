"""
Pipeline to plot movement of a few particles with time.
"""
from __future__ import annotations

import dataclasses
import logging

import h5py
import matplotlib.pyplot as plt
import numpy as np

from library.plotting import common as common_plt
from pipelines.base import Pipeline


@dataclasses.dataclass
class PlotStarParticleFractionPipeline(Pipeline):
    """
    Pipeline to plot the fraction of particles in stars over time.

    Pipeline plots the fraction of particles in star particles in every
    snapshot over time for every halo, as well as the mean over all
    halos.
    """

    def run(self) -> int:
        """
        Plot particle star fraction with time.

        :return: Exit code.
        """
        # Step 0: verify directories exist
        self._verify_directories()

        # Step 1: allocate memory
        logging.info("Plotting star fraction for TNG-Cluster.")
        star_part_frac = np.zeros((352, 100), dtype=float)

        # Step 2: Open the file with the data
        f = h5py.File(
            self.paths["data_dir"] / "particle_ids" / "TNG_Cluster"
            / "particle_ids_from_snapshot_99.hdf5"
        )

        # Step 3: Create a figure
        fig, axes = plt.subplots()
        xs = common_plt.make_redshift_plot(axes)
        axes.set_ylabel("Tracer fraction in stars & wind")
        plot_config = {
            "linestyle": "solid",
            "marker": "none",
            "color": "gold",
            "alpha": 0.1,
        }

        # Step 4: go through clusters and plot their star fraction
        for zoom_id in range(352):
            type_flags = f[f"ZoomRegion_{zoom_id:03d}/particle_type_flags"][()]
            stars = np.count_nonzero(type_flags == 4, axis=1)
            star_frac = stars / type_flags.shape[1]
            star_part_frac[zoom_id] = star_frac
            axes.plot(xs, star_frac, **plot_config)

        f.close()

        # Step 5: Find mean and median
        mean = np.mean(star_part_frac, axis=0)
        median = np.median(star_part_frac, axis=0)
        axes.plot(xs, mean, linestyle="solid", color="black", marker="none")
        axes.plot(xs, median, linestyle="dashed", color="black", marker="none")

        # Step 6: Save plot
        self._save_fig(fig)
        logging.info("Successfully plotted star fraction plot!")

        return 0
