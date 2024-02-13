"""
Pipeline to plot radial temperature profiles for individual halos.
"""
from __future__ import annotations

import logging
import time
import tracemalloc
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

import matplotlib.pyplot as plt
import numpy as np

from library.config import config
from library.data_acquisition import gas_daq, halos_daq
from library.loading import load_mass_trends
from library.plotting import common
from library.processing import selection
from pipelines.base import DiagnosticsPipeline

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass
class ClusterCoolGasMassTrendPipeline(DiagnosticsPipeline):
    """
    Pipeline to create plots of cool gas fraction vs halo mass.

    Pipeline creates plot of cool gas fraction vs. cluster mass for all
    clusters (i.e. halos with log M > 14.0) from TNG300-1 and TNG-Cluster.
    They can optionally be colored by a third halo quantity such as SFR
    or black hole mass.

    This pipeline must load all particle data in order to be able to
    plot gas particles that do ot belong to halos as well. There is
    another pipeline that instead infers the cool gas fraction from the
    radial density profiles of the clusters that is far less memory
    intensive when the data for these profiles already exists.
    """

    color_field: str | None
    log: bool = False

    n_clusters: ClassVar[int] = 632
    n300: ClassVar[int] = 280  # number of clusters in TNG300-1
    nclstr: ClassVar[int] = 352  # number of clusters in TNG-Cluster

    def __post_init__(self):
        data_root = self.config.data_home
        self.part_id_dir = (
            data_root / "radial_profiles" / "individuals" / "TNG300_1"
            / "particle_ids"
        )
        self.tng300_basepath = config.get_simulation_base_path("TNG300-1")
        self.tngclstr_basepath = config.get_simulation_base_path("TNG-Cluster")
        # default label
        self.label = self._get_cbar_label(self.color_field)

    def run(self) -> int:
        """
        Create plots of cool gas fraction vs. halo mass for all clusters.

        Steps:

        1. Allocate memory space for the halo gas fraction, masses and
           optional third quantity.
        2. Load halo data for TNG300-1, restrict to clusters. Place mass
           into allocated space. Place third quantity into allocated
           space.
        3. Get gas cell temperatures for TNG300-1.
        4. Get gas cell masses for TNG300-1.
        5. For every halo in TNG300-1:
           1. Select cells within 2R_vir using particle ID filed, mask
              masses and temperatures.
           2. Sum their masses.
           3. Mask masses to cool gas only, sum cool gas mass.
           4. Calculate cool gas fraction, place into allocated array.
           5. Clean-up.
        6. Load halo data for TNG-Cluster, restrict to zoom-ins. Place
           mass into allocated space. Place third quantity into allocated
           array.
        7. For every cluster in TNG-Cluster:
           1. Load gas cell data for the cluster (positions, mass, all
              data required for temperature)
           2. Calculate distances, restrict temperature and masses to
              only cells within 2R_vir
           3. Sum masses of gas cells.
           4. Mask masses to cool gas only, sum cool gas mass.
           5. Calculate cool gas fraction, place into allocated array.
           6. Clean-up.
        8. Plot scatter plot of the data; save data to file.

        :return: Exit code.
        """
        # Step 0: create directories, start memory monitoring, timing
        self._create_directories()
        tracemalloc.start()
        begin = time.time()

        # Step 1: allocate memory for plot data
        halo_masses = np.zeros(self.n_clusters)
        cool_gas_fracs = np.zeros(self.n_clusters)
        color_quantity = np.ones(self.n_clusters)

        # Step 2: load halo data for TNG300-1, restrict to clusters
        logging.info("Loading halo data for TNG300-1.")
        fields = [self.config.mass_field, self.config.radius_field]
        if self.color_field is not None:
            fields.append(self.color_field)
        halo_data = halos_daq.get_halo_properties(
            self.tng300_basepath,
            self.config.snap_num,
            fields=fields,
        )
        cluster_data = selection.select_clusters(
            halo_data, self.config.mass_field, expected_number=self.n300
        )
        # assign halo masses and third quantity
        halo_masses[:self.n300] = cluster_data[self.config.mass_field]
        if self.color_field is not None:
            color_quantity[:self.n300] = cluster_data[self.color_field]
        # clean-up
        del halo_data
        timepoint = self._diagnostics(
            begin, "loading TNG300 halo data", unit="MB"
        )

        # Step 3: Get gas cell temperatures for TNG300-1
        temperatures = gas_daq.get_gas_temperatures(
            self.tng300_basepath, self.config.snap_num
        )
        timepoint = self._diagnostics(
            timepoint, "calculating gas cell temperatures"
        )

        # Step 4: Get gas cell masses for TNG300-1
        masses = gas_daq.get_gas_properties(
            self.tng300_basepath,
            self.config.snap_num,
            ["Masses"],
        )["Masses"]
        timepoint = self._diagnostics(timepoint, "loading gas cell masses")

        # Step 5: Loop through clusters in TNG300-1
        logging.info("Begin processing TNG300-1 halos to get gas fraction.")
        for i, halo_id in enumerate(cluster_data["IDs"]):
            logging.debug(f"Processing halo {halo_id} ({i+ 1} / 280).")
            # Step 5.1: Select cells within 2R_vir, mask masses & temps
            neighbors = np.load(
                self.part_id_dir / f"particles_halo_{halo_id}.npy"
            )
            cur_masses = masses[neighbors]
            cur_temps = temperatures[neighbors]

            # Step 5.2: Sum masses to get total gas mass
            total_mass = np.sum(cur_masses)

            # Step 5.3: Mask masses to cool gas only, sum cool gas mass
            mask = np.digitize(cur_temps, [0, 10**4.5, np.inf])
            cool_gas_masses = selection.mask_quantity(
                cur_masses, mask, index=1, compress=True
            )
            cool_mass = np.sum(cool_gas_masses)

            # Step 5.4: Calculate and save cool gas fraction
            cool_gas_fracs[i] = cool_mass / total_mass

            # Step 5.5: Clean-up
            del neighbors, cur_masses, cur_temps, mask, cool_gas_masses
        timepoint = self._diagnostics(
            timepoint, "processing all TNG300 clusters"
        )

        # Step 6: Load halo data from TNG-Cluster
        logging.info("Loading halo data for TNG-Cluster.")
        cluster_data = halos_daq.get_halo_properties(
            self.tng300_basepath,
            self.config.snap_num,
            fields=fields,
            cluster_restrict=True,
        )
        halo_masses[self.n300:] = cluster_data[self.config.mass_field]
        if self.color_field is not None:
            color_quantity[self.n300:] = cluster_data[self.color_field]
        timepoint = self._diagnostics(
            timepoint, "loading TNG-Cluster halo data"
        )

        # Step 7: Loop through clusters
        logging.info("Begin processing TNG-Cluster halos to get gas fraction.")
        for i, halo_id in enumerate(cluster_data["IDs"]):
            logging.debug(f"Processing halo {halo_id} ({i + 1}/352).")
            # Step 7.1: Load gas cell data for temperature
            cluster_temperatures = gas_daq.get_cluster_temperature(
                halo_id,
                self.tngclstr_basepath,
                self.config.snap_num,
            )

            # Step 7.2: Load gas cell positions, calculate distance
            gas_data = gas_daq.get_gas_properties(
                self.config.base_path,
                self.config.snap_num,
                fields=["Coordinates", "Masses"],
                cluster=halo_id,
            )
            gas_distances = np.linalg.norm(
                gas_data["Coordinates"] - cluster_data["GroupPos"][i], axis=1
            ) / cluster_data[self.config.radius_field][i]

            # Step 7.3: Restrict masses and temperatures to 2R_vir
            cur_masses = gas_data["Masses"][gas_distances <= 2.0]
            cur_temps = cluster_temperatures[gas_distances <= 2.0]
            # TODO: check if gas_data can be deleted here already
            del cluster_temperatures, gas_distances  # clean-up

            # Step 7.4: Sum mass to get total mass
            total_mass = np.sum(cur_masses)

            # Step 7.5: Mask masses to only cool gas
            mask = np.digitize(cur_temps, [0, 10**4.5, np.inf])
            cool_gas_masses = selection.mask_quantity(
                cur_masses, mask, index=1, compress=True
            )
            cool_mass = np.sum(cool_gas_masses)

            # Step 7.6: Calculate gas fraction; place in array
            cool_gas_fracs[self.n300 + i] = cool_mass / total_mass

            # Step 7.7: Clean-up
            del gas_data, cool_gas_masses

        # Step 8: Save data to file and plot it
        if self.to_file:
            filename = f"{self.paths['data_file_stem']}.npz"
            np.savez(
                self.paths["data_dir"] / filename,
                halo_masses=halo_masses,
                cool_gas_fracs=cool_gas_fracs,
                color_quantity=color_quantity,
            )

        if self.color_field is None:
            self._plot(halo_masses, cool_gas_fracs, None, None)
        else:
            if self.log:
                color_quantity = np.log10(color_quantity)
            self._plot(halo_masses, cool_gas_fracs, color_quantity, self.label)
        return 0

    def _plot(
        self,
        halo_masses: NDArray,
        gas_fraction: NDArray,
        colored_quantity: NDArray | None,
        label: str | None
    ) -> None:
        """
        Plot the cool gas fraction vs. halo mass plot.

        :param halo_masses: Array of shape (632, ) containing the halo
            masses in units of M_sol.
        :param gas_fraction: Array of shape (632, ) containing the gas
            fraction of cool gas in halos.
        :param colored_quantity: Array of shape (632, ) containing the
            quantity that shall be used for coloring the data points.
            Can be in arbitrary units. If no color is to be used, set
            this to None. There will be nor transformation applied to
            this data, i.e. if you wish to plot it in log scale, you
            must pass it already in log scale.
        :return: None, figure is saved to file.
        """
        logging.info("Plotting cool gas fraction mass trend for clusters.")
        fig, axes = plt.subplots(figsize=(5, 4))
        axes.set_xlabel(r"Halo mass $M_{200c}$ [$\log M_\odot$]")
        axes.set_ylabel("Cool gas fraction")

        if colored_quantity:
            logging.info(f"Coloring scatter points by {self.color_field}.")
            # determine min and max of colorbar values
            cbar_min = np.min(colored_quantity)
            cbar_max = np.max(colored_quantity)
            if label is None:
                label = self.color_field
            # plot TNG300 data
            common.plot_scatterplot(
                fig,
                axes,
                np.log10(halo_masses[:self.n300]),
                gas_fraction[:self.n300],
                colored_quantity[:self.n300],
                legend_label="TNG300-1",
                marker_style="o",
                cbar_label=label,
                cbar_range=[cbar_min, cbar_max],
            )
            # plot TNG-Cluster data
            common.plot_scatterplot(
                fig,
                axes,
                np.log10(halo_masses[self.n300:]),
                gas_fraction[self.n300:],
                colored_quantity[self.n300:],
                legend_label="TNG-Cluster",
                marker_style="D",
                suppress_colorbar=True,
                cbar_range=[cbar_min, cbar_max],
            )
        else:
            # plot TNG300 data
            common.plot_scatterplot(
                fig,
                axes,
                np.log10(halo_masses[:self.n300]),
                gas_fraction[:self.n300],
                legend_label="TNG300-1",
                marker_style="o",
            )
            # plot TNG-Cluster data
            common.plot_scatterplot(
                fig,
                axes,
                np.log10(halo_masses[self.n300:]),
                gas_fraction[self.n300:],
                legend_label="TNG-Cluster",
                marker_style="D",
            )
        axes.legend()
        ident_flag = self.color_field.lower() if self.color_field else ""
        self._save_fig(fig, ident_flag=ident_flag)

    @staticmethod
    def _get_cbar_label(field: str) -> str:
        """
        Translate a TNG group catalogue field name into a cbar label.

        :param field: The name of the field.
        :return: An appropriate colorbar label.
        """
        match field:
            case "GroupSFR":
                return r"SFR [$M_\odot / yr$]"
            case _:
                return field


class ClusterCoolGasFromFilePipeline(ClusterCoolGasMassTrendPipeline):
    """
    Pipeline to plot the gas fraction vs. mass trend from file.
    """

    def run(self) -> int:
        """Load data from file"""
        filename = f"{self.paths['data_file_stem']}.npz"
        data = load_mass_trends.load_cool_gas_mass_trend_data(
            self.paths["data_dir"] / filename,
            self.n_clusters,
        )
        self._plot(*data, label=self.label)
        return 0
