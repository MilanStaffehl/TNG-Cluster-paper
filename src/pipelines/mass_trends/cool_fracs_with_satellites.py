"""
Pipeline to plot cool gas fracs with a line for cool gas in satellites.
"""
from __future__ import annotations

import copy
import logging
from typing import TYPE_CHECKING

import h5py
import numpy as np
from matplotlib.lines import Line2D

from library.data_acquisition import gas_daq, halos_daq
from library.processing import membership, selection
from pipelines.mass_trends import cool_gas_fracs_clusters

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from numpy.typing import NDArray


class CoolGasMassTrendSatellitesPipeline(
        cool_gas_fracs_clusters.ClusterCoolGasMassTrendPipeline):
    """
    Plot gas frac/mass vs. halo mass, with additional lines for satellites.

    Additional lines show the running mean of the overall cool gas
    fraction/mass and one line for the cool gas only in satellites.
    """

    def __post_init__(self):
        super().__post_init__()

    def run(self) -> int:
        """
        Create plots of cool gas frac/mass vs. cluster mass.

        Adds lines for the running mean plus the running mean of cool
        gas only in satellites. To this end, the pipeline will identify
        this gas in TNG300-1 and load the required data for TNG-Cluster
        from the cool gas history archive.

        Steps:

        1. Find cool gas mass/cool gas fraction in satellites of
           clusters in TNG300-1:
           1. Load the offsets and lengths of FoF-groups in TNG300-1.
           2. Load the group primary subhalo ID for all TNG300-1 clusters
           3. For every cluster:
              1. Load the particle indices located within 2R_200c as
                 saved to file previously.
              2. Use (already global) indices to determine parent subhalo.
              3. Create boolean mask for only those particles that have
                 a parent subhalo which is not the cluster primary.
                 Save these to file, if desired.
              4. Load temperature and mass data for all gas cells in
                 TNG300-1. This is tabulated and can be directly loaded
                 from file.
              5. Filter particles to only cool gas, find their total mass
                 and mass fraction (both w.r.t. total gas mass within
                 2R_200c and w.r.t. only gas in satellites)
              6. Save the fractions to memory.
        2. Find cool gas mass/fraction in satellites of clusters in
           TNG-Cluster:
           1. Load group primary subhalo ID for all clusters.
           2. Open cool gas history archive.
           3. For every cluster:
              1. Load the subhalo parent ID of particles.
              2. Limit particles to only those that have a parent subhalo
                 which is not the cluster primary (including from other
                 FoFs)
              3. Calculate the total mass of all these gas particles.
              4. Save their total mass and gas mass fraction (both w.r.t.
                total gas mass within 2R_200c and w.r.t. only gas in
                satellites) to file and memory
        3. Save all the gas fractions/gas masses in only satellites to
           file.
        4. Load the plot data previously created for the normal version
           of these plots.
        5. Find the running mean in 0.1 dex halo mass bins for all
           clusters.
        6. Create a scatterplot as previously.
        7. Overplot the lines onto the scatterplot.

        :return: Exit code.
        """
        logging.info(
            "Starting pipeline to retrieve and plot cool gas fraction and "
            "mass in satellites only."
        )

        # Step 0: create directories, allocate memory
        self._create_directories()
        satellite_cool_gas_mass = np.zeros(self.n_clusters)
        satellite_cool_gas_frac = np.zeros(self.n_clusters)

        # Step 1: acquire base halo data
        logging.info("Loading basic plot data.")
        base_file = self.paths["data_dir"] / self.base_filename
        if base_file.exists() and not self.force_recalculation:
            logging.info("Found base data on file, loading it from there.")
            with np.load(base_file) as base_data:
                halo_masses = base_data["halo_masses"]
                cool_gas_fracs = base_data["cool_gas_fracs"]
                cool_gas_masses = base_data["cool_gas_masses"]
        elif self.forbid_recalculation:
            halo_masses, cool_gas_fracs, cool_gas_masses = self._load_base_data()
        else:
            halo_masses, cool_gas_fracs, cool_gas_masses = self._get_base_data()

        # Step 2: Find cool gas mass and frac for all satellite gas in
        #  TNG300-1 only.
        logging.info(
            "Starting process to find satellite-only cool gas for "
            "TNG300 clusters."
        )
        _, _, gr_offs, gr_lens = membership.load_offsets_and_lens(
            self.tng300_basepath, self.config.snap_num, subhalo_only=True
        )
        halo_data = halos_daq.get_halo_properties(
            self.tng300_basepath,
            self.config.snap_num, ["GroupFirstSub", self.config.mass_field]
        )
        cluster_data = selection.select_clusters(
            halo_data, self.config.mass_field, expected_number=self.n300
        )
        logging.info(
            "Going through individual clusters to identify bound gas."
        )
        for i, halo_id in enumerate(cluster_data["IDs"]):
            logging.debug(
                f"Finding particles in satellites of halo {halo_id} "
                f"({i + 1}/{self.n300})."
            )
            # Step 1: load particle IDs
            filepath = self.config.data_home / "particle_ids/TNG300_1"
            filename = f"particles_halo_{halo_id}.npy"
            ids = np.load(filepath / filename)

            # Step 2: find parent subhalo IDs
            parent_subhalo = membership.find_parent(
                ids, gr_offs[:, 0], gr_lens[:, 0]
            )

            # Step 3: mask satellites only
            has_parent = (parent_subhalo != -1)
            is_satellite = (parent_subhalo != cluster_data["GroupFirstSub"][i])
            satellite_mask = np.logical_and(has_parent, is_satellite)
            masked_ids = ids[satellite_mask]

            # Step 4: save indices to file and memory
            if self.to_file:
                filepath = (
                    self.config.data_home / "particle_ids_satellites/TNG300_1"
                )
                if not filepath.exists():
                    filepath.mkdir(parents=True)
                filename = f"satellite_particle_ids_halo_{halo_id}.npy"
                np.save(filepath / filename, masked_ids)
                logging.debug(
                    "Saved particle IDs for particles in satellites to file."
                )

            # Step 5: load temperature regimes and masses from file
            filepath = self.config.data_home / "particle_regimes/TNG300_1"
            filename = f"particle_temperature_regimes_halo_{halo_id}.npy"
            regimes = np.load(filepath / filename)
            filepath = self.config.data_home / "particle_masses/TNG300_1"
            filename = f"gas_masses_halo_{halo_id}.npy"
            masses = np.load(filepath / filename)
            cool_mask = (regimes == 1)

            # Step 6: mask ids to both satellites and temperature
            cool_satellite_gas_mask = np.logical_and(satellite_mask, cool_mask)
            cool_sat_masses = masses[cool_satellite_gas_mask]

            # Step 7: save gas fraction/masses to memory
            cool_sat_mass = np.sum(cool_sat_masses)
            if cool_sat_mass > np.sum(masses[cool_mask]):
                logging.warning(
                    f"Halo {halo_id}: found more satellite cool gas mass than total cool gas mass."
                )
            satellite_cool_gas_mass[i] = cool_sat_mass
            satellite_cool_gas_frac[i] = cool_sat_mass / np.sum(masses)

        # Step 3: Find cool gas mass and frac for all satellite gas in
        #  TNG-Cluster only.
        cluster_data = halos_daq.get_halo_properties(
            self.tngclstr_basepath,
            self.config.snap_num,
            ["GroupFirstSub", "GroupPos", self.config.radius_field],
            cluster_restrict=True,
        )
        archive = h5py.File(self.config.cool_gas_history, "r")
        for zoom_in in range(self.nclstr):
            logging.debug(
                f"Finding particles in satellites of zoom-in {zoom_in}."
            )
            # Step 1: Load the masses and parent subhalo IDs
            grp = f"ZoomRegion_{zoom_in:03d}"
            s = int(self.config.snap_num)
            parent_subhalo = archive[grp]["ParentSubhaloIndex"][s, :]
            masses = archive[grp]["Mass"][s, :]
            uniqueness = archive[grp]["uniqueness_flags"][s, :]

            # Step 2: mask satellites only
            has_parent = (parent_subhalo != -1)
            is_satellite = (
                parent_subhalo != cluster_data["GroupFirstSub"][zoom_in]
            )
            satellite_mask = np.logical_and(has_parent, is_satellite)
            mask = np.logical_and(satellite_mask, uniqueness)
            selected_masses = masses[mask]

            # Step 3: find cool gas mass in satellites only
            cool_sat_mass = np.sum(selected_masses)
            i = self.n300 + zoom_in
            total_gas_mass = cool_gas_masses[i] / cool_gas_fracs[i]
            # total_gas_mass = self._get_total_gas_mass(
            #     zoom_in,
            #     cluster_data["GroupPos"][zoom_in],
            #     cluster_data[self.config.radius_field][zoom_in],
            # )
            satellite_cool_gas_mass[i] = cool_sat_mass
            satellite_cool_gas_frac[i] = cool_sat_mass / total_gas_mass

        archive.close()

        # Step 4: Save data to file
        if self.to_file:
            filename = f'{self.paths["data_filestem"]}_satellite_gas_data.npz'
            np.savez(
                self.paths["data_dir"] / filename,
                satellite_cool_gas_frac=satellite_cool_gas_frac,
                satellite_cool_gas_mass=satellite_cool_gas_mass,
            )
        self._diagplot(
            halo_masses, cool_gas_masses, satellite_cool_gas_mass, "mass"
        )
        self._diagplot(
            halo_masses, cool_gas_fracs, satellite_cool_gas_frac, "frac"
        )

        # Step 5: Acquire color data
        color_data = self._get_color_quantity()

        # Step 6: Plot mass dependence with satellite gas only
        kwargs = self._get_plot_kwargs()
        if self.color_scale == "log":
            plot_color = self._nanlog10(color_data)
        else:
            plot_color = np.copy(color_data)
        # mass plot
        self.use_absolute_mass = True  # trick method for correct label
        fm, _ = self._plot(halo_masses, satellite_cool_gas_mass, plot_color, copy.copy(kwargs))
        self._save_fig(fm, ident_flag="satellites_only_mass")
        # fraction plot
        self.use_absolute_mass = False  # trick method for correct label
        ff, _ = self._plot(halo_masses, satellite_cool_gas_frac, plot_color, copy.copy(kwargs))
        self._save_fig(ff, ident_flag="satellites_only_fraction")

        # Step 7: re-plot original data for both gas fraction and gas mass
        self.use_absolute_mass = True
        f_mass, a_mass = self._plot(halo_masses, cool_gas_masses, plot_color, copy.copy(kwargs))
        self.use_absolute_mass = False
        f_frac, a_frac = self._plot(halo_masses, cool_gas_fracs, plot_color, copy.copy(kwargs))

        # Step 8: overplot the satellite gas fractions
        self._overplot_lines(
            a_mass, cool_gas_masses, satellite_cool_gas_mass, halo_masses
        )
        self._overplot_lines(
            a_frac, cool_gas_fracs, satellite_cool_gas_frac, halo_masses
        )

        # Step 9: save figures to file
        self._save_fig(f_mass, ident_flag="mass")
        self._save_fig(f_frac, ident_flag="fraction")
        logging.info("Saved extended mass dependence plots to file!")
        return 0

    def _overplot_lines(
        self,
        axes: Axes,
        cool_gas_all: NDArray,
        cool_gas_satellites: NDArray,
        halo_masses: NDArray,
    ) -> None:
        """
        Plot running average of cool gas mass/frac.

        The method plots a line for the overall cool gas mass/fraction,
        and one for only the gas locked in satellite galaxies.

        :param axes: Axes object onto which to draw the lines.
        :param cool_gas_all: The array of the mean cool gas mass or mean
            cool gas fraction of every cluster.
        :param cool_gas_satellites: The array of the mean cool gas mass
            or mean cool gas fraction of cool gas exclusively in
            satellites.
        :param halo_masses: Array of masses of all clusters.
        :return: None, axes object is altered in place.
        """
        halo_masses = np.log10(halo_masses)
        # Step 1: find mean in 7 mass bins
        left_bin_edges = np.linspace(14.0, 15.3, num=14)
        overall_means = np.zeros_like(left_bin_edges, dtype=np.float64)
        satellite_means = np.zeros_like(overall_means)
        for i, left_bin_edge in enumerate(left_bin_edges):
            right_bin_edge = left_bin_edge + 0.1
            if i == len(left_bin_edges) - 1:
                right_bin_edge += 0.01  # catch the one cluster to right
            where = np.logical_and(
                halo_masses >= left_bin_edge, halo_masses < right_bin_edge
            )
            overall_means[i] = np.nanmean(cool_gas_all[where])
            satellite_means[i] = np.nanmean(cool_gas_satellites[where])

        # Step 2: overplot running mean
        l1, = axes.plot(
            left_bin_edges + 0.05,
            overall_means,
            linestyle="solid",
            color="black",
            zorder=20,
            label="Mean (all gas)",
        )
        l2, = axes.plot(
            left_bin_edges + 0.05,
            satellite_means,
            linestyle="dashed",
            color="black",
            zorder=20,
            label="Mean (satellites only)",
        )
        # restore legend
        tng300_handle = Line2D(
            [],
            [],
            color="black",
            marker="o",
            ls="",
            markersize=3,
            label="TNG300-1",
        )
        tngclstr_handle = Line2D(
            [],
            [],
            color="black",
            marker="D",
            ls="",
            markersize=3,
            label="TNG-Cluster",
        )
        legend = axes.legend(
            handles=[tng300_handle, tngclstr_handle, l1, l2],
            fontsize="small",
        )
        legend.set_zorder(20)

    def _get_total_gas_mass(
        self, zoom_in: int, cluster_pos: NDArray, virial_radius: float
    ) -> float:
        """
        Return the total mass of all gas within 2 R200c of the cluster.

        Method also saves mass of all particles within 2 R200c to file
        if instructed. Method only works for TNG-Cluster.

        :param zoom_in: ID of the zoom-in region of the cluster.
        :param cluster_pos: The 3D position vector of the cluster center.
        :param virial_radius: The virial radius of the cluster in ckpc.
        :return: The total mass of all gas cells within twice the virial
            radius of the cluster, in solar masses.
        """
        logging.debug(
            f"Finding total gas mass for cluster of zoom-in {zoom_in}."
        )

        # attempt to load from file first
        filepath = self.config.data_home / "particle_masses/TNG_cluster"
        filename = f"gas_masses_zoom_in_{zoom_in}.npy"
        if (filepath / filename).exists():
            logging.debug(
                f"Found gas mass for zoom-in {zoom_in} on file. Will load "
                f"from there."
            )
            masses = np.load(filepath / filename)
            return np.sum(masses)

        # otherwise, load from the simulation
        gas_data = gas_daq.get_gas_properties(
            self.tngclstr_basepath,
            self.config.snap_num,
            fields=["Coordinates", "Masses"],
            zoom_id=zoom_in,
        )
        gas_distances = np.linalg.norm(
            gas_data["Coordinates"] - cluster_pos, axis=1
        ) / virial_radius
        selected_masses = gas_data["Masses"][gas_distances <= 2.0]

        # save to file if desired
        if self.to_file:
            if not filepath.exists():
                filepath.mkdir()
            np.save(filepath / filename, selected_masses)
            logging.debug(
                f"Wrote gas masses for all gas within 2 R200 for cluster "
                f"of zoom-in {zoom_in} to file."
            )

        return np.sum(selected_masses)

    def _diagplot(self, masses, total, satellites, i):
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(figsize=(5, 5))
        axes.plot(
            np.log10(masses), total - satellites, marker="o", linestyle="none"
        )
        self._save_fig(fig, ident_flag=f"diag{i}")
