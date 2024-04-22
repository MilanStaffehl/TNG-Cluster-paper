"""
Pipeline to plot radial temperature profiles for individual halos.
"""
from __future__ import annotations

import logging
import sys
import time
import tracemalloc
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, ClassVar

import h5py
import matplotlib.cm
import matplotlib.pyplot as plt
import numpy
import numpy as np
from matplotlib.lines import Line2D
from numpy.typing import NDArray

from library.config import config
from library.data_acquisition import bh_daq, gas_daq, halos_daq
from library.loading import load_radial_profiles
from library.plotting import colormaps, common
from library.processing import selection, statistics
from pipelines.base import DiagnosticsPipeline


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
    color_log: bool = False
    forbid_recalculation: bool = True
    median_deviation: bool = False
    core_only: bool = False

    n_clusters: ClassVar[int] = 632
    n300: ClassVar[int] = 280  # number of clusters in TNG300-1
    nclstr: ClassVar[int] = 352  # number of clusters in TNG-Cluster

    def __post_init__(self):
        super().__post_init__()
        # file paths
        core = "_core" if self.core_only else ""
        self.base_filename = f"mass_trends_clusters{core}_base_data.npz"
        id_subdir = "particle_ids_core" if self.core_only else "particle_ids"
        self.part_id_dir = self.config.data_home / id_subdir / "TNG300_1"
        self.tng300_basepath = config.get_simulation_base_path("TNG300-1")
        self.tngclstr_basepath = config.get_simulation_base_path("TNG-Cluster")
        # list of supported fields to color the data with and the methods
        # that can generate them
        self.field_generators: dict[str, Callable[[], NDArray]] = {
            "SFR": self._get_cluster_sfr,
            "TotalBHMass": self._get_cluster_total_bh_mass,
            "BHMass": self._get_cluster_bh_mass,
            "BHMdot": self._get_cluster_bh_mdot,
            "GasMetallicity": self._get_cluster_gas_metallicity,
            "BHMode": self._get_cluster_bh_mode,
            "BHCumEnergy": self._get_cluster_bh_cum_energy,
            "BHCumMass": self._get_cluster_bh_cum_mass,
            "RelaxednessMass": self._get_cluster_relaxedness_by_mass,
            "RelaxednessDist": self._get_cluster_relaxedness_by_dist,
            "CCT": self._get_cluster_central_cooling_time,
            "CoolCore": self._get_cluster_cool_core_category,
            "FormationRedshift": self._get_cluster_formation_redshift,
        }

    def run(self) -> int:
        """
        Create plots of cool gas fraction vs. halo mass for all clusters.

        Steps:

        1. Get the halo masses and gas fractions either from file or,
           if not created yet, by following these following steps:
            1. Allocate memory space for the halo gas fraction, masses.
            2. Load halo data for TNG300-1, restrict to clusters.
            3. Get gas cell temperatures for TNG300-1.
            4. Get gas cell masses for TNG300-1.
            5. For every halo in TNG300-1:
               1. Select cells within 2R_vir using particle ID filed, mask
                  masses and temperatures.
               2. Sum their masses.
               3. Mask masses to cool gas only, sum cool gas mass.
               4. Calculate cool gas fraction.
               5. Clean-up.
            6. Load halo data for TNG-Cluster, restrict to zoom-ins.
            7. For every cluster in TNG-Cluster:
               1. Load gas cell data for the cluster (positions, mass, all
                  data required for temperature)
               2. Calculate distances, restrict temperature and masses to
                  only cells within 2R_vir
               3. Sum masses of gas cells.
               4. Mask masses to cool gas only, sum cool gas mass.
               5. Calculate cool gas fraction, place into allocated array.
               6. Clean-up.
        3. Acquire third quantity for all clusters.
        4. Plot scatter plot of the data; save it to file.

        :return: Exit code.
        """
        # Step 0: create directories, start memory monitoring, timing
        self._create_directories()
        if self.core_only:
            logging.info("Plotting mass trends for core only.")

        # Step 1: acquire base halo data
        base_file = self.paths["data_dir"] / self.base_filename
        if base_file.exists():
            logging.info("Found base data on file, loading it from there.")
            with np.load(base_file) as base_data:
                halo_masses = base_data["halo_masses"]
                cool_gas_fracs = base_data["cool_gas_fracs"]
        elif self.forbid_recalculation:
            halo_masses, cool_gas_fracs = self._load_base_data()
        else:
            halo_masses, cool_gas_fracs = self._get_base_data()

        # Step 2: acquire data to color the points with
        getter = self.field_generators.get(self.color_field)
        if getter is None:
            logging.error(
                f"Unknown or unsupported field name for color field: "
                f"{self.color_field}. Will plot uncolored plot."
            )
            color_quantity = None
            self.color_field = None  # plot only black dots
        else:
            color_quantity = getter()
            # since there is color data, save it to file
            logging.info(f"Writing color data {self.color_field} to file.")
            filename = f"{self.paths['data_file_stem']}.npy"
            np.save(self.paths["data_dir"] / filename, color_quantity)

        # Step 3: optionally plot not the quantity, but its trend at fixed mass
        if self.median_deviation and color_quantity is not None:
            color_quantity = statistics.find_deviation_from_median_per_bin(
                color_quantity,
                np.log10(halo_masses),
                min_mass=14.0,
                max_mass=15.4,
                num_bins=7,
            )

        # Step 4: plot data
        if self.color_field is None:
            self._plot(halo_masses, cool_gas_fracs, None, {})
        else:
            kwargs = self._get_plot_kwargs(self.color_field)
            if self.color_log:
                color_quantity = np.log10(color_quantity)
            self._plot(halo_masses, cool_gas_fracs, color_quantity, kwargs)

        return 0

    def _get_base_data(self) -> tuple[NDArray, NDArray]:
        """
        Generate the base data for the plot.

        Function requires the particle IDs of particles associated with
        all clusters in TNG300-1 to already exist. These can be generated
        with the :mod:`~pipelines.radial_profiles.individuals` pipelines.

        :return: The tuple of arrays of length 632 of the halo masses
            and their corresponding cool gas fraction.
        """
        tracemalloc.start()
        begin = time.time()

        # Step 1: allocate memory for plot data
        halo_masses = np.zeros(self.n_clusters)
        cool_gas_fracs = np.zeros(self.n_clusters)

        # Step 2: load halo data for TNG300-1, restrict to clusters
        logging.info("Loading halo data for TNG300-1.")
        fields = [self.config.mass_field, self.config.radius_field]
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
        log_level = logging.getLogger("root").level
        core = "_core" if self.core_only else ""
        for i, halo_id in enumerate(cluster_data["IDs"]):
            if log_level <= 15:
                print(f"Processing halo {halo_id} ({i + 1} / 280).", end="\r")
            # Step 5.1: Select cells within 2R_vir, mask masses & temps
            neighbors = np.load(
                self.part_id_dir / f"particles_halo_{halo_id}{core}.npy"
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
        fields.append("GroupPos")
        cluster_data = halos_daq.get_halo_properties(
            self.tngclstr_basepath,
            self.config.snap_num,
            fields=fields,
            cluster_restrict=True,
        )
        halo_masses[self.n300:] = cluster_data[self.config.mass_field]
        timepoint = self._diagnostics(
            timepoint, "loading TNG-Cluster halo data"
        )

        # Step 7: Loop through clusters
        logging.info("Begin processing TNG-Cluster halos to get gas fraction.")
        for i, halo_id in enumerate(cluster_data["IDs"]):
            if log_level <= 15:
                print(f"Processing halo {halo_id} ({i + 1}/352).", end="\r")
            # Step 7.1: Load gas cell data for temperature
            cluster_temperatures = gas_daq.get_cluster_temperature(
                halo_id,
                self.tngclstr_basepath,
                self.config.snap_num,
            )

            # Step 7.2: Load gas cell positions, calculate distance
            gas_data = gas_daq.get_gas_properties(
                self.tngclstr_basepath,
                self.config.snap_num,
                fields=["Coordinates", "Masses"],
                cluster=halo_id,
            )
            gas_distances = np.linalg.norm(
                gas_data["Coordinates"] - cluster_data["GroupPos"][i], axis=1
            ) / cluster_data[self.config.radius_field][i]

            # Step 7.3: Restrict masses and temperatures
            limit = 0.05 if self.core_only else 2.0
            cur_masses = gas_data["Masses"][gas_distances <= limit]
            cur_temps = cluster_temperatures[gas_distances <= limit]
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

        self._diagnostics(timepoint, "processing all TNG-Cluster halos")

        # Step 8: Save data to file and plot it
        if self.to_file:
            np.savez(
                self.paths["data_dir"] / self.base_filename,
                halo_masses=halo_masses,
                cool_gas_fracs=cool_gas_fracs,
            )
        self._timeit(begin, "entire base data acquisition")
        return halo_masses, cool_gas_fracs

    def _load_base_data(self):
        """
        Load the halo masses and cool gas fraction from the density profiles.

        .. attention:: The order of the halos is not preserved with this
            method. Halos may appear in the array in arbitrary order, not
            in the order of their IDs. They are however still split by
            simulation (i.e. first 280 halos are guaranteed to be from
            TNG300, the last 352 are from TNG-Cluster)

        :return: The tuple of arrays of length 632 of the halo masses
            and their corresponding cool gas fraction.
        """
        halo_masses = np.zeros(self.n_clusters)
        cool_gas_fracs = np.zeros(self.n_clusters)

        data_path = self.config.data_home / "radial_profiles" / "individuals"
        subdir_name = "density_profiles"
        if self.core_only:
            subdir_name += "_core"

        limit = 0.05 if self.core_only else 2.0
        edges = np.linspace(0, limit, num=51, endpoint=True)
        shell_volumes = 4 / 3 * np.pi * (edges[1:]**3 - edges[:-1]**3)

        # load data for TNG300-1
        logging.info("Loading data for TNG300-1.")
        halo_loader = load_radial_profiles.load_individuals_1d_profile(
            data_path / "TNG300_1" / subdir_name, (50, )
        )
        idx = 0
        for halo_data in halo_loader:
            halo_masses[idx] = halo_data["halo_mass"]
            cool_profile = halo_data["cool_inflow"] + halo_data["cool_outflow"]
            cool_gas = np.sum(cool_profile * shell_volumes)
            total_profile = halo_data["total_inflow"] + halo_data[
                "total_outflow"]
            total_gas = np.sum(total_profile * shell_volumes)
            cool_gas_fracs[idx] = cool_gas / total_gas
            idx += 1

        # load data for TNG-Cluster
        logging.info("Loading data for TNG-Cluster.")
        halo_loader = load_radial_profiles.load_individuals_1d_profile(
            data_path / "TNG_Cluster" / subdir_name, (50, )
        )
        for halo_data in halo_loader:
            halo_masses[idx] = halo_data["halo_mass"]
            cool_profile = halo_data["cool_inflow"] + halo_data["cool_outflow"]
            cool_gas = np.sum(cool_profile * shell_volumes)
            total_profile = halo_data["total_inflow"] + halo_data[
                "total_outflow"]
            total_gas = np.sum(total_profile * shell_volumes)
            cool_gas_fracs[idx] = cool_gas / total_gas
            idx += 1

        return halo_masses, cool_gas_fracs

    def _plot(
        self,
        halo_masses: NDArray,
        gas_fraction: NDArray,
        colored_quantity: NDArray | None,
        additional_kwargs: dict[str, Any],
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
        :param additional_kwargs: A dictionary containing additional
            keywords for the :func:`plot_scatterplot` function.
        :return: None, figure is saved to file.
        """
        logging.info("Plotting cool gas fraction mass trend for clusters.")
        fig, axes = plt.subplots(figsize=(5, 4))
        axes.set_xlabel(r"Halo mass $M_{200c}$ [$\log M_\odot$]")
        # axes.set_xlim([14.0, 15.4])
        # axes.set_ylim((1e-4, 2e-2))
        if self.core_only:
            axes.set_ylabel(r"Cool gas fraction within $0.05R_{200c}$")
        else:
            axes.set_ylabel(r"Cool gas fraction within $2R_{200c}$")

        if self.log:
            axes.set_yscale("log")
            logging.debug(f"Smallest gas frac value: {np.min(gas_fraction)}")
            # make zero-values visible; scatter them a little
            rng = np.random.default_rng(42)
            n_zeros = len(gas_fraction) - np.count_nonzero(gas_fraction)
            randnums = np.power(5, rng.random(n_zeros))
            gas_fraction[gas_fraction == 0] = 1e-7 * randnums

        if colored_quantity is not None:
            logging.info(f"Coloring scatter points by {self.color_field}.")
            # determine min and max of colorbar values
            cbar_min, cbar_max = None, None
            if "cbar_range" in additional_kwargs.keys():
                cbar_min, cbar_max = additional_kwargs.pop("cbar_range")
            if cbar_min is None:
                cbar_min = np.nanmin(colored_quantity)
            if cbar_max is None:
                cbar_max = np.nanmax(colored_quantity)
            # extract color label or set a default
            try:
                label = additional_kwargs.pop("cbar_label")
            except KeyError:
                logging.warning("Received no color bar label!")
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
                **additional_kwargs,
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
                **additional_kwargs
            )
            # if bin median deviation is used, plot vertical lines
            # if self.median_deviation:
            #     for x in np.linspace(14.0, 15.4, num=7):
            #         axes.axvline(
            #             x,
            #             ymin=0,
            #             ymax=1,
            #             color="grey",
            #             linestyle="dashed",
            #             zorder=0,
            #         )
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
        # set handles manually to avoid coloring them
        tng300_handle = Line2D(
            [], [],
            color="black",
            marker="o",
            ls="",
            markersize=3,
            label="TNG300-1"
        )
        tngclstr_handle = Line2D(
            [], [],
            color="black",
            marker="D",
            ls="",
            markersize=3,
            label="TNG-Cluster"
        )
        axes.legend(handles=[tng300_handle, tngclstr_handle])
        if self.median_deviation:
            self._save_fig(fig, ident_flag="median_dev")
        else:
            self._save_fig(fig)
        logging.info(
            f"Successfully plotted mass trend colored by {self.color_field}."
        )

    def _get_cluster_quantity(
        self,
        call_func: Callable[[str, int], float],
        quantity_descr: str,
    ) -> NDArray:
        """
        Load a quantity for every cluster.

        Given a function ``call_func`` that loads a quantity for a halo
        of a given ID and a simulation base path, return an array of the
        quantities that this function returns for *one* halo for all
        clusters in both TNG300-1 and TNG-Cluster.

        For example, if the function ``call_func`` returns the SFR of
        a halo from a simulation under base path X with ID Y, then the
        return value of this method will be an array of all 632 values
        of the SFR for the clusters in TNG300-1 and TNG-Cluster.

        :param call_func: A function of method that takes only two
            arguments namely a simulation base path and a halo ID and
            returns a single quantity for the given halo as float.
        :param quantity_descr: A description of the quantity. Used for
            logging.
        :return: An array of the quantity returned by ``call_func`` for
            all clusters in TNG300-1 and TNG-Cluster.
        """
        logging.info(f"Loading {quantity_descr} for TNG300-1 and TNG-Cluster.")
        quantity = np.zeros(self.n_clusters)

        # load and restrict TNG300-1 SFRs
        halo_data = halos_daq.get_halo_properties(
            self.tng300_basepath,
            self.config.snap_num,
            [self.config.mass_field],
        )
        cluster_data = selection.select_clusters(
            halo_data, self.config.mass_field, expected_number=self.n300
        )
        # assign return value of quantity getter to the array
        quantity[:self.n300] = np.array(
            [
                call_func(self.tng300_basepath, hid)
                for hid in cluster_data["IDs"]
            ]
        )

        # load TNG-Cluster SFRs
        halo_data = halos_daq.get_halo_properties(
            self.tngclstr_basepath,
            self.config.snap_num,
            ["GroupSFR"],
            cluster_restrict=True,
        )
        # assign return value of quantity getter to the array
        quantity[self.n300:] = np.array(
            [
                call_func(self.tngclstr_basepath, hid)
                for hid in halo_data["IDs"]
            ]
        )

        return quantity

    def _get_cluster_sfr(self) -> NDArray:
        """
        Return the SFR of all clusters in TNG300-1 and TNG-Cluster.

        :return: Array of shape (632, ) of SFRs, and an appropriate
            color bar label for it and plot kwargs.
        """
        logging.info("Loading SFR for TNG300-1 and TNG-Cluster.")
        sfrs = np.zeros(self.n_clusters)

        # load and restrict TNG300-1 SFRs
        halo_data = halos_daq.get_halo_properties(
            self.tng300_basepath,
            self.config.snap_num,
            ["GroupSFR", self.config.mass_field],
        )
        cluster_data = selection.select_clusters(
            halo_data, self.config.mass_field, expected_number=self.n300
        )
        sfrs[:self.n300] = cluster_data["GroupSFR"]

        # load TNG-Cluster SFRs
        halo_data = halos_daq.get_halo_properties(
            self.tngclstr_basepath,
            self.config.snap_num,
            ["GroupSFR"],
            cluster_restrict=True,
        )
        sfrs[self.n300:] = halo_data["GroupSFR"]

        # adjust for zero-SFR when plotting it in log scale
        if self.color_log:
            sfrs[sfrs == 0] = 0.1

        return sfrs

    def _get_cluster_gas_metallicity(self) -> NDArray:
        """
        Return the gas metallicity of all clusters in TNG300-1 and TNG-Cluster.

        Note that the metallicity will be returned in units of solar
        metallicities, not in code units, which simply give the ratio
        M_Z / M_tot.

        :return: Array of shape (632, ) of SFRs, and an appropriate
            color bar label for it and plot kwargs.
        """
        logging.info("Loading gas metallicity for TNG300-1 and TNG-Cluster.")
        gas_z = np.zeros(self.n_clusters)

        # load and restrict TNG300-1 SFRs
        halo_data = halos_daq.get_halo_properties(
            self.tng300_basepath,
            self.config.snap_num,
            ["GroupGasMetallicity", self.config.mass_field],
        )
        cluster_data = selection.select_clusters(
            halo_data, self.config.mass_field, expected_number=self.n300
        )
        gas_z[:self.n300] = cluster_data["GroupGasMetallicity"]

        # load TNG-Cluster SFRs
        halo_data = halos_daq.get_halo_properties(
            self.tngclstr_basepath,
            self.config.snap_num,
            ["GroupGasMetallicity"],
            cluster_restrict=True,
        )
        gas_z[self.n300:] = halo_data["GroupGasMetallicity"]

        # adjust units to solar units
        gas_z /= 0.0127  # convert to solar units

        return gas_z

    def _get_cluster_total_bh_mass(self) -> NDArray:
        """
        Return the total black hole mass per clusters.

        :return: Array of shape (632, ) of black hole masses per cluster,
            and an appropriate color bar label and plot kwargs.
        """
        logging.info("Loading BH masses for TNG300-1 and TNG-Cluster.")
        bh_masses = np.zeros(self.n_clusters)

        # load and restrict TNG300-1 BH masses
        halo_data = halos_daq.get_halo_properties(
            self.tng300_basepath,
            self.config.snap_num,
            ["GroupMassType", self.config.mass_field],
        )
        cluster_data = selection.select_clusters(
            halo_data, self.config.mass_field, expected_number=self.n300
        )
        bh_masses[:self.n300] = cluster_data["GroupMassType"][:, 5]

        # load TNG-Cluster BH masses
        halo_data = halos_daq.get_halo_properties(
            self.tngclstr_basepath,
            self.config.snap_num,
            ["GroupMassType"],
            cluster_restrict=True,
        )
        bh_masses[self.n300:] = halo_data["GroupMassType"][:, 5]

        return bh_masses

    def _get_cluster_bh_mode(self) -> NDArray:
        """
        Return BH mode of all clusters.

        :return: Array of shape (632, ) of black hole mass accretion
            rate per cluster, and an appropriate color bar label and
            plot kwargs.
        """

        # helper func
        def get_black_hole_mode_index(base_path: str, hid: int) -> float:
            """
            Return the black hole mode index.

            Index is given as the difference between the accretion rate
            MDot and the threshold at which the BH switches over from
            kinetic to thermal mode (the threshold is mass dependent).
            See Weinberger et al. (2017) for details.

            :param base_path: Sim base path.
            :param hid: Halo ID.
            :return: The ratio of the Eddington ratio over the mode
                switchover threshold: (Mdor / Mdot_EDdd) / chi.
            """
            logging.debug(f"Finding black hole mode for halo {hid}.")
            # load all required data
            fields = ["BH_Mass", "BH_Mdot", "BH_MdotEddington"]
            bh_data = bh_daq.get_most_massive_blackhole(
                base_path,
                self.config.snap_num,
                hid,
                fields=fields,
            )
            mass = bh_data["BH_Mass"]
            mdot = bh_data["BH_Mdot"]
            eddington_limit = bh_data["BH_MdotEddington"]
            # calculate the threshold for mode switchover
            chi = min(0.002 * (mass / 1e8)**2, 0.1)
            # calculate actual ratio
            eddington_ratio = mdot / eddington_limit
            return eddington_ratio / chi

        return self._get_cluster_quantity(get_black_hole_mode_index, "BH mode")

    def _get_cluster_bh_mass(self) -> NDArray:
        """
        Return mass of the most massive BH per clusters.

        :return: Array of shape (632, ) of mass of the most massive BH
            in every cluster.
        """
        logging.info("Loading BH masses for TNG300-1 and TNG-Cluster.")
        bh_masses = np.zeros(self.n_clusters)

        # load and restrict TNG300-1 mass data (required for restriction)
        halo_data = halos_daq.get_halo_properties(
            self.tng300_basepath,
            self.config.snap_num,
            [self.config.mass_field],
        )
        cluster_data = selection.select_clusters(
            halo_data, self.config.mass_field, expected_number=self.n300
        )
        # load the black hole data for every halo
        fields = ["BH_Mass"]
        i = 0
        for halo_id in cluster_data["IDs"]:
            bh_data = bh_daq.get_most_massive_blackhole(
                self.tng300_basepath, self.config.snap_num, halo_id, fields
            )
            bh_masses[i] = bh_data["BH_Mass"]
            i += 1

        # load TNG-Cluster IDs
        halo_data = halos_daq.get_halo_properties(
            self.tngclstr_basepath,
            self.config.snap_num,
            [self.config.mass_field],
            cluster_restrict=True,
        )
        # load the black hole data for every halo
        for halo_id in halo_data["IDs"]:
            bh_data = bh_daq.get_most_massive_blackhole(
                self.tngclstr_basepath, self.config.snap_num, halo_id, fields
            )
            bh_masses[i] = bh_data["BH_Mass"]
            i += 1

        return bh_masses

    def _get_cluster_bh_mdot(self) -> NDArray:
        """
        Return mass accretion of all clusters.

        :return: Array of shape (632, ) of black hole mass accretion
            rate per cluster, and an appropriate color bar label and
            plot kwargs.
        """
        logging.info(
            "Loading BH accretion rates for TNG300-1 and TNG-Cluster."
        )
        bh_mdots = np.zeros(self.n_clusters)

        # load and restrict TNG300-1 mass data (required for restriction)
        halo_data = halos_daq.get_halo_properties(
            self.tng300_basepath,
            self.config.snap_num,
            [self.config.mass_field],
        )
        cluster_data = selection.select_clusters(
            halo_data, self.config.mass_field, expected_number=self.n300
        )
        # load the black hole data for every halo
        fields = ["BH_Mdot"]
        i = 0
        for halo_id in cluster_data["IDs"]:
            bh_data = bh_daq.get_most_massive_blackhole(
                self.tng300_basepath, self.config.snap_num, halo_id, fields
            )
            bh_mdots[i] = bh_data["BH_Mdot"]
            i += 1

        # load TNG-Cluster IDs
        halo_data = halos_daq.get_halo_properties(
            self.tngclstr_basepath,
            self.config.snap_num,
            [self.config.mass_field],
            cluster_restrict=True,
        )
        # load the black hole data for every halo
        for halo_id in halo_data["IDs"]:
            bh_data = bh_daq.get_most_massive_blackhole(
                self.tngclstr_basepath, self.config.snap_num, halo_id, fields
            )
            bh_mdots[i] = bh_data["BH_Mdot"]
            i += 1

        return bh_mdots

    def _get_cluster_bh_cum_energy(self):
        """
        Return the cumulative energy fraction of most massive BH.

        The fraction is the fraction of the cumulative energy injected
        in kinetic mode over the total energy injected (kinetic + thermal).

        :return: Array of cumulative kinetic energy fraction of most
            massive BH for every cluster.
        """

        # helper func
        def get_black_hole_kinetic_fraction(base_path: str, hid: int) -> float:
            """
            Return the black hole cumulative kinetic energy fraction.

            This fraction is the ratio of the cumulative energy injected
            in kinetic mode over the total cumulative energy injected.

            :param base_path: Sim base path.
            :param hid: Halo ID.
            :return: The ratio of the Eddington ratio over the mode
                switchover threshold: (Mdor / Mdot_EDdd) / chi.
            """
            logging.debug(
                f"Finding black hole cumulative kinetic energy fraction "
                f"for halo {hid}."
            )
            # load all required data
            fields = ["BH_CumEgyInjection_QM", "BH_CumEgyInjection_RM"]
            bh_data = bh_daq.get_most_massive_blackhole(
                base_path,
                self.config.snap_num,
                hid,
                fields=fields,
            )
            total_energy_injected = (
                bh_data["BH_CumEgyInjection_RM"]
                + bh_data["BH_CumEgyInjection_RM"]
            )
            return bh_data["BH_CumEgyInjection_RM"] / total_energy_injected

        return self._get_cluster_quantity(
            get_black_hole_kinetic_fraction,
            "BH kinetic energy injection fraction",
        )

    def _get_cluster_bh_cum_mass(self):
        """
        Return the cumulative mass accretion fraction of most massive BH.

        The fraction is the fraction of the cumulative mass accreted in
        kinetic mode over the total mass accreted (kinetic + thermal).

        :return: Array of cumulative mass accretion fraction of most
            massive BH for every cluster.
        """

        # helper func
        def get_black_hole_kinetic_fraction(base_path: str, hid: int) -> float:
            """
            Return the black hole cumulative kinetic accretion fraction.

            This fraction is the ratio of the cumulative mass accreted
            in kinetic mode over the total cumulative mass accreted.

            :param base_path: Sim base path.
            :param hid: Halo ID.
            :return: The ratio of the Eddington ratio over the mode
                switchover threshold: (Mdor / Mdot_EDdd) / chi.
            """
            logging.debug(
                f"Finding black hole cumulative kinetic accretion fraction "
                f"for halo {hid}."
            )
            # load all required data
            fields = ["BH_CumMassGrowth_QM", "BH_CumMassGrowth_RM"]
            bh_data = bh_daq.get_most_massive_blackhole(
                base_path,
                self.config.snap_num,
                hid,
                fields=fields,
            )
            total_mass_accreted = (
                bh_data["BH_CumMassGrowth_QM"] + bh_data["BH_CumMassGrowth_RM"]
            )
            return bh_data["BH_CumMassGrowth_RM"] / total_mass_accreted

        return self._get_cluster_quantity(
            get_black_hole_kinetic_fraction,
            "BH kinetic mass accretion fraction",
        )

    def _get_cluster_relaxedness_by_mass(self) -> NDArray:
        """
        Return the relaxedness of the clusters for TNG-Cluster only.

        :return: Array of relaxedness according to mass criterion.
        """
        relaxedness = numpy.zeros(self.n_clusters)
        relaxedness[:self.n300] = np.nan
        path = (
            Path(self.tngclstr_basepath).parent / "postprocessing" / "released"
            / "Relaxedness.hdf5"
        )
        with h5py.File(path, "r") as file:
            relaxedness[self.n300:] = np.array(
                file["Halo"]["Mass_Criterion"][99]
            )
        return relaxedness

    def _get_cluster_relaxedness_by_dist(self) -> NDArray:
        """
        Return the relaxedness of the clusters for TNG-Cluster only.

        :return: Array of relaxedness according to distance criterion.
        """
        relaxedness = numpy.zeros(self.n_clusters)
        relaxedness[:self.n300] = np.nan
        path = (
            Path(self.tngclstr_basepath).parent / "postprocessing" / "released"
            / "Relaxedness.hdf5"
        )
        with h5py.File(path, "r") as file:
            relaxedness[self.n300:] = np.array(
                file["Halo"]["Distance_Criterion"][99]
            )
        return relaxedness

    def _get_cluster_central_cooling_time(self) -> NDArray:
        """
        Return the central cooling time of clusters in TNG Cluster.

        :return: Central cooling time in Gyr.
        """
        cct = np.zeros(self.n_clusters)
        cct[:self.n300] = np.nan
        path = (
            Path(self.tngclstr_basepath).parent / "postprocessing" / "released"
            / "CCcriteria.hdf5"
        )
        with h5py.File(path, "r") as file:
            cct[self.n300:] = np.array(file["centralCoolingTime"][:, 99])
        return cct

    def _get_cluster_cool_core_category(self) -> NDArray:
        """
        Return the central cooling time of clusters in TNG Cluster.

        :return: Central cooling time in Gyr.
        """
        cct = np.zeros(self.n_clusters)
        cct[:self.n300] = np.nan
        path = (
            Path(self.tngclstr_basepath).parent / "postprocessing" / "released"
            / "CCcriteria.hdf5"
        )
        with h5py.File(path, "r") as file:
            cct[self.n300:] = np.array(file["centralCoolingTime_flag"][:, 99])
        return cct

    def _get_cluster_formation_redshift(self) -> NDArray:
        """
        Return array of formation redshifts.

        :return: Formation redshifts only for TNG-Cluster clusters.
        """
        # Load data for TNG-Cluster
        formation_z = np.zeros(self.n_clusters)
        formation_z[:self.n300] = np.nan
        path = (
            Path(self.tngclstr_basepath).parent / "postprocessing" / "released"
            / "FormationHistories.hdf5"
        )
        with h5py.File(path, "r") as file:
            formation_z[self.n300:] = np.array(
                file["Halo"]["Redshift_formation"][:, 99]
            )
        return formation_z

    def _get_plot_kwargs(self, field: str) -> dict[str, Any]:
        """
        Return keyword parameters for the plotting function.

        For the given field, return a dictionary of keyword arguments
        for the plotting function.

        :param field: Name of the field.
        :return: Dict of keyword argument.
        """
        if self.median_deviation:
            return self._get_plot_kwargs_for_median_diff(field)
        else:
            return self._get_plot_kwargs_normal(field)

    def _get_plot_kwargs_normal(self, field: str) -> dict[str, Any]:
        """
        Return keyword parameters for plotting normal quantities.

        For the given field, return a dictionary of keyword arguments
        for the plotting function. This method returns the kwargs for
        the normal plotting of color data.

        :param field: Name of the field.
        :return: Dict of keyword argument.
        """
        # TODO: make this monstrosity readable
        match field:
            case "SFR":
                if self.color_log:
                    label = r"SFR [$\log(M_\odot / yr)$]"
                else:
                    label = r"SFR [$M_\odot / yr$]"
                config_dict = {
                    "cbar_label": label,
                    "cmap": "viridis",
                    "cbar_range": (0., 2.),
                    "cbar_caps": "both"
                }
                return config_dict

            case "GasMetallicity":
                if self.color_log:
                    label = r"Gas metallicity [$\log Z_\odot$]"
                else:
                    label = r"Gas metallicity [$Z_\odot$]"
                return {"cbar_label": label, "cmap": "cividis"}

            case "TotalBHMass":
                if self.color_log:
                    label = r"Total BH mass [$\log M_\odot$]"
                else:
                    label = r"Total BH mass [$M_\odot$]"
                return {"cbar_label": label, "cmap": "plasma"}

            case "BHMass":
                if self.color_log:
                    label = r"Most massive BH mass [$\log M_\odot$]"
                else:
                    label = r"Most massive BH mass [$M_\odot$]"
                return {"cbar_label": label, "cmap": "plasma"}

            case "BHMdot":
                if self.color_log:
                    label = r"BH accretion rate [$\log (M_\odot / Gyr)$]"
                else:
                    label = r"BH accretion rate [$M_\odot / Gyr$]"
                config_dict = {
                    "cbar_label": label,
                    "cmap": "magma",
                }
                return config_dict

            case "BHMode":
                if self.color_log:
                    norm_config = {"vmin": -5, "vcenter": 0, "vmax": 5}
                    label = (
                        r"BH mode [$\log_{10}((\dot{M} / "
                        r"\dot{M}_{Edd}) / \chi)$]"
                    )
                else:
                    norm_config = {"vmin": 0, "vcenter": 1, "vmax": 10}
                    label = r"BH mode [$(\dot{M} / \dot{M}_{Edd}) / \chi$]"
                cmap, norm = colormaps.two_slope_cmap(
                    "winter", "autumn", **norm_config
                )
                config_dict = {
                    "cbar_label": label,
                    "norm": norm,
                    "cmap": cmap,
                    "cbar_caps": "both",
                }
                return config_dict

            case "BHCumEnergy":
                if self.color_log:
                    label = r"Cumulative kinetic energy fraction [$\log_{10}$]"
                else:
                    label = r"Cumulative kinetic energy fraction"
                return {"cbar_label": label, "cmap": "Wistia"}

            case "BHCumMass":
                if self.color_log:
                    label = r"Cumulative accreted mass fraction [$\log_{10}$]"
                else:
                    label = r"Cumulative accreted mass fraction"
                return {"cbar_label": label, "cmap": "magma"}

            case "RelaxednessDist":
                if self.color_log:
                    norm_config = {
                        "vmin": -2, "vcenter": np.log10(0.1), "vmax": 1
                    }
                    label = (
                        r"$\log_{10} (|\vec{r}_{center} - \vec{r}_{COM}| "
                        r"/ R_{200c})$"
                    )
                else:
                    norm_config = {"vmin": 0, "vcenter": 0.1, "vmax": 1}
                    label = r"$|\vec{r}_{center} - \vec{r}_{COM}| / R_{200c}$"
                cmap, norm = colormaps.two_slope_cmap(
                    "cool", "Wistia", **norm_config
                )
                config_dict = {
                    "cbar_label": label,
                    "norm": norm,
                    "cmap": cmap,
                    "cbar_caps": "both",
                }
                return config_dict

            case "RelaxednessMass":
                if self.color_log:
                    norm_config = {
                        "vmin": -0.5, "vcenter": np.log10(0.85), "vmax": 0
                    }
                    label = r"$\log_{10} (M_{central} / M_{tot})$"
                else:
                    norm_config = {"vmin": 0.3, "vcenter": 0.85, "vmax": 1}
                    label = r"$M_{central} / M_{tot}$"
                cmap, norm = colormaps.two_slope_cmap("cool", "Wistia", **norm_config)
                config_dict = {
                    "cbar_label": label,
                    "norm": norm,
                    "cmap": cmap,
                    "cbar_caps": "both",
                }
                return config_dict

            case "FormationRedshift":
                if self.color_log:
                    label = r"Redshift [$\log_{10}(z)$]$"
                else:
                    label = "Redshift z"
                return {"cbar_label": label, "cmap": "gist_heat"}

            case "CCT":
                if self.color_log:
                    label = r"Central cooling time [$\log_{10} Gyr$]"
                else:
                    label = r"Central cooling time [$Gyr$]"
                cmap = colormaps.custom_cmap((100, 190, 230), (50, 0, 40))
                return {"cbar_label": label, "cmap": cmap}

            case "CoolCore":
                if self.color_log:
                    logging.warning("Cool core criteria cannot be log scaled.")
                    raise ValueError("Log scale not supported")
                else:
                    label = "Core classification"
                    norm = matplotlib.colors.BoundaryNorm(
                        [-0.5, 0.5, 1.5, 2.5], ncolors=256
                    )
                    return {
                        "cbar_label": label, "norm": norm, "cmap": "plasma"
                    }

            case _:
                logging.error(
                    f"Unknown field {field} received, returning empty "
                    f"kwargs dict."
                )
                return {}

    def _get_plot_kwargs_for_median_diff(self, field: str) -> dict[str, Any]:
        """
        Return keyword parameters for plotting median deviation.

        For the given field, return a dictionary of keyword arguments
        for the plotting function. This method returns the kwargs for
        plotting the deviation of a quantity from a mass bin median.

        :param field: Name of the field.
        :return: Dict of keyword argument.
        """
        # find a proper description of the values
        match field:
            case "SFR":
                descr = r"SFR / \widetilde{SFR}"
                limits = [0, 4, -1, 1]
            case "BHMass":
                descr = r"M / \tilde{M}"
                limits = [0, 4, -0.5, 0.35]
            case "TotalBHMass":
                descr = r"M / \tilde{M}"
                limits = [0.5, 2, -0.2, 0.5]
            case "BHMdot":
                descr = r"\dot{M} / \tilde{\dot{M}}"
                limits = [0, 10, -1.5, 1.5]
            case "GasMetallicity":
                descr = r"Z / \tilde{Z}"
                limits = [0.8, 1.2, -0.075, 0.075]
            case _:
                logging.warning(
                    f"Unknown field {field} received, label will be empty."
                )
                descr = "?"
                limits = [0, 4, -1, 1]
        # config values
        if self.color_log:
            label = rf"Deviation from median [$\log_{{10}}({descr})$]"
            norm_config = {"vmin": limits[2], "vcenter": 0, "vmax": limits[3]}
        else:
            label = rf"Deviation from median [${descr}$]"
            norm_config = {"vmin": limits[0], "vcenter": 1, "vmax": limits[1]}

        # create a custom colormap
        full_range_channel = np.linspace(0, 1, 256)
        full_cmap = np.zeros((512, 3))
        full_cmap[:256, 2] = np.flip(full_range_channel)  # blue channel
        full_cmap[256:, 0] = full_range_channel  # red channel
        deviation_map = matplotlib.colors.LinearSegmentedColormap.from_list(
            "deviation_map", full_cmap
        )
        norm = matplotlib.colors.TwoSlopeNorm(**norm_config)

        config_dict = {
            "cbar_label": label,
            "norm": norm,
            "cmap": deviation_map,
            "cbar_caps": "both",
        }
        return config_dict


class ClusterCoolGasFromFilePipeline(ClusterCoolGasMassTrendPipeline):
    """
    Pipeline to plot the gas fraction vs. mass trend from file.

    Plots a single quantity as a single panel.
    """

    def __post_init__(self):
        super().__post_init__()

    def run(self) -> int:
        """Load data from file and plot it."""
        self._verify_directories()
        if self.color_field is None:
            logging.fatal("Require color field to plot.")
            sys.exit(10)

        # load base data
        logging.info("Loading base data from file.")
        base_file = self.paths["data_dir"] / self.base_filename
        if not base_file.exists():
            logging.fatal(f"Base data file {base_file} does not exist!")
            return 1

        with np.load(base_file) as base_data:
            halo_masses = base_data["halo_masses"]
            cool_gas_fracs = base_data["cool_gas_fracs"]

        # adjust zero-values
        if self.log:
            # make zero-values visible; scatter them a little
            rng = np.random.default_rng(42)
            n_zeros = len(cool_gas_fracs) - np.count_nonzero(cool_gas_fracs)
            randnums = np.power(5, rng.random(n_zeros))
            cool_gas_fracs[cool_gas_fracs == 0] = 1e-7 * randnums

        # load color data
        filename = f"{self.paths['data_file_stem']}.npy"
        color_data = np.load(self.paths["data_dir"] / filename)
        color_kwargs = self._get_plot_kwargs(self.color_field)

        # plot deviation instead if desired
        if self.median_deviation:
            color_data = statistics.find_deviation_from_median_per_bin(
                color_data,
                np.log10(halo_masses),
                min_mass=14.0,
                max_mass=15.4,
                num_bins=7,
            )

        if self.color_log:
            color_data = np.log10(color_data)
        self._plot(halo_masses, cool_gas_fracs, color_data, color_kwargs)
