"""
Pipeline to plot radial temperature profiles for individual halos.
"""
from __future__ import annotations

import logging
import time
import tracemalloc
from dataclasses import dataclass
from typing import Any, Callable, ClassVar, TypeAlias

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm
from numpy.typing import NDArray

from library.config import config
from library.data_acquisition import gas_daq, halos_daq
from library.loading import load_radial_profiles
from library.plotting import common
from library.processing import selection
from pipelines.base import DiagnosticsPipeline

ColorData: TypeAlias = tuple[NDArray, dict[str, Any]]


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
    field_idx: int = -1
    log: bool = False
    color_log: bool = False
    forbid_recalculation: bool = True

    n_clusters: ClassVar[int] = 632
    n300: ClassVar[int] = 280  # number of clusters in TNG300-1
    nclstr: ClassVar[int] = 352  # number of clusters in TNG-Cluster
    base_filename: ClassVar[str] = "mass_trends_clusters_base_data.npz"

    def __post_init__(self):
        data_root = self.config.data_home
        self.part_id_dir = (
            data_root / "radial_profiles" / "individuals" / "TNG300_1"
            / "particle_ids"
        )
        self.tng300_basepath = config.get_simulation_base_path("TNG300-1")
        self.tngclstr_basepath = config.get_simulation_base_path("TNG-Cluster")
        # list of supported fields to color the data with and the methods
        # that can generate them
        self.field_generators: dict[str, Callable[[], ColorData]] = {
            "SFR": self._get_cluster_sfr,
            "BHMass": self._get_cluster_bh_mass,
            "BHMdot": self._get_cluster_bh_mdot,
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

        # Step 1: acquire base halo data
        base_file = self.paths["data_dir"] / self.base_filename
        if base_file.exists():
            logging.info("Loading base data from file.")
            with np.load(base_file) as base_data:
                halo_masses = base_data["halo_masses"]
                cool_gas_fracs = base_data["cool_gas_fracs"]
        elif self.forbid_recalculation:
            halo_masses, cool_gas_fracs = self._load_base_data()
        else:
            halo_masses, cool_gas_fracs = self._get_base_data()

        # Step 2: acquire data to color the points with
        try:
            color_quantity, kwargs = self.field_generators[self.color_field]()
        except KeyError:
            logging.error(
                f"Unknown or unsupported field name for color field: "
                f"{self.color_field}. Will plot uncolored plot."
            )
            color_quantity = None
            kwargs = {}
            self.color_field = None  # plot only black dots
        else:
            # since there is color data, save it to file
            logging.info(f"Writing color data {self.color_field} to file.")
            filename = f"{self.paths['data_file_stem']}_{self.color_field}.npy"
            np.save(self.paths["data_dir"] / filename, color_quantity)

        # Step 3: plot data
        if self.color_field is None:
            self._plot(halo_masses, cool_gas_fracs, None, kwargs)
        else:
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
        for i, halo_id in enumerate(cluster_data["IDs"]):
            if log_level <= 15:
                print(f"Processing halo {halo_id} ({i + 1} / 280).", end="\r")
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

        edges = np.linspace(0, 2, num=51, endpoint=True)
        shell_volumes = 4 / 3 * np.pi * (edges[1:]**3 - edges[:-1]**3)

        # load data for TNG300-1
        logging.info("Loading data for TNG300-1.")
        halo_loader = load_radial_profiles.load_individuals_1d_profile(
            data_path / "TNG300_1" / "density_profiles", (50, )
        )
        idx = 0
        for halo_data in halo_loader:
            halo_masses[idx] = halo_data["halo_mass"]
            cool_gas = np.sum(halo_data["cool_histogram"] * shell_volumes)
            total_gas = np.sum(halo_data["total_histogram"] * shell_volumes)
            cool_gas_fracs[idx] = cool_gas / total_gas
            idx += 1

        # load data for TNG-Cluster
        logging.info("Loading data for TNG-Cluster.")
        halo_loader = load_radial_profiles.load_individuals_1d_profile(
            data_path / "TNG_Cluster" / "density_profiles", (50, )
        )
        for halo_data in halo_loader:
            halo_masses[idx] = halo_data["halo_mass"]
            cool_gas = np.sum(halo_data["cool_histogram"] * shell_volumes)
            total_gas = np.sum(halo_data["total_histogram"] * shell_volumes)
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
            keywords for the :fucn:`plot_scatterplot` function.
        :return: None, figure is saved to file.
        """
        logging.info("Plotting cool gas fraction mass trend for clusters.")
        fig, axes = plt.subplots(figsize=(5, 4))
        axes.set_xlabel(r"Halo mass $M_{200c}$ [$\log M_\odot$]")
        axes.set_ylabel("Cool gas fraction")
        if self.log:
            axes.set_yscale("log")
            logging.debug(f"Smallest gas frac value: {np.min(gas_fraction)}")
            # make zero-values visible
            gas_fraction[gas_fraction == 0] = 1e-7

        if colored_quantity is not None:
            logging.info(f"Coloring scatter points by {self.color_field}.")
            # determine min and max of colorbar values
            cbar_min = np.min(colored_quantity)
            cbar_max = np.max(colored_quantity)
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
        legend = axes.legend()
        # make dots black
        for handle in legend.legend_handles:
            handle.set_color("black")
        self._save_fig(fig)

    def _get_cluster_sfr(self) -> ColorData:
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

        # label
        if self.color_log:
            label = r"SFR [$\log(M_\odot / yr)$]"
        else:
            label = r"SFR [$M_\odot / yr$]"

        return sfrs, {"cbar_label": label}

    def _get_cluster_bh_mass(self) -> ColorData:
        """
        Return the black hole mass in all clusters, plus aux data.

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

        if self.color_log:
            label = r"BH mass [$\log M_\odot$]"
        else:
            label = r"BH mass [$M_\odot$]"

        return bh_masses, {"cbar_label": label}

    def _get_cluster_bh_mdot(self) -> ColorData:
        """
        Return mass accretion of all clusters, plus aux data.

        :return: Array of shape (632, ) of black hole mass accretion
            rate per cluster, and an appropriate color bar label and
            plot kwargs.
        """
        logging.info("Loading BH accretion rate for TNG300-1 and TNG-Cluster.")
        bh_mdots = np.zeros(self.n_clusters)

        # load and restrict TNG300-1 BH accretion rates
        halo_data = halos_daq.get_halo_properties(
            self.tng300_basepath,
            self.config.snap_num,
            ["GroupBHMdot", self.config.mass_field],
        )
        cluster_data = selection.select_clusters(
            halo_data, self.config.mass_field, expected_number=self.n300
        )
        bh_mdots[:self.n300] = cluster_data["GroupBHMdot"]

        # load TNG-Cluster BH accretion rates
        halo_data = halos_daq.get_halo_properties(
            self.tngclstr_basepath,
            self.config.snap_num,
            ["GroupBHMdot"],
            cluster_restrict=True,
        )
        bh_mdots[self.n300:] = halo_data["GroupBHMdot"]

        boundaries = np.array([np.min(bh_mdots), 1, np.max(bh_mdots)])
        if self.color_log:
            label = r"BH accretion rate [$\log (M_\odot / Gyr)$]"
            # set boundaries to scale
            boundaries = np.log10(boundaries)
        else:
            label = r"BH accretion rate [$M_\odot / Gyr$]"

        norm = BoundaryNorm(boundaries, ncolors=2)

        return bh_mdots, {"cbar_label": label, "norm": norm}


class ClusterCoolGasFromFilePipeline(ClusterCoolGasMassTrendPipeline):
    """
    Pipeline to plot the gas fraction vs. mass trend from file.
    """

    def run(self) -> int:
        """Load data from file"""
        raise NotImplementedError("Currently unavailable.")
        self._verify_directories()

        # load base data
        base_file = self.paths["data_dir"] / self.base_filename
        if not base_file.exists():
            logging.fatal(f"Base data file {base_file} does not exist!")
            return 1

        with np.load(base_file) as base_data:
            halo_masses = base_data["halo_masses"]
            cool_gas_fracs = base_data["cool_gas_fracs"]

        # check if any color data is loaded
        if self.color_field is None:
            self._plot(halo_masses, cool_gas_fracs, None, None)
            return 0

        # Load color data
        color_fname = f"{self.paths['data_file_stem']}_{self.color_field}.npy"
        color_file = self.paths["data_dir"] / color_fname
        if not color_file.exists():
            logging.fatal(
                f"File for color data {color_file} does not exist yet. Run "
                f"the pipeline without the --load flag first to generate it."
            )
            return 2
        color_data = np.load(color_file)

        # TODO: fix this: label and norm must be retrieved differently
        label = self._get_cbar_label(self.color_field)
        self._plot(halo_masses, cool_gas_fracs, color_data, label=label)
        return 0
