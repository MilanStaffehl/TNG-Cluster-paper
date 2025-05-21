"""
Pipeline to create data files of important cluster data.
"""
from __future__ import annotations

import logging
import sys
import time
import tracemalloc
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.spatial import KDTree

from library import compute, constants
from library.data_acquisition import gas_daq, halos_daq
from library.processing import selection
from pipelines.base import DiagnosticsPipeline

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass
class TabulateClusterDataPipeline(DiagnosticsPipeline):
    """
    Create files containing particle data for clusters.

    Pipeline will tabulate the following data for every particle
    belonging to a cluster (i.e. a halo with log M > 14.0):

    - Particle ID
    - Radial velocity w.r.t the cluster position in km/s
    - Temperature of gas cell in Kelvin
    - The temperature regime index (cool, warm, hot)
    - The halocentric distance of the particle in units of virial radii

    Each of these will be saved for every gas cell within two times the
    virial radius of every cluster. For every cluster, an individual
    file will be created, holding the array of the data for every cell
    attributed to it.

    If the particle IDs are already tabulated, this pipeline can run
    its task without construction of a KDTree over all particles in the
    TNG300-1 simulation volume, which speeds up calculation considerably.
    This might be the case if the pipeline for individual radial profiles
    has already been run before.
    """

    forbid_tree: bool = True  # whether KDTree construction is allowed
    force_tree: bool = False  # whether KDTree construction must happen

    def __post_init__(self):
        super().__post_init__()
        if self.force_tree and self.forbid_tree:
            logging.fatal(
                "Cannot force and forbid tree creation at the same time. "
                "Cancelling execution."
            )
            sys.exit(3)

        # particle id directory and file suffix
        self.particle_id_dir = (
            self.config.data_home / "particle_ids" / self.config.sim_path
        )
        self.velocity_dir = (
            self.config.data_home / "particle_velocities"
            / self.config.sim_path
        )
        self.temperature_dir = (
            self.config.data_home / "particle_temperatures"
            / self.config.sim_path
        )
        self.regime_dir = (
            self.config.data_home / "particle_regimes" / self.config.sim_path
        )
        self.distances_dir = (
            self.config.data_home / "particle_distances" / self.config.sim_path
        )
        self.gas_mass_dir = (
            self.config.data_home / "particle_masses" / self.config.sim_path
        )

        self.dir_list = [
            self.particle_id_dir,
            self.velocity_dir,
            self.temperature_dir,
            self.regime_dir,
            self.distances_dir,
            self.gas_mass_dir,
        ]

        # create directories
        for directory in self.dir_list:
            if not directory.exists():
                logging.debug(f"Creating missing directory {directory}")
                directory.mkdir(parents=True)

    def run(self) -> int:
        """
        Create data files for clusters in the given simulation.

        Steps:

        1. Load halo data.
        2. Restrict halo data to halos above mass threshold.
        3. Check if tree construction is required.
        4. If tree is required, create particle ID files, discard tree.
        5. Load data for temperature calculation. Calculate gas cell
           temperature, discard rest of loaded data.
        6. Load particle positions and velocities as float32.
        7. For every cluster:
            1. Get particle IDs from file.
            2. Mask gas cell properties to only associated particles.
            3. Calculate halocentric distance and radial velocities for
               all associated particles.
            4. Create temperature regime mask from temperatures.
            5. Save temperatures, distances, velocities and regime
               indices to file and discard data.

        :return: Exit code.
        """
        tracemalloc.start()
        begin = time.time()

        # Step 1: acquire halo data
        fields = [
            self.config.mass_field,
            self.config.radius_field,
            "GroupPos",
            "GroupVel",
        ]
        halo_data = halos_daq.get_halo_properties(
            self.config.base_path, self.config.snap_num, fields=fields
        )
        mem = tracemalloc.get_traced_memory()
        self._memlog("Halo gas data memory usage", mem[0], "MB")

        # Step 2: select only halos above threshold mass
        logging.info("Restricting halo data to log(M) > 14.")
        selected_halos = selection.select_clusters(
            halo_data,
            self.config.mass_field,
            expected_number=constants.N_CLUSTERS[self.config.sim_name],
        )
        del halo_data
        mem = tracemalloc.get_traced_memory()
        self._memlog("Memory usage after restricting halos", mem[0], "kB")
        timepoint = self._timeit(begin, "loading and selecting halo data")

        # Step 3: Check if tree construction is required and/or allowed.
        # If creation is required, it is created here.
        positions, tree = self._check_if_tree_required(selected_halos)
        if tree is not None:
            timepoint = self._diagnostics(timepoint, "constructing KDTree")
        else:
            timepoint = self._diagnostics(timepoint, "KDTree check", unit="MB")

        # Step 4: If tree was required, create particle ID files
        if tree is not None:
            logging.info(
                "Writing particle IDs of associated gas cells for all "
                "clusters to file."
            )
            self._tabulate_associated_particle_ids(
                selected_halos, tree, self.processes
            )
            timepoint = self._diagnostics(
                timepoint, "writing particle IDs to file"
            )
            del tree  # cleanup; tree is large!

        # Step 5: Load data for temperature calculation
        temperatures = gas_daq.get_gas_temperatures(
            self.config.base_path, self.config.snap_num
        )
        timepoint = self._diagnostics(
            timepoint, "calculating gas cell temperatures"
        )

        # Step 6: Load particle positions and velocities
        fields = ["Velocities", "Masses"]
        if positions is None:
            fields.append("Coordinates")
        gas_data = gas_daq.get_gas_properties(
            self.config.base_path, self.config.snap_num, fields=fields
        )
        # append existing data
        if positions is not None:
            gas_data.update({"Coordinates": positions})
        gas_data.update({"Temperatures": temperatures})
        timepoint = self._diagnostics(timepoint, "loading gas cell data")

        # Step 7: Iterate through clusters
        for i, halo_id in enumerate(selected_halos["IDs"]):
            # mask the quantities
            cur_data = self._restrict_gas_data_to_halo(
                gas_data,
                halo_id,
                selected_halos["GroupPos"][i],
                selected_halos["GroupVel"][i],
                selected_halos[self.config.radius_field][i],
            )
            # save data to file
            filename = f"radial_velocity_halo_{halo_id}.npy"
            np.save(self.velocity_dir / filename, cur_data["RadialVelocities"])
            filename = f"particle_distances_halo_{halo_id}.npy"
            np.save(self.distances_dir / filename, cur_data["Distances"])
            filename = f"particle_temperatures_halo_{halo_id}.npy"
            np.save(self.temperature_dir / filename, cur_data["Temperatures"])
            filename = f"gas_masses_halo_{halo_id}.npy"
            np.save(self.gas_mass_dir / filename, cur_data["Masses"])

            # create regime mask and save to file
            mask = np.digitize(
                cur_data["Temperatures"],
                np.array([0, 10**4.5, 10**5.5, np.inf]),
            )
            filename = f"particle_temperature_regimes_halo_{halo_id}.npy"
            np.save(self.regime_dir / filename, mask.astype(np.uint8))

        self._diagnostics(timepoint, "masking and saving cluster data")
        logging.info("Successfully tabulated cluster data for TNG300-1.")

        return 0

    def _check_if_tree_required(
        self, selected_halos: dict[str, NDArray]
    ) -> tuple[NDArray | None, KDTree | None]:
        """
        Checks whether the construction of a KDTree is required.

        If the construction of a KDTree is required, the tree will be
        constructed and returned. If the construction of a KDTree was
        explicitly forced or forbidden, this will be taken into
        consideration.

        Since construction of a KDTree requires the positions of all gas
        cells in TNG300-1 and these are required later, the method loads
        them if a tree is constructed and returns them alongside the
        tree so that they may be used again later without having to load
        them again.

        :param selected_halos: The dictionary containing the restricted
            halo data.
        :return: The tuple of the number of workers and the KDTree, if
            construction of it is required.
        """
        logging.debug(
            f"Searching particle ID directory: {self.particle_id_dir}"
        )
        # get a set of available IDs
        try:
            available_ids = set(
                [f.stem for f in self.particle_id_dir.iterdir()]
            )
        except IOError:
            logging.warning(
                f"Could not find or read the particle IDs from the directory "
                f"{self.particle_id_dir}. Did you delete/move the directory?"
            )
            available_ids = set()

        # get a set of required IDs
        required_ids = set(
            [f"particles_halo_{i}" for i in selected_halos["IDs"]]
        )

        # check whether all halos have particle ID files available
        if required_ids.issubset(available_ids) and not self.force_tree:
            logging.info(
                "Found particle IDs of associated particles for all halos. "
                "Tree construction will be skipped."
            )
            position_data = None
            tree = None
        else:
            logging.debug(f"Missing files: {required_ids - available_ids}")
            # if the user explicitly forbade tree creation, cancel execution
            if self.forbid_tree:
                logging.fatal(
                    "Not all selected halos have associated particle IDs on "
                    "file, but tree creation was forbidden. Cannot continue "
                    "with the job at hand, canceling execution."
                )
                sys.exit(2)
            # otherwise, create the tree
            logging.info(
                "Not all selected halos have particle IDs of associated "
                "particles saved. Continuing with KDTree construction."
            )
            logging.info("Loading gas particle positions for tree.")
            gas_data = gas_daq.get_gas_properties(
                self.config.base_path,
                self.config.snap_num,
                fields=["Coordinates"],
            )
            tree = KDTree(
                gas_data["Coordinates"],
                balanced_tree=True,
                compact_nodes=True,
            )
            position_data = gas_data["Coordinates"]

        return position_data, tree

    def _restrict_gas_data_to_halo(
        self,
        gas_data: dict[str, NDArray],
        halo_id: int,
        halo_pos: NDArray,
        halo_vel: NDArray,
        halo_radius: float,
    ) -> dict[str, NDArray]:
        """
        Restrict the given gas data only to the halo of the given ID.

        Appends to the gas data catalogue also the distance to the
        current halo center in units of virial radii.

        :param gas_data: The dictionary containing the gas data to
            constrain to only the particles within 2 R_vir of the given
            halo.
        :param halo_id: ID of the halo.
        :param halo_pos: The 3D cartesian vector giving the coordinates
            of the halo position. In units of kpc.
        :param halo_vel: The peculiar velocity 3D vector of the halo in
            km/s.
        :param halo_radius: The virial radius of the halo in units of
            kpc.
        :return: The dictionary of gas data, but only containing as
            values arrays, that have been restricted to particles within
            2 R_vir of the given halo. Additionally, also contains new
            fields 'Distances', which contains the distance of every gas
            particle to the halo position in units of virial radii, and
            'RadialVelocities', which contains the radial velocity with
            respect to the halo position in units of km/s.
        """
        filename = f"particles_halo_{halo_id}.npy"
        neighbors = np.load(self.particle_id_dir / filename)

        # restrict gas data to chosen particles only:
        restricted_gas_data = {}
        for field, value in gas_data.items():
            if field == "count":
                restricted_gas_data["count"] = len(neighbors)
                continue
            restricted_gas_data[field] = gas_data[field][neighbors]

        # calculate distances
        part_distances = np.linalg.norm(
            restricted_gas_data["Coordinates"] - halo_pos, axis=1
        )
        part_distances /= halo_radius
        assert np.max(part_distances) <= 2.0

        restricted_gas_data.update({"Distances": part_distances})

        # calculate radial velocities
        radial_vel = compute.get_radial_velocities(
            halo_pos,
            halo_vel,
            restricted_gas_data["Coordinates"],
            restricted_gas_data["Velocities"],
        )
        restricted_gas_data.update({"RadialVelocities": radial_vel})

        return restricted_gas_data

    def _tabulate_associated_particle_ids(
        self,
        selected_halos: dict[str, NDArray],
        positions_tree: KDTree | None,
        workers: int
    ) -> None:
        """
        Save the particle IDs of particles belonging to a halo to file.

        :param selected_halos: The dictionary containing halo data.
            Must contain fields for the radius, the position and the
            halo ID.
        :param positions_tree: If ``self.use_tree`` is True and the
            particles are queried from an existing KDTree, this must be
            the KDTree. Otherwise, if no tree is required, this can be
            set to None.
        :param workers: The number of cores used to query the tree. If
            no tree is used, this can be arbitrarily set to 1.
        :return: None
        """
        if positions_tree is None:
            logging.warning(
                "Was asked to tabulate particle IDs but not given a KDTree. "
                "This should not happen and means that no particle IDs can be "
                "tabulated!"
            )
            return
        for i, halo_id in enumerate(selected_halos["IDs"]):
            # find all particles within 2 * R_vir
            virial_radius = selected_halos[self.config.radius_field][i]
            neighbors = positions_tree.query_ball_point(
                selected_halos["GroupPos"][i],
                2 * virial_radius,
                workers=workers
            )
            logging.debug(
                f"Saving particle indices of halo {halo_id} to file."
            )
            np.save(
                self.particle_id_dir / f"particles_halo_{halo_id}.npy",
                neighbors
            )
