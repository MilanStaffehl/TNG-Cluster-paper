"""
Pipeline to generate the data required to work with tracers.
"""
from __future__ import annotations

import dataclasses
import logging
from typing import TYPE_CHECKING

import illustris_python as il
import numpy as np

from library.data_acquisition import gas_daq, halos_daq, tracers_daq
from library.processing import selection
from pipelines import base

if TYPE_CHECKING:
    from numpy.typing import NDArray


class GenerateTNGClusterTracerIDsAtRedshiftZero(base.DiagnosticsPipeline):
    """
    Find, for every cluster in TNG-Cluster, tracers in cool gas.

    Pipeline finds all tracers that, at redshift zero, are inside gas
    cells with a temperature below 10^4.5 Kelvin and that are within
    two virial radii of their clusters center. The tracer IDs are saved
    to file.
    """

    def __post_init__(self):
        super().__post_init__()
        self.data_subdir = f"cool_gas_tracer_ids_{self.config.snap_num}"

    def run(self) -> int:
        """
        Generate tracer ID files.

        :return: Exit code.
        """
        # Step 0: create directories
        self._create_directories(
            subdirs=[self.data_subdir],
            force=True,
        )

        # Step 1: Load cluster properties
        cluster_data = halos_daq.get_halo_properties(
            self.config.base_path,
            self.config.snap_num,
            fields=["GroupPos", self.config.radius_field],
            cluster_restrict=True,
        )

        for i, halo_id in enumerate(cluster_data["IDs"]):
            if logging.getLogger().level <= 15:
                logging.info(f"Processing cluster {i + 1}/352.")
            # Step 2: calculate temperature of remaining particles
            logging.debug(f"Halo {i + 1}: Calculating gas temperatures.")
            gas_temperatures = gas_daq.get_cluster_temperature(
                halo_id, self.config.base_path, self.config.snap_num
            )

            # Step 3: get gas particles of this zoom-in region
            logging.debug(f"Halo {i + 1}: Loading gas particle data.")
            fields = ["Coordinates", "ParticleIDs"]
            gas_data = gas_daq.get_gas_properties(
                self.config.base_path,
                self.config.snap_num,
                fields=fields,
                cluster=halo_id,
            )

            # Step 4: restrict particles to those within 2 R_vir
            logging.debug(f"Halo {i + 1}: Restricting to 2 R_vir.")
            center = cluster_data["GroupPos"][i]
            r_vir = cluster_data[self.config.radius_field][i]
            pos = gas_data["Coordinates"]
            mask = np.nonzero(
                np.linalg.norm(pos - center, axis=1) <= 2 * r_vir
            )[0]
            logging.debug(
                f"Halo {i + 1}: Found {mask.size} particles within 2R_vir."
            )
            # restrict temperatures and particle IDs
            pids_cluster = gas_data["ParticleIDs"][mask]
            temperatures_cluster = gas_temperatures[mask]

            # Step 5: restrict particles to those with cool gas
            logging.debug(f"Halo {i + 1}: Restricting to cool gas.")
            selected_pids = pids_cluster[temperatures_cluster <= 10**4.5]
            logging.debug(
                f"Halo {i + 1}: Found {len(selected_pids)} particles with "
                f"required temperature within 2 R_vir."
            )
            # clean-up
            # del gas_temperatures, gas_data, coords, mask

            # Step 6: select tracers in the cool gas particles within 2R_vir
            logging.debug(f"Halo {i + 1}: Loading tracers.")
            tracer_data = tracers_daq.load_tracers(
                self.config.base_path,
                self.config.snap_num,
                cluster_id=halo_id
            )
            logging.debug(
                f"Halo {i + 1}: Restricting tracers to selected particle IDs."
            )
            indices = selection.select_if_in(
                tracer_data["ParentID"],
                selected_pids,
            )
            # small sanity check
            mask = selection.select_if_in(
                gas_data["ParticleIDs"], tracer_data["ParentID"][indices]
            )
            selected_temps = gas_temperatures[mask]
            if not np.all(np.log10(selected_temps) <= 4.5):
                logging.warning(
                    f"Not all selected temperatures are actually valid: max "
                    f"temperature found was 10^"
                    f"{np.log10(np.max(selected_temps))} K."
                )

            # Step 7: save data to file
            filepath = self.paths["data_dir"] / self.data_subdir
            filename = (
                f"tracer_ids_snapshot{self.config.snap_num}"
                f"_cluster_{halo_id}.npz"
            )
            logging.debug(
                f"Halo {i + 1}: Writing data to file under "
                f"{str(filepath / filename)}."
            )
            np.savez(
                filepath / filename,
                tracer_ids=tracer_data["TracerID"][indices],
                parent_ids=tracer_data["ParentID"][indices],
            )
            logging.debug(
                f"Halo {i + 1}: Found {len(indices)} cool gas tracers in "
                f"cluster {halo_id} with PIDs "
                f"{tracer_data['ParentID'][indices]}. Saved them to file."
            )

        return 0


@dataclasses.dataclass
class FindTracedParticleIDsInSnapshot(base.DiagnosticsPipeline):
    """
    Find particles pointed to by tracers saved to file in a snapshot.

    Pipeline loads the tracer IDs of a halo in TNG-Cluster and finds
    the particles that these tracers are in at the given snapshot. It
    then saves their unique particle IDs and their particle type to
    file.
    """

    snap_num: int  # snapshot to query

    def __post_init__(self):
        super().__post_init__()
        self.data_save_subdir = f"particle_ids/snapshot_{self.snap_num:02d}/"

    def run(self) -> int:
        """
        Find particles pointed to by tracers.

        :return: Exit code.
        """
        # Step 0: check directories exist, create new directories
        self._create_directories(subdirs=[self.data_save_subdir], force=True)

        # Step 1: Load cluster data to get IDs
        cluster_data = halos_daq.get_halo_properties(
            self.config.base_path,
            self.config.snap_num,
            fields=[self.config.radius_field],
            cluster_restrict=True,
        )

        for halo_id in cluster_data["IDs"]:
            self._generate_particle_indices(halo_id)

        return 0

    def _generate_particle_indices(self, halo_id: int) -> int:
        """
        Find indices of particles that end up in cool gas for this cluster.

        Function loads the tracer IDs of all tracers in cool gas at
        redshift zero and matches them against tracers in the current
        snapshot. It then correlates these tracers to their parent
        particles and saves the indices of these particles to file,
        such that one can load the indices from file and use them to
        directly select only particles that will end up in cool gas at
        redshift zero.

        The indices will be saved all together, with an additional
        array that identifies their particle type; the final data file
        has two fields:

        - ``particle_ids``
        - ``particle_type``

        These indices are indices into the list of all particles _of
        the zoom-in region of the associated cluster_, i.e. the list of
        particles loaded with
        :func:`~library.data_acquisition.gas_daq.get_gas_properties`.
        Note that to use them on just one type of particles, they will
        have to be reduced to only indices of that type.

        This function can be run in parallel for different clusters.

        .. attention:: This function contains DEBUG level logs. When
            run in parallel, these will not be emitted!

        :param halo_id: The ID of the halo. Must not be the original
            halo ID but the actual halo ID.
        :return: Status of success. Zero for successful execution, one
            for failure to execute.
        """
        if self.processes <= 1:
            logging.debug(f"Processing halo {halo_id}.")
        # Step 1: Load tracer IDs we wish to follow
        filepath = self.paths["data_dir"] / "cool_gas_tracer_ids"
        filename = (
            f"tracer_ids_snapshot{self.config.snap_num}_cluster_"
            f"{halo_id}.npz"
        )
        with np.load(filepath / filename) as data_file:
            selected_tracers = data_file["tracer_ids"]
        if self.processes <= 1:
            logging.debug(
                f"Loaded {len(selected_tracers)} tracer IDs from file."
            )

        # Step 2: load tracers at the current snapshot
        tracer_data = tracers_daq.load_tracers(
            self.config.base_path, self.snap_num, cluster_id=halo_id
        )

        # Step 3: match selected IDs to all IDs, find parent IDs
        tracer_indices = selection.select_if_in(
            tracer_data["TracerID"],
            selected_tracers,
            assume_unique=True,
            assume_subset=True,
        )
        selected_particle_ids = tracer_data["ParentID"][tracer_indices]
        if self.processes <= 1:
            logging.debug(
                f"Selected {len(selected_particle_ids)} particles from list "
                f"of particle IDs."
            )

        # Step 4: match selected parent IDs to particles
        ids, ptypes = self._match_particle_ids_to_particles(
            selected_particle_ids, halo_id
        )

        # Step 5: save particle IDs and type flag to file
        filepath = self.paths["data_dir"] / self.data_save_subdir
        filename = f"particle_ids_halo_{halo_id}.npz"
        np.savez(
            filepath / filename,
            particle_ids=ids,
            particle_type=ptypes,
        )
        if self.processes <= 1:
            logging.debug(f"Saved indices to file {filename}.")
        return 0

    def _match_particle_ids_to_particles(
        self,
        parent_ids: NDArray,
        cluster_id: int,
    ) -> tuple[NDArray, NDArray]:
        """
        Return indices of particles with the given particle IDs.

        Method takes a list of unique particle IDs of gas, star and BH
        particles, and finds the position of them inside the array of
        all particles of their respective type from a zoom-in region.
        This means that the return value of this method is three arrays
        which index the list of all gas, star and BH particles from a
        zoom-in region respectively, such that the particles selected
        have the particle IDs provided.

        :param parent_ids: Array of unique particle IDs to search for.
            Shape (N, ).
        :param cluster_id: ID of the cluster for which to load the
            zoom-in regions particles.
        :return: Three arrays giving the indices into the list of particle
            data for gas cells, star cells, and BH cells respectively,
            such that the particles selected with these indices have
            the particle IDs provided. Shapes (A, ), (B, ), and (C, ),
            such that A + B + C = N. Note that in principle A, B and/or
            C can be zero.
        """
        # gather all particle data for all three types
        particle_ids_list = []
        particle_types_list = []

        # load data and append it to lists
        for part_type in [0, 4, 5]:
            cur_particle_ids = il.snapshot.loadOriginalZoom(
                self.config.base_path,
                self.snap_num,
                cluster_id,
                partType=part_type,
                fields=["ParticleIDs"],
            )
            if cur_particle_ids.size == 0:
                continue  # no particle data available, skip
            particle_ids_list.append(cur_particle_ids)
            # create a list of part type indices
            cur_type_flags = np.empty_like(cur_particle_ids, dtype=np.int8)
            cur_type_flags[:] = part_type
            particle_types_list.append(cur_type_flags)

        # concatenate data and select only desired
        particle_ids = np.concatenate(particle_ids_list, axis=0)
        particle_types = np.concatenate(particle_types_list, axis=0)

        indices = selection.select_if_in(
            particle_ids,
            parent_ids,
            mode="searchsort",
            assume_subset=True,
        )
        np.testing.assert_equal(particle_ids[indices], parent_ids)
        return particle_ids[indices], particle_types[indices]
