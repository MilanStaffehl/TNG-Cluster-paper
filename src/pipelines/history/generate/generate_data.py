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

    def run(self) -> int:
        """
        Generate tracer ID files.

        :return: Exit code.
        """
        # Step 0: create directories
        self._create_directories(subdirs=["cool_gas_tracer_ids"], force=True)

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
            filepath = self.paths["data_dir"] / "cool_gas_tracer_ids"
            filename = f"tracer_ids_snapshot99_cluster_{halo_id}.npz"
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
                f"{tracer_data['ParentID'][indices]} Saved them to file."
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
        self.data_save_subdir = f"particle_ids/snapshot_{self.snap_num:02d}"

    def run(self) -> int:
        """
        Find particles pointed to by tracers.

        :return: Exit code.
        """
        # Step 0: check directories exist, create new directories
        self._create_directories(subdirs=[self.data_save_subdir])

        # Step 1: Load cluster data to get IDs
        cluster_data = halos_daq.get_halo_properties(
            self.config.base_path,
            self.config.snap_num,
            fields=[self.config.radius_field,
                    "GroupPos"],  # arbitrary dummy field
            cluster_restrict=True,
        )

        for i, halo_id in enumerate(cluster_data["IDs"]):
            logging.info(f"Processing halo {i + 1}/352.")
            # Step 2: Load tracer IDs we wish to follow
            filepath = self.paths["data_dir"] / "cool_gas_tracer_ids"
            filename = f"tracer_ids_snapshot99_cluster_{halo_id}.npz"
            with np.load(filepath / filename) as data_file:
                selected_tracers = data_file["tracer_ids"]
            logging.debug(
                f"Loaded {len(selected_tracers)} tracer IDs from file."
            )

            # Step 3: load tracers at the current snapshot
            tracer_data = tracers_daq.load_tracers(
                self.config.base_path, self.snap_num, cluster_id=halo_id
            )
            logging.debug(f"Loaded tracer data of halo {halo_id}.")

            # Step 4: match selected IDs to all IDs, find parent IDs
            tracer_indices = selection.select_if_in(
                tracer_data["TracerID"],
                selected_tracers,
                assume_unique=True,
                assume_subset=True,
            )
            selected_particle_ids = tracer_data["ParentID"][tracer_indices]
            logging.debug(
                f"Selected {len(selected_particle_ids)} particles from list "
                f"of particle IDs."
            )

            # Step 5: match selected parent IDs to particles
            logging.debug("Begin matching PIDs to particles.")
            ids = self._match_particle_ids_to_particles(
                selected_particle_ids, halo_id
            )
            logging.debug("Found indices of particles.")

            # TODO: remove me! Check: load particle temperature and position
            #  and verify that they meet the requirement at snapshot 99
            if self.snap_num == 99:
                gas_temperatures = gas_daq.get_cluster_temperature(
                    halo_id, self.config.base_path, self.snap_num
                )
                if not np.all(gas_temperatures[ids[0]] <= 10**4.5):
                    logging.warning(
                        f"Not all selected gas temperatures are valid: "
                        f"Highest gas temperature found was 10^"
                        f"{np.max(np.log10(gas_temperatures[ids[0]]))} K."
                    )
                else:
                    logging.info("Selected temperatures OK.")
                gas_coords = gas_daq.get_gas_properties(
                    self.config.base_path,
                    self.snap_num,
                    fields=["Coordinates"],
                    cluster=halo_id,
                )
                center = cluster_data["GroupPos"][i]
                r_vir = cluster_data[self.config.radius_field][i]
                distances = np.linalg.norm(
                    gas_coords["Coordinates"][ids[0]] - center, axis=1
                )
                if not np.all(distances <= 2 * r_vir):
                    logging.warning(
                        f"Not all selected gas particles are within 2 R_vir: "
                        f"Furthest distance was {np.max(distances) / r_vir} "
                        f"R_vir."
                    )
                else:
                    logging.info("Selected distances OK.")
                continue

            # Step 6: save particle IDs and type flag to file
            filepath = self.paths["data_dir"] / self.data_save_subdir
            filename = f"particle_ids_halo_{halo_id}.npz"
            np.savez(
                filepath / filename,
                particle_ids_gas=ids[0],
                particle_ids_stars=ids[1],
                particle_ids_bhs=ids[2],
            )

        return 0

    def _match_particle_ids_to_particles(
        self,
        parent_ids: NDArray,
        cluster_id: int,
    ) -> list[NDArray, NDArray, NDArray]:
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
        :param cluster_id: IDof the cluster for which to load the zoom-in
            regions particles.
        :return: Three arrays giving the indices into the list of particle
            data for gas cells, star cells, and BH cells respectively,
            such that the particles selected with these indices have
            the particle IDs provided. Shapes (A, ), (B, ), and (C, ),
            such that A + B + C = N.
        """
        indices = []
        for part_type in [0, 4, 5]:
            cur_particle_ids = il.snapshot.loadOriginalZoom(
                self.config.base_path,
                self.snap_num,
                cluster_id,
                partType=part_type,
                fields=["ParticleIDs"],
            )
            # match particle IDs
            cur_indices = selection.select_if_in(
                cur_particle_ids,
                parent_ids,
                mode="searchsort",
                assume_subset=False
            )
            indices.append(cur_indices)
        return indices
