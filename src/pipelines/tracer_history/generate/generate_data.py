"""
Pipeline to generate the data required to work with tracers.
"""
from __future__ import annotations

import dataclasses
import logging
from typing import TYPE_CHECKING, ClassVar

import h5py
import numpy as np

from library.data_acquisition import gas_daq, halos_daq, particle_daq, tracers_daq
from library.processing import parallelization, selection
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
                self.config.base_path, self.config.snap_num, halo_id
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

        # Step 2: For every cluster, find the corresponding tracers
        n_halos = len(cluster_data["IDs"])
        logging.debug(
            f"Will process {n_halos} halos in {self.processes} processes."
        )
        # depending on number of processes, use multiprocessing or not:
        if self.processes > 1:
            parallelization.process_data_starmap(
                self._generate_particle_indices,
                self.processes,
                cluster_data["IDs"],
                np.linspace(0, n_halos - 1, n_halos),
                chunksize=1,  # one halo per process
            )
        else:
            for i, halo_id in enumerate(cluster_data["IDs"]):
                self._generate_particle_indices(halo_id, i)

        logging.info(
            f"Completed job! Found particle indices for snapshot "
            f"{self.snap_num} and saved them to file."
        )
        return 0

    def _generate_particle_indices(
        self, halo_id_at_zero: int, zoom_id: int
    ) -> int:
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
        has two fields:selected

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

        :param halo_id_at_zero: The halo ID of the current halo at
            redshift zero, i.e. in snapshot 99.
        :param zoom_id: The ID of the zoom-in region.
        :return: Status of success. Zero for successful execution, one
            for failure to execute.
        """
        # ensure correct type (needed for multi-processing)
        if not isinstance(halo_id_at_zero, int):
            halo_id_at_zero = int(halo_id_at_zero)
        if not isinstance(zoom_id, int):
            zoom_id = int(zoom_id)

        if self.processes <= 1:
            logging.debug(f"Processing zoom-in {zoom_id}/352.")
        # Step 1: Load tracer IDs we wish to follow
        filepath = (
            self.paths["data_dir"]
            / f"cool_gas_tracer_ids_{self.config.snap_num}"
        )
        filename = (
            f"tracer_ids_snapshot{self.config.snap_num}_cluster_"
            f"{halo_id_at_zero}.npz"
        )
        with np.load(filepath / filename) as data_file:
            selected_tracers = data_file["tracer_ids"]
        if self.processes <= 1:
            logging.debug(
                f"Loaded {len(selected_tracers)} tracer IDs from file."
            )

        # Step 2: load tracers at the current snapshot
        tracer_data = tracers_daq.load_tracers(
            self.config.base_path, self.snap_num, zoom_id=zoom_id
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
        indices, ptypes, lens = self._match_particle_ids_to_particles(
            selected_particle_ids, zoom_id
        )
        if self.processes <= 1:
            logging.debug(
                f"Found {len(indices)} particles matching the "
                f"{len(selected_particle_ids)} tracer parent IDs."
            )

        # Step 5: save particle IDs and type flag to file
        filepath = self.paths["data_dir"] / self.data_save_subdir
        filename = f"particle_ids_zoom_region_{zoom_id}.npz"
        np.savez(
            filepath / filename,
            particle_indices=indices,
            particle_type=ptypes,
            total_part_len=lens,
        )
        if self.processes <= 1:
            logging.debug(f"Saved indices to file '{filename}'.")
        return 0

    def _match_particle_ids_to_particles(
        self,
        parent_ids: NDArray,
        zoom_id: int,
    ) -> tuple[NDArray, NDArray, NDArray]:
        """
        Return indices of particles with the given particle IDs.

        Method takes a list of unique particle IDs of gas, star and BH
        particles, and finds the position of them inside the array of
        all particles of their respective type from a zoom-in region.
        This means that the return value of this method is two arrays,
        the first of which indexes the list of all gas, star and
        BH particles from a zoom-in region when concatenated into one
        array of that order, while the second aray gives the particle
        type as an integer flag (0 for gas, 4 for stars and wind, 5 for
        black holes) so that the indices can be restricted to only one
        particle type.

        :param parent_ids: Array of unique particle IDs to search for.
            Shape (N, ).
        :param zoom_id: ID of the zoom-in region for which to load the
            particles.
        :return: Three arrays, the first giving the indices into the list
            of particle data for gas cells, star cells, and BH cells
            when concatenated in that order, the second one giving the
            particle type as an integer for every corresponding index.
            This allows filtering the indices to only one particle type.
            The third array is the number of particles of type 0, 4, and
            5 in the zoom-region respectively. This is useful to convert
            indices into the concatenated list of all particles into
            indices of only one type.
        """
        # gather all particle data for all three types
        particle_ids_list = []
        particle_types_list = []
        particle_len_list = []

        # load data and append it to lists
        for part_type in [0, 4, 5]:
            cur_particle_ids = particle_daq.get_particle_ids(
                self.config.base_path,
                self.snap_num,
                part_type=part_type,
                zoom_id=zoom_id,
            )
            particle_len_list.append(cur_particle_ids.size)
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
        particle_len = np.array(particle_len_list, dtype=np.uint64)

        indices = selection.select_if_in(
            particle_ids,
            parent_ids,
            mode="searchsort",
            assume_subset=True,
        )
        np.testing.assert_equal(particle_ids[indices], parent_ids)
        return indices, particle_types[indices], particle_len


@dataclasses.dataclass
class ArchiveTNGClusterTracerDataPipeline(base.Pipeline):
    """
    Write created tracer data to common hdf5 archive and clean up.

    This pipeline finds all tracer data files and places them into a
    single, common hdf5 file and deletes the files it read afterward.
    """

    unlink: bool = False
    n_clusters: ClassVar[int] = 352

    def __post_init__(self):
        super().__post_init__()
        self.filepath = (
            self.paths["data_dir"] / "particle_ids" / "TNG_Cluster"
            / f"particle_ids_from_snapshot_{self.config.snap_num}.hdf5"
        )

    def run(self) -> int:
        """
        Read tracer data and write it to hdf5 file.

        :return: Exit code.
        """
        # Step 0: check all files and directories exist
        self._create_directories(
            subdirs=["particle_ids/TNG_Cluster"], force=True
        )

        logging.info("Checking all required intermediate files exist.")
        files_exist = True
        for snap_num in range(100):
            for zoom_id in range(self.n_clusters):
                data_subdir = f"particle_ids/snapshot_{snap_num:02d}/"
                filepath = self.paths["data_dir"] / data_subdir
                filename = f"particle_ids_zoom_region_{zoom_id}.npz"
                if not (filepath / filename).exists():
                    files_exist = False
                    logging.error(f"Missing file {filepath}/{filename}.")
        if not files_exist:
            logging.fatal(
                "Not all required tracer data files were found. Aborting "
                "execution."
            )
            return 1

        # Step 1: Create a hdf5 archive and its structure from snap 99
        logging.info("Creating hdf5 archive.")
        f = h5py.File(self.filepath, "w")
        snapdir_99 = self.paths["data_dir"] / "particle_ids/snapshot_99/"
        for zoom_id in range(self.n_clusters):
            # create hdf5 group for current zoom-in region
            grp = f.create_group(f"ZoomRegion_{zoom_id:03d}")

            # load data from .npz archive
            npz_filename = f"particle_ids_zoom_region_{zoom_id}.npz"
            with np.load(snapdir_99 / npz_filename, "r") as orig_data:
                indices = orig_data["particle_indices"]
                type_flags = orig_data["particle_type"]
                part_num = orig_data["total_part_len"]

            # write data to the hdf5 file
            grp.create_dataset(
                "particle_indices",
                shape=(100, indices.size),
                dtype=indices.dtype,
            )
            f[f"{grp.name}/particle_indices"][99, :] = indices
            grp.create_dataset(
                "particle_type_flags",
                shape=(100, type_flags.size),
                dtype=type_flags.dtype,
            )
            f[f"{grp.name}/particle_type_flags"][99, :] = type_flags
            grp.create_dataset(
                "total_particle_num",
                shape=(100, 3),
                dtype=part_num.dtype,
            )
            f[f"{grp.name}/total_particle_num"][99, :] = part_num

            # clean-up
            if self.unlink:
                (snapdir_99 / npz_filename).unlink()  # delete old archive

        # Step 2: go through archives and add them to the file
        for snap_num in range(99):
            logging.info(f"Archiving data from snapshot {snap_num}.")
            snapdir = (
                self.paths["data_dir"]
                / f"particle_ids/snapshot_{snap_num:02d}/"
            )
            for zoom_id in range(self.n_clusters):
                # load data from .npz archive
                npz_filename = f"particle_ids_zoom_region_{zoom_id}.npz"
                with np.load(snapdir / npz_filename, "r") as orig_data:
                    indices = orig_data["particle_indices"]
                    type_flags = orig_data["particle_type"]
                    part_num = orig_data["total_part_len"]

                # write data to the hdf5 file
                group = f"ZoomRegion_{zoom_id:03d}"
                f[f"{group}/particle_indices"][snap_num, :] = indices
                f[f"{group}/particle_type_flags"][snap_num, :] = type_flags
                f[f"{group}/total_particle_num"][snap_num, :] = part_num

                # clean-up
                if self.unlink:
                    logging.debug(
                        f"Unlinking intermediate file "
                        f"{snapdir / npz_filename}."
                    )
                    (snapdir / npz_filename).unlink()  # delete old archive

            # clean-up (hopefully) empty dir
            if self.unlink:
                logging.debug(f"Cleaning up empty directory {snapdir}.")
                try:
                    snapdir.rmdir()
                except OSError:
                    logging.error(
                        f"Unable to remove directory {snapdir}: Directory is "
                        f"not empty. Manual clean-up is required."
                    )

        # Step 3: close file
        f.close()
        logging.info(f"Successfully wrote all data to file {self.filepath}!")
        return 0


class TestArchivedTracerDataTNGClusterPipeline(
        ArchiveTNGClusterTracerDataPipeline):
    """
    Pipeline tests that tracer data saved to file is self-consistent.

    Pipeline loads the particle IDs of all particles in every zoom-in
    region in every snapshot and compares it to the parent IDs of all
    tracers. It then checks that the correspond tracer IDs of these
    matching tracers never change across snapshots. Any failures in
    this check are logged with ``WARNING`` level.

    Pipeline exits with the number of failed snapshot checks.
    """

    def __post_init__(self):
        super().__post_init__()

    def run(self) -> int:
        """
        Test tracer data saved to file is self-consistent.

        :return: Exit code (number of failed snapshot checks)
        """
        failed = 0

        # Step 0: get cluster IDs
        cluster_data = halos_daq.get_halo_properties(
            self.config.base_path,
            self.config.snap_num,
            fields=[self.config.mass_field],
            cluster_restrict=True,
        )

        # Step 1: open hdf5 file
        f = h5py.File(self.filepath, "r")

        # Step 2: Loop over zoom-in regions
        for zoom_id in range(self.n_clusters):
            logging.info(f"Checking zoom-in region {zoom_id}.")

            # Step 2.1: load tracer IDs at redshift zero
            halo_id = cluster_data["IDs"][zoom_id]
            path = f"cool_gas_tracer_ids_{self.config.snap_num}"
            file = f"tracer_ids_snapshot99_cluster_{halo_id}.npz"
            with np.load(self.paths["data_dir"] / path / file) as data_file:
                tracer_ids = data_file["tracer_ids"]

            # Step 2.2: loop over snapshots
            for snap_num in reversed(range(100)):
                # Get particle IDs at current snap
                pids = self._get_particle_ids(zoom_id, snap_num)

                # Get tracer data
                tracer_data = tracers_daq.load_tracers(
                    self.config.base_path, snap_num, zoom_id=zoom_id
                )

                # Get indices into particles
                dataset = f"ZoomRegion_{zoom_id:03d}/particle_indices"
                indices = f[dataset][snap_num, :]

                # mask particle IDs to only selected
                selected_pids = pids[indices]

                # match selected indices to tracer parent IDs
                tracer_indices = selection.select_if_in(
                    tracer_data["ParentID"],
                    selected_pids,
                    mode="searchsort",
                    assume_unique=False,
                    assume_subset=False,
                )

                # mask tracer IDs to only those with matching parent IDs
                current_tracer_ids = tracer_data["TracerID"][tracer_indices]

                # check they match
                if not np.array_equal(current_tracer_ids, tracer_ids):
                    logging.warning(
                        f"Tracers of zoom-in {zoom_id} at snapshot {snap_num} "
                        f"do not match the original tracer IDs."
                    )
                    logging.debug(
                        f"Number of differences: "
                        f"{np.count_nonzero(tracer_ids - current_tracer_ids)}"
                        f"\nDifferences: {tracer_ids - current_tracer_ids}"
                    )
                    failed += 1

                if failed > 1:
                    break
            if failed > 1:
                break

        return failed

    def _get_particle_ids(self, zoom_id: int, snap_num: int) -> NDArray:
        """Load contiguous array of particle IDs for gas, stars, and BHs"""
        # gather all particle data for all three types
        particle_ids_list = []

        # load data and append it to lists
        for part_type in [0, 4, 5]:
            cur_particle_ids = particle_daq.get_particle_ids(
                self.config.base_path,
                snap_num=snap_num,
                part_type=part_type,
                zoom_id=zoom_id,
            )
            if cur_particle_ids.size == 0:
                continue  # no particle data available, skip
            particle_ids_list.append(cur_particle_ids)

        # concatenate data and select only desired
        particle_ids = np.concatenate(particle_ids_list, axis=0)
        return particle_ids
