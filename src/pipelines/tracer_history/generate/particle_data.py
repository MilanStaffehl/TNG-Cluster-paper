"""
Trace back some simple quantities of the tracer particles.
"""
from __future__ import annotations

import abc
import contextlib
import dataclasses
import logging
from typing import TYPE_CHECKING, ClassVar

import h5py
import illustris_python as il
import numpy as np

from library import constants, units
from library.data_acquisition import gas_daq, halos_daq, particle_daq
from library.processing import membership, parallelization
from pipelines import base

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclasses.dataclass
class TraceSimpleQuantitiesBackABC(base.Pipeline, abc.ABC):
    """
    Base class to trace back simple tracer quantities.

    Needs to have its abstract methods implemented.
    """

    unlink: bool = False  # delete intermediate files after archiving?
    force_overwrite: bool = False  # overwrite intermediate files?
    zoom_id: int | None = None  # process only one zoom-in or all?
    archive_single: bool = False  # archive data even for a single zoom?

    quantity: ClassVar[str] = "unset"  # name of the field in the archive
    n_clusters: ClassVar[int] = 352
    n_snaps: ClassVar[int] = 100 - constants.MIN_SNAP

    def __post_init__(self):
        super().__post_init__()
        self.tmp_dir = (
            self.paths["data_dir"] / "intermediate" / self.quantity.lower()
        )

    def run(self) -> int:
        """
        Trace back quantity and plot it.

        :return: Exit code.
        """
        # Step 0: set up directories, create archive for gas data
        if self.quantity == "unset":
            logging.fatal(
                "Quantity name unset. Does your pipeline implementation "
                "overwrite the class variable `quantity` with a proper name "
                "for the quantity?"
            )
            return 3
        self._create_directories(
            subdirs=[f"intermediate/{self.quantity.lower()}"], force=True
        )
        if self.zoom_id is None:
            logging.info(f"Tracing {self.quantity} of particles back in time.")
        else:
            logging.info(
                f"Tracing {self.quantity} of particles back in time for "
                f"zoom-in {self.zoom_id} only."
            )

        # Step 1: Load cluster primary
        group_primaries = halos_daq.get_halo_properties(
            self.config.base_path,
            self.config.snap_num,
            ["GroupFirstSub"],
            cluster_restrict=True,
        )["GroupFirstSub"]

        # Step 2: Loop through snapshots and zooms to get quantity
        if self.processes > 1:
            self._multiprocess(group_primaries)
        elif self.zoom_id is None:
            # find data for all zoom-ins
            tracer_file = h5py.File(self.config.cool_gas_history, "r")
            for zoom_id in range(self.n_clusters):
                logging.info(f"Processing zoom-in region {zoom_id}.")
                for snap_num in range(constants.MIN_SNAP, 100):
                    self._save_intermediate_file(
                        snap_num,
                        zoom_id,
                        group_primaries[zoom_id],
                        tracer_file,
                    )
            tracer_file.close()
        else:
            # find data only for selected zoom-in
            tracer_file = h5py.File(self.config.cool_gas_history, "r")
            for snap_num in range(constants.MIN_SNAP, 100):
                logging.info(f"Processing snap {snap_num}.")
                self._save_intermediate_file(
                    snap_num,
                    self.zoom_id,
                    group_primaries[self.zoom_id],
                    tracer_file,
                )
            tracer_file.close()

        # Step 3: archive data
        if self.zoom_id is not None and not self.archive_single:
            logging.info(
                "Processed only one zoom, will not attempt to archive data."
            )
            return 0
        logging.info("Starting to archive all created data.")
        tracer_file = h5py.File(self.config.cool_gas_history, "r+")
        if self.zoom_id is None:
            for zoom_id in range(self.n_clusters):
                self._archive_zoom_in(zoom_id, tracer_file)
        else:
            self._archive_zoom_in(self.zoom_id, tracer_file)
        tracer_file.close()

        # Step 4: clean-up
        if not self.unlink:
            return 0  # done, can end pipeline execution
        self._clean_up()

        return 0

    def _save_intermediate_file(
        self,
        snap_num: int,
        zoom_id: int,
        primary_subhalo_id: int,
        tracer_file: h5py.File | None = None,
    ) -> None:
        """
        Load current quantity, select only traced particles, and save to file.

        Function loads the current particle property using the
        ``_load_quantity`` method and then selects from the property
        only those particles that are being traced. The resulting array
        is written to an intermediate temporary file.

        :param snap_num: Snapshot to load from.
        :param zoom_id: Zoom-in region ID to load from.
        :param primary_subhalo_id: ID of the primary subhalo of the
            cluster in the current zoom-in at redshift zero.
        :param tracer_file: Either the opened tracer file archive or
            None. None must be used during multiprocessing to avoid
            concurrency issue in reading. If set to None, the tracer
            file is opened again on every call of the method, which is
            desired for parallel execution, but adds unnecessary
            overhead in sequential execution.
        :return: None.
        """
        # Step 0: skip if file exists
        if not self.force_overwrite:
            filename = (
                f"{self.quantity}z{int(zoom_id):03d}s{int(snap_num):02d}.npy"
            )
            if (self.tmp_dir / filename).exists():
                logging.debug(
                    f"Rewrite was not forced and file {filename} exists; "
                    f"skipping."
                )
                return

        # Step 1: open file if necessary
        if tracer_file is None:
            multiprocessing = True
            tracer_file = h5py.File(self.config.cool_gas_history, "r")
            # coerce type
            snap_num = int(snap_num)
            zoom_id = int(zoom_id)
            primary_subhalo_id = int(primary_subhalo_id)
        else:
            multiprocessing = False
            logging.debug(f"Processing snap {snap_num}, zoom-in {zoom_id}.")

        # Step 2: Get particle data
        part_data = self._load_quantity(snap_num, zoom_id)

        # Step 3: Find gas particle indices
        group = f"ZoomRegion_{zoom_id:03d}"
        indices = tracer_file[f"{group}/particle_indices"][snap_num, :]
        flags = (tracer_file[f"{group}/particle_type_flags"][snap_num, :])

        # Step 4: Create an array for the results
        if len(part_data.shape) > 1:
            shape = indices.shape + part_data.shape[1:]
        else:
            shape = indices.shape
        traced_part_data = np.empty(shape, dtype=part_data.dtype)
        if np.issubdtype(part_data.dtype, np.floating):
            sentinel_value = np.nan
        elif np.issubdtype(part_data.dtype, np.unsignedinteger):
            # we set this to -1 as we require all uint values to be
            # converted to a signed integer type by subclasses
            sentinel_value = -1
        else:
            logging.warning(
                f"Could not assign proper sentinel value for allocation "
                f"of result array of dtype {part_data.dtype}. Setting "
                f"sentinel value to 0 which may cause problems later."
            )
            sentinel_value = 0
        traced_part_data[:] = sentinel_value  # fill with dummy value

        # Step 5: Mask data and fill array with results
        if np.max(indices) > part_data.shape[0]:
            # gas only
            traced_part_data[flags == 0] = part_data[indices[flags == 0]]
        else:
            # all particles available
            traced_part_data[:] = part_data[indices]

        # Step 6: Process particle data into sought quantity
        quantity = self._process_into_quantity(
            zoom_id,
            snap_num,
            traced_part_data,
            primary_subhalo_id,
            tracer_file,
        )

        # Step 6: Save to intermediate file
        filename = f"{self.quantity}z{zoom_id:03d}s{snap_num:02d}.npy"
        np.save(self.tmp_dir / filename, quantity)

        if multiprocessing:
            tracer_file.close()

    def _archive_zoom_in(self, zoom_id: int, tracer_file: h5py.File) -> None:
        """
        Write data for the zoom-in to hdf5 archive from intermediate file.

        :param zoom_id: Zoom-in ID of the zoom-in to archive.
        :return: None.
        """
        logging.debug(f"Archiving zoom-in {zoom_id}.")

        group = f"ZoomRegion_{zoom_id:03d}"
        fn = f"{self.quantity}z{zoom_id:03d}s99.npy"
        test_data = np.load(self.tmp_dir / fn)
        shape = test_data.shape
        dtype = test_data.dtype

        # create a dataset if non-existent
        if self.quantity not in tracer_file[group].keys():
            logging.debug(f"Creating missing dataset for {self.quantity}.")
            tracer_file[group].create_dataset(
                self.quantity, shape=(100, ) + shape, dtype=dtype
            )

        # find appropriate sentinel value
        if np.issubdtype(dtype, np.integer):
            sentinel = -1
        else:
            sentinel = np.nan

        # fill with data from intermediate files
        for snap_num in range(100):
            if snap_num < constants.MIN_SNAP:
                data = np.empty(shape, dtype=dtype)
                data[:] = sentinel
            else:
                fn = f"{self.quantity}z{zoom_id:03d}s{snap_num:02d}.npy"
                data = np.load(self.tmp_dir / fn)
            tracer_file[group][self.quantity][snap_num, :] = data

    def _multiprocess(self, group_primaries: NDArray) -> None:
        """
        Process multiple snapshots and zoom-ins in parallel.

        Method creates arguments for processing all snapshots of one or
        all zoom-ins in parallel (depending on pipeline set-up). The
        individual processes write the data to file, so this method
        returns nothing.

        :param group_primaries: List of primary subhalos IDs of every
            zoom-in.
        :return: None
        """
        # create combinations of args
        if self.zoom_id is None:
            snap_nums = np.arange(
                constants.MIN_SNAP, 100, step=1, dtype=np.uint64
            )
            zoom_ids = np.arange(0, 352, step=1)
            snap_nums = np.broadcast_to(
                snap_nums[:, None],
                (self.n_snaps, 352),
            ).flatten()
            zoom_ids = np.broadcast_to(
                zoom_ids[:, None],
                (352, self.n_snaps),
            ).transpose().flatten()
        else:
            # create data for only one zoom-in
            snap_nums = np.arange(
                constants.MIN_SNAP, 100, step=1, dtype=np.uint64
            )
            zoom_ids = np.empty_like(snap_nums, dtype=np.uint64)
            zoom_ids[:] = self.zoom_id
        # get a list of primaries belonging to each pair of snaps/zooms
        primaries = group_primaries[zoom_ids]
        # run all jobs in parallel
        parallelization.process_data_starmap(
            self._save_intermediate_file,
            self.processes,
            snap_nums,
            zoom_ids,
            primaries,
        )

    def _clean_up(self):
        """
        Clean up temporary intermediate files.

        :return: None
        """
        logging.info("Cleaning up temporary intermediate files.")
        if self.zoom_id is None:
            for file in self.tmp_dir.iterdir():
                file.unlink()
            self.tmp_dir.rmdir()
            logging.info("Successfully cleaned up all intermediate files.")
        else:
            for snap_num in range(constants.MIN_SNAP, 100):
                f = f"{self.quantity}z{self.zoom_id:03d}s{snap_num:02d}.npy"
                with contextlib.suppress(FileNotFoundError):
                    (self.tmp_dir / f).unlink()
            logging.info(
                f"Successfully cleaned up all intermediate files of zoom-in "
                f"{self.zoom_id}."
            )

    @abc.abstractmethod
    def _load_quantity(self, snap_num: int, zoom_id: int) -> NDArray:
        """
        Abstract method to load a cluster quantity.

        Subclasses to this class must implement this method in such a
        way that it returns the array of the quantity that will, together
        with information about subhalos of the cluster, be processed into
        the quantity that will be saved for all gas particles of the
        given zoom at the given snapshot.

        For example, if the distance to a certain type of subhalo is
        required, this should be the coordinates of all gas particles.

        :param snap_num: The snapshot to query. Must be a number between
            0 and 99.
        :param zoom_id: The zoom-in region ID. Must be a number between
            0 and 351.
        :return: The gas quantity for every gas cell in the zoom-in at
            that snapshot, such that it can be indexed by the indices
            saved by the generation pipeline.
        """
        pass

    @abc.abstractmethod
    def _process_into_quantity(
        self,
        zoom_id: int,
        snap_num: int,
        particle_data: NDArray,
        primary_subhalo_id: int,
        data_file: h5py.File,
    ) -> NDArray:
        """
        Abstract method to process particle data.

        Subclasses to this class must implement this method such that it
        can take the particle data loaded by ``_load_quantity``, restricted
        to only the traced particles, and turn it into the sought after
        quantity. For example, if a pipeline should give the distance to
        the primary subhalo, this method must accept particle coordinates
        for the traced particles and load the position of the given
        primary subhalo, then compute the distance and return it as an
        array.

        :param zoom_id: The ID of the zoom-in region at which data is
            processed.
        :param snap_num: The snapshot at which to process the data.
        :param particle_data: The array of particle data acquired from
            ``_load_quantity`` and restricted to only the particles that
            are tracked.
        :param primary_subhalo_id: The ID of the primary subhalo of the
            cluster at redshift 0, i.e. at snapshot 99. Required to load
            the MPB of this subhalo for distance and velocity calculations.
        :param data_file: The opened file containing the gas particle
            data.
        :return: The array of whatever quantity is to be saved to file.
            Must be a 1D array.
        """
        pass


# -----------------------------------------------------------------------------
# CONCRETE CLASSES:


class TraceDistancePipeline(TraceSimpleQuantitiesBackABC):
    """
    Trace distance of all particles to cluster with time.
    """

    quantity: ClassVar[str] = "DistanceToMP"

    def _load_quantity(self, snap_num: int, zoom_id: int) -> NDArray:
        """
        Find the position of every gas particle to the cluster.

        :param snap_num: Snapshot to find the positions at.
        :param zoom_id: The ID of the zoom-in region.
        :return: Array of the positions of all particles.
        """
        positions_list = []
        for part_type in [0, 4, 5]:
            data = particle_daq.get_particle_properties(
                self.config.base_path,
                snap_num,
                part_type=part_type,
                fields=["Coordinates"],
                zoom_id=zoom_id,
            )
            if data["count"] == 0:
                continue  # no particles of this type exist
            positions_list.append(data["Coordinates"])

        # concatenate particle positions
        part_positions = np.concatenate(positions_list, axis=0)
        return part_positions

    def _process_into_quantity(
        self,
        zoom_id: int,
        snap_num: int,
        particle_data: NDArray,
        primary_subhalo_id: int,
        data_file: h5py.File,
    ) -> NDArray:
        """
        Process particle coordinates into distances to primary subhalo.

        :param zoom_id: ID of zoom-in region.
        :param snap_num: Snapshot number.
        :param particle_data: Array of particle coordinates, shape (N, 3).
        :param primary_subhalo_id: ID of the primary subhalo of this
            cluster at redshift 0.
        :param data_file: Dummy parameter.
        :return:
        """
        # Step 1: load subhalo position at this snap
        mpb = il.sublink.loadTree(
            self.config.base_path,
            self.config.snap_num,
            primary_subhalo_id,
            fields=["SubhaloPos", "SnapNum"],
            onlyMPB=True,
        )
        positions = mpb["SubhaloPos"]
        snaps = mpb["SnapNum"]

        # Step 2: convert position into physical units
        try:
            primary_pos_code_units = positions[snaps == snap_num][0]
        except IndexError:
            # snap is not in sublink, have to interpolate
            prev_pos = positions[snaps == snap_num - 1][0]
            next_pos = positions[snaps == snap_num + 1][0]
            primary_pos_code_units = (prev_pos + next_pos) / 2
        primary_pos = units.UnitConverter.convert(
            primary_pos_code_units, "SubhaloPos", snap_num=snap_num
        )

        # Step 3: Calculate distances
        distances = np.linalg.norm(primary_pos - particle_data, axis=1)
        return distances


class TraceTemperaturePipeline(TraceSimpleQuantitiesBackABC):
    """
    Trace temperature of gas particles with time.
    """

    quantity: ClassVar[str] = "Temperature"

    def _load_quantity(self, snap_num: int, zoom_id: int) -> NDArray:
        """
        Find the temperature of gas particles in the zoom-in.

        This loads the temperature of all gas cells in the zoom-in region
        at the given snapshot and returns it, such that it can be indexed
        with the pre-saved indices.

        :param snap_num: The snap for which to load temperatures.
        :param zoom_id: The ID of the zoom-in region.
        :return: Array of the temperatures of all gas cells in the zoom-in
            region.
        """
        return gas_daq.get_cluster_temperature(
            self.config.base_path,
            snap_num,
            zoom_id=zoom_id,
        )

    def _process_into_quantity(
        self,
        zoom_id: int,
        snap_num: int,
        particle_data: NDArray,
        primary_subhalo_id: int,
        data_file: h5py.File,
    ) -> NDArray:
        """
        Return temperatures as-is, no processing required.

        :param zoom_id: Dummy parameter.
        :param snap_num: Dummy parameter.
        :param particle_data: Array of particle temperatures.
        :param primary_subhalo_id: Dummy parameter.
        :param data_file: Dummy parameter.
        :return: Array of particle temperatures.
        """
        return particle_data


class TraceDensityPipeline(TraceSimpleQuantitiesBackABC):
    """
    Trace gas density of gas particles with time.
    """

    quantity: ClassVar[str] = "Density"

    def _load_quantity(self, snap_num: int, zoom_id: int) -> NDArray:
        """
        Load the density of all gas cells in the zoom-in.

        Loads only the density of the gas cells, even if there is a
        density field available for black holes.

        :param snap_num: The snap for which to load densities.
        :param zoom_id: The ID of the zoom-in region.
        :return: Array of the density of all gas cells in the zoom-in
            region.
        """
        gas_data = gas_daq.get_gas_properties(
            self.config.base_path,
            snap_num,
            fields=["Density"],
            zoom_id=zoom_id,
        )
        return gas_data["Density"]

    def _process_into_quantity(
        self,
        zoom_id: int,
        snap_num: int,
        particle_data: NDArray,
        primary_subhalo_id: int,
        data_file: h5py.File,
    ) -> NDArray:
        """
        Return density as-is, no processing required.

        :param zoom_id: Dummy parameter.
        :param snap_num: Dummy parameter.
        :param particle_data: Array of particle densities.
        :param primary_subhalo_id: Dummy parameter.
        :param data_file: Dummy parameter.
        :return: Array of particle densities.
        """
        return particle_data


class TraceMassPipeline(TraceSimpleQuantitiesBackABC):
    """
    Trace particle mass of all particles with time.
    """

    quantity: ClassVar[str] = "Mass"

    def _load_quantity(self, snap_num: int, zoom_id: int) -> NDArray:
        """
        Load the mass of all particles in the zoom-in.

        :param snap_num: Snapshot at which to load the particle mass.
        :param zoom_id: ID of the zoom-in from which to load mass.
        :return: Array of particle mass for all particles, in order or
            particle type (i.e. type 0, 4, and 5 in that order).
        """
        masses_list = []
        for part_type in [0, 4, 5]:
            data = particle_daq.get_particle_properties(
                self.config.base_path,
                snap_num,
                part_type=part_type,
                fields=["Masses"],
                zoom_id=zoom_id,
            )
            if data["count"] == 0:
                continue  # no particles of this type exist
            masses_list.append(data["Masses"])

        # concatenate particle positions
        part_masses = np.concatenate(masses_list, axis=0)
        return part_masses

    def _process_into_quantity(
        self,
        zoom_id: int,
        snap_num: int,
        particle_data: NDArray,
        primary_subhalo_id: int,
        data_file: h5py.File,
    ) -> NDArray:
        """
        Return masses as-is, no processing required.

        :param zoom_id: Dummy parameter.
        :param snap_num: Dummy parameter.
        :param particle_data: Array of particle masses.
        :param primary_subhalo_id: Dummy parameter.
        :param data_file: Dummy parameter.
        :return: Array of particle masses.
        """
        return particle_data


class TraceParticleParentHaloPipeline(TraceSimpleQuantitiesBackABC):
    """
    Trace particle parent halo indices back in time.
    """

    quantity: ClassVar[str] = "ParentHaloIndex"

    def _load_quantity(self, snap_num: int, zoom_id: int) -> NDArray:
        """
        Load the particle IDs of all particles in the zoom-in.

        :param snap_num: Snapshot at which to load the particles.
        :param zoom_id: ID of the zoom-in from which to load particle IDs.
        :return: Array of particle IDs for all particles, in order of
            particle type.
        """
        pids_list = []
        for part_type in [0, 4, 5]:
            cur_pids = particle_daq.get_particle_ids(
                self.config.base_path,
                snap_num,
                part_type=part_type,
                zoom_id=zoom_id,
            )
            if cur_pids.size == 0:
                continue  # no particles of this type exist
            pids_list.append(cur_pids)

        # concatenate particle positions
        pids = np.concatenate(pids_list, axis=0)
        return pids

    def _process_into_quantity(
        self,
        zoom_id: int,
        snap_num: int,
        particle_data: NDArray,
        primary_subhalo_id: int,
        data_file: h5py.File,
    ) -> NDArray:
        """
        Process particle IDs of traced particles into parent halo indices.

        :param zoom_id: ID of the zoom-in region to process.
        :param snap_num: The snapshot at which to find parents.
        :param particle_data: The array of particle IDs of all traced
            particles at this snapshot and zoom-in.
        :param primary_subhalo_id: Dummy parameter.
        :return: Array of parent halo indices.
        """
        field = f"ZoomRegion_{zoom_id:03d}/particle_type_flags"
        part_types = data_file[field][snap_num, :]
        halo_indices, _ = membership.particle_parents(
            particle_data, part_types, snap_num, self.config.base_path
        )
        return halo_indices


class TraceParticleParentSubhaloPipeline(TraceParticleParentHaloPipeline):
    """
    Trace particle parent subhalo index back in time.
    """

    quantity: ClassVar[str] = "ParentSubhaloIndex"

    def _process_into_quantity(
        self,
        zoom_id: int,
        snap_num: int,
        particle_data: NDArray,
        primary_subhalo_id: int,
        data_file: h5py.File,
    ) -> NDArray:
        """
        Process particle IDs of traced particles into parent subhalo indices.

        :param zoom_id: ID of the zoom-in region to process.
        :param snap_num: The snapshot at which to find parents.
        :param particle_data: The array of particle IDs of all traced
            particles at this snapshot and zoom-in.
        :param primary_subhalo_id: Dummy parameter.
        :return: Array of parent subhalo indices.
        """
        field = f"ZoomRegion_{zoom_id:03d}/particle_type_flags"
        part_types = data_file[field][snap_num, :]
        _, subhalo_indices = membership.particle_parents(
            particle_data, part_types, snap_num, self.config.base_path
        )
        return subhalo_indices
