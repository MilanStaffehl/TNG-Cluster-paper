"""
Trace back some simple quantities of the tracer particles.
"""
from __future__ import annotations

import abc
import contextlib
import dataclasses
import logging
from typing import TYPE_CHECKING, ClassVar, Protocol

import h5py
import illustris_python as il
import multiprocess as mp
import numpy as np

from library import compute, constants, units
from library.data_acquisition import gas_daq, halos_daq, particle_daq
from library.processing import parallelization
from pipelines import base

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray


class TracePipelineProtocol(Protocol):
    """Dummy protocol to make mixin classes work without complaints."""

    @property
    def quantity(self) -> str:
        ...

    @property
    def tmp_dir(self) -> Path:
        ...

    @property
    def zoom_id(self) -> int | None:
        ...


class ArchiveMixin:
    """
    Mixin to provide methods for archiving data.
    """

    def _archive_zoom_in(
        self: TracePipelineProtocol, zoom_id: int, tracer_file: h5py.File
    ) -> None:
        """
        Write data for the zoom-in to hdf5 archive from intermediate file.

        :param zoom_id: Zoom-in ID of the zoom-in to archive.
        :return: None.
        """
        logging.debug(f"Archiving zoom-in {zoom_id}.")

        group = f"ZoomRegion_{zoom_id:03d}"
        fn = f"{self.quantity}_z{zoom_id:03d}s99.npy"
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
                fn = f"{self.quantity}_z{zoom_id:03d}s{snap_num:02d}.npy"
                data = np.load(self.tmp_dir / fn)
            tracer_file[group][self.quantity][snap_num, :] = data

    def _clean_up(self: TracePipelineProtocol):
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


@dataclasses.dataclass
class TraceSimpleQuantitiesBackABC(base.Pipeline, ArchiveMixin, abc.ABC):
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

        # Step 1: Loop through snapshots and zooms to get quantity
        if self.processes > 1:
            self._multiprocess()
        elif self.zoom_id is None:
            # find data for all zoom-ins
            tracer_file = h5py.File(self.config.cool_gas_history, "r")
            for zoom_id in range(self.n_clusters):
                logging.info(f"Processing zoom-in region {zoom_id}.")
                for snap_num in range(constants.MIN_SNAP, 100):
                    self._save_intermediate_file(
                        snap_num,
                        zoom_id,
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
                    tracer_file,
                )
            tracer_file.close()

        # Step 2: archive data
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

        # Step 3: clean-up
        if not self.unlink:
            return 0  # done, can end pipeline execution
        self._clean_up()

        return 0

    def _save_intermediate_file(
        self,
        snap_num: int,
        zoom_id: int,
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
                f"{self.quantity}_z{int(zoom_id):03d}s{int(snap_num):02d}.npy"
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
        else:
            multiprocessing = False
            logging.debug(f"Processing snap {snap_num}, zoom-in {zoom_id}.")

        # Step 2: Get particle data
        part_data = self._load_quantity(snap_num, zoom_id)

        # Step 3: Find gas particle indices
        group = f"ZoomRegion_{zoom_id:03d}"
        indices = tracer_file[f"{group}/particle_indices"][snap_num, :]
        flags = tracer_file[f"{group}/particle_type_flags"][snap_num, :]

        # Step 4: Create an array for the results
        if len(part_data.shape) > 1:
            shape = indices.shape + part_data.shape[1:]
        else:
            shape = indices.shape
        quantity = np.empty(shape, dtype=part_data.dtype)
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
        quantity[:] = sentinel_value  # fill with dummy value

        # Step 5: Mask data and fill array with results
        if np.max(indices) > part_data.shape[0]:
            # gas only
            quantity[flags == 0] = part_data[indices[flags == 0]]
        else:
            # all particles available
            quantity[:] = part_data[indices]

        # Step 6: Save to intermediate file
        filename = f"{self.quantity}_z{zoom_id:03d}s{snap_num:02d}.npy"
        np.save(self.tmp_dir / filename, quantity)

        if multiprocessing:
            tracer_file.close()

    def _multiprocess(self) -> None:
        """
        Process multiple snapshots and zoom-ins in parallel.

        Method creates arguments for processing all snapshots of one or
        all zoom-ins in parallel (depending on pipeline set-up). The
        individual processes write the data to file, so this method
        returns nothing.

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
        # run all jobs in parallel
        parallelization.process_data_starmap(
            self._save_intermediate_file,
            self.processes,
            snap_nums,
            zoom_ids,
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


# -----------------------------------------------------------------------------
# CONCRETE CLASSES:


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


# -----------------------------------------------------------------------------
# COMPLEX QUANTITIES


@dataclasses.dataclass
class TraceDistancePipeline(base.DiagnosticsPipeline, ArchiveMixin):
    """
    Trace distance to main progenitor back in time.
    """

    unlink: bool = False  # delete intermediate files after archiving?
    force_overwrite: bool = False  # overwrite intermediate files?
    zoom_id: int | None = None  # process only one zoom-in or all?
    archive_single: bool = False  # archive data even for a single zoom?

    quantity: ClassVar[str] = "DistanceToMP"
    n_clusters: ClassVar[int] = 352
    n_snaps: ClassVar[int] = 100 - constants.MIN_SNAP

    def __post_init__(self):
        super().__post_init__()
        self.tmp_dir = (
            self.paths["data_dir"] / "intermediate" / self.quantity.lower()
        )

    def run(self) -> int:
        """
        Trace the distance of every particle to the main progenitor
        with time.

        :return: Exit code.
        """
        # Step 0: prepare directories
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

        # TODO: move to implementation of seq/multiproc
        # Step 1: Load cluster primary subhalo IDs
        group_primaries = halos_daq.get_halo_properties(
            self.config.base_path,
            self.config.snap_num,
            ["GroupFirstSub"],
            cluster_restrict=True,
        )["GroupFirstSub"]

        # Step 2: process data:
        if self.processes > 1:
            self._multiprocess(group_primaries)
        elif self.zoom_id is None:
            self._sequential(group_primaries)
        else:
            self._sequential_single(group_primaries[self.zoom_id])

        # Step 3: archive data
        archive_file = h5py.File(self.config.cool_gas_history, "r+")
        if self.zoom_id is None:
            for zoom_id in range(self.n_clusters):
                self._archive_zoom_in(zoom_id, archive_file)
        elif self.archive_single:
            self._archive_zoom_in(self.zoom_id, archive_file)
        else:
            logging.info("Was instructed to not archive data.")

        # Step 4: clean-up
        if not self.unlink:
            return 0
        self._clean_up()

        return 0

    def _sequential(self, group_primaries: NDArray[np.integer]) -> None:
        """
        Find distance to MBP of every particle sequentially.

        :param group_primaries: List of primary subhalo IDs at snapshot
            99, i.e. redshift zero.
        :return: None, saves intermediate results to file.
        """
        logging.debug("Started processing clusters sequentially.")
        archive_file = h5py.File(self.config.cool_gas_history, "r")
        for zoom_id in range(self.n_clusters):
            logging.info(f"Processing zoom-in {zoom_id}.")
            # Step 1: get primary positions for this zoom-id
            primary_positions = self._get_main_progenitor_positions(
                group_primaries[zoom_id], zoom_id
            )

            # Step 2: get particle indices
            dataset = f"ZoomRegion_{zoom_id:03d}/particle_indices"
            particle_indices = archive_file[dataset][()]

            # Step 3: loop over snapshots
            for snap_num in range(constants.MIN_SNAP, 100):
                logging.debug(
                    f"Processing zoom-in {zoom_id}, snap {snap_num}."
                )
                index = snap_num - constants.MIN_SNAP
                self._save_particle_distances(
                    zoom_id,
                    snap_num,
                    primary_positions[index],
                    particle_indices
                )
        archive_file.close()

    def _sequential_single(self, primary_subhalo_id: int) -> None:
        """
        Find distance to MBP of every particle for one zoom-in sequentially.

        :param primary_subhalo_id: ID of primary subhalo at snapshot
            99, i.e. redshift zero.
        :return: None, saves intermediate results to file.
        """
        logging.debug(f"Processing zoom-in {self.zoom_id} sequentially.")
        archive_file = h5py.File(self.config.cool_gas_history, "r")
        # Step 1: get primary positions for this zoom-id
        primary_positions = self._get_main_progenitor_positions(
            primary_subhalo_id, self.zoom_id
        )

        # Step 2: get particle indices
        dataset = f"ZoomRegion_{self.zoom_id:03d}/particle_indices"
        particle_indices = archive_file[dataset][()]

        # Step 3: loop over snapshots
        for snap_num in range(constants.MIN_SNAP, 100):
            logging.debug(
                f"Processing zoom-in {self.zoom_id}, snap {snap_num}."
            )
            index = snap_num - constants.MIN_SNAP
            self._save_particle_distances(
                self.zoom_id,
                snap_num,
                primary_positions[index],
                particle_indices
            )

        archive_file.close()

    def _multiprocess(self, group_primaries: NDArray[np.integer]) -> None:
        """
        Find distance to MBP using multiple processes.

        Method must first load all required data to reduce the load on
        the file system during parallel execution, which can take quite
        some time.

        :param group_primaries: List of primary subhalo IDs at snapshot
            99, i.e. redshift zero.
        :return: None, saves intermediate results to file.
        """
        logging.info("Start preparing args for multiprocessing.")
        if self.zoom_id is None:
            args = self._prepare_multiproc_args(group_primaries)
        else:
            args = self._prepare_multiproc_args_single(
                group_primaries[self.zoom_id]
            )

        # open a pool for all arguments
        chunksize = round(len(args) / self.processes / 4, -2)
        chunksize = max(chunksize, 1)
        logging.info(
            f"Starting {self.processes} processes with auto-determined "
            f"chunksize {chunksize} to find distances. This will take a "
            f"while..."
        )
        with mp.Pool(processes=self.processes) as pool:
            pool.starmap(
                self._save_particle_distances, args, chunksize=int(chunksize)
            )
            pool.close()
            pool.join()
        logging.info("Finished calculating distance for all particles!")

    def _prepare_multiproc_args(
        self, group_primaries: NDArray
    ) -> list[tuple[NDArray | int]]:
        """
        Load all info required for multiprocessing and arrange it.

        The method creates all possible combinations of zoom-in ID and
        snapshot number, and adds to each pair of zoom-in Id and snapshot
        number the corresponding primary subhalo position and the array
        of particle indices pointing to the traced particles.

        :param group_primaries: List of IDs of primary subhalo for every
            zoom-in at redshift zero.
        :return: List of tuples, containing zoom-in ID, snapshot number
            and the corresponding MBP primary position and list of traced
            particle indices.
        """
        # We must construct a list of tuples, containing zoom-ID, snap
        # num, group primary position at that snap for that zoom-in, and
        # the array of particle indices at that snapshot for that zom-in.

        # Zoom-IDs and snap nums
        snap_nums = np.arange(constants.MIN_SNAP, 100, step=1, dtype=np.uint64)
        zoom_ids = np.arange(0, self.n_clusters, step=1)
        snap_nums = np.broadcast_to(
            snap_nums[:, None],
            (self.n_snaps, self.n_clusters),
        ).transpose().flatten()
        zoom_ids = np.broadcast_to(
            zoom_ids[:, None],
            (self.n_clusters, self.n_snaps),
        ).flatten()

        # primary positions at every zoom-in
        primaries_positions = np.empty((self.n_clusters, self.n_snaps, 3))
        for zoom_id in range(self.n_clusters):
            primaries_positions[zoom_id] = self._get_main_progenitor_positions(
                group_primaries[zoom_id], zoom_id
            )
        # collapsing first two axes into one gives us desired result
        primaries_positions = primaries_positions.reshape([-1, 3])

        # particles indices (these are trickier since the arrays have
        # different shapes)
        particle_indices = []
        archive_file = h5py.File(self.config.cool_gas_history, "r")
        for zoom_id in range(self.n_clusters):
            dataset = f"ZoomRegion_{zoom_id:03d}/particle_indices"
            indices = archive_file[dataset][constants.MIN_SNAP:]
            for i in range(self.n_snaps):
                particle_indices.append(indices[i])
        archive_file.close()

        # combine list of arguments together
        args = list(
            zip(zoom_ids, snap_nums, primaries_positions, particle_indices)
        )
        logging.info("Finished constructing list of arguments.")
        return args

    def _prepare_multiproc_args_single(
        self, primary_subhalo_id: int
    ) -> list[tuple[NDArray | int]]:
        """
        Load info required for multiprocessing a single zoom-in.

        The method creates a list of snapshot numbers and a list of
        constant zoom-in ID, namely the selected current one, and adds
        to each pair of zoom-in ID and snapshot number the corresponding
        primary subhalo position and the array of particle indices
        pointing to the traced particles.

        :param primary_subhalo_id: ID of primary subhalo for selected
            zoom-in at redshift zero.
        :return: List of tuples, containing zoom-in ID, snapshot number
            and the corresponding MBP primary position and list of traced
            particle indices.
        """
        # We must construct a list of tuples, containing zoom-ID, snap
        # num, group primary position at that snap for that zoom-in, and
        # the array of particle indices at that snapshot for that zom-in.

        # Zoom-IDs and snap nums
        snap_nums = np.arange(constants.MIN_SNAP, 100, step=1, dtype=np.uint64)
        zoom_ids = np.empty_like(snap_nums, dtype=np.uint64)
        zoom_ids[:] = self.zoom_id

        # primary positions at selected zoom-in
        primary_positions = self._get_main_progenitor_positions(
            primary_subhalo_id, self.zoom_id
        )

        # particles indices (these are trickier since the arrays have
        # different shapes)
        particle_indices = []
        archive_file = h5py.File(self.config.cool_gas_history, "r")
        dataset = f"ZoomRegion_{self.zoom_id:03d}/particle_indices"
        indices = archive_file[dataset][constants.MIN_SNAP:]
        for i in range(self.n_snaps):
            particle_indices.append(indices[i])
        archive_file.close()

        # combine list of arguments together
        args = list(
            zip(zoom_ids, snap_nums, primary_positions, particle_indices)
        )
        logging.info(
            f"Finished constructing list of arguments for zoom-in "
            f"{self.zoom_id}."
        )
        return args

    def _get_main_progenitor_positions(
        self, primary_id_at_snap99: int, zoom_id: int
    ) -> NDArray:
        """
        Load the positions of the primary subhalo along the MPB.

        Function loads the positions of the main progenitor branch for
        the given subhalo and returns its position at all snapshots,
        starting from ``constants.MIN_SNAP``.

        :param primary_id_at_snap99: ID of the primary at redshift zero.
        :param zoom_id: ID of the zoom-in region.
        :return: Array of position vectors of the primary along its
            main progenitor branch.
        """
        logging.debug(f"Loading primary positions for zoom-in {zoom_id}.")
        primary_positions = np.empty((self.n_snaps, 3))
        primary_positions[:] = np.nan

        mpb = il.sublink.loadTree(
            self.config.base_path,
            self.config.snap_num,
            primary_id_at_snap99,
            fields=["SubhaloPos", "SnapNum"],
            onlyMPB=True,
        )
        positions = units.UnitConverter.convert(
            mpb["SubhaloPos"], "SubhaloPos"
        )
        snaps = mpb["SnapNum"]

        # assign existing positions to array of results
        snap_indices = snaps - constants.MIN_SNAP
        primary_positions[snap_indices] = positions

        # fill missing entries by interpolation
        where_nan = np.argwhere(np.isnan(primary_positions))
        if where_nan.size == 0:
            return primary_positions

        where_nan = where_nan[::3, 0]  # need only one index per 3-vector
        logging.debug(
            f"Interpolating missing main branch progenitor position for "
            f"zoom_in {zoom_id} at snapshots {', '.join(where_nan)}."
        )
        for index in where_nan:
            before = primary_positions[index - 1]
            after = primary_positions[index + 1]
            primary_positions[index] = (before + after) / 2
        return primary_positions

    def _particle_positions(self, snap_num: int, zoom_id: int) -> NDArray:
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

    def _save_particle_distances(
        self,
        zoom_id: int,
        snap_num: int,
        primary_position: NDArray,
        particle_indices: NDArray,
    ) -> None:
        """
        Calculate distance to MP for every particle and save them to file.

        :param zoom_id: Zoom-in region to process.
        :param snap_num: Snapshot at which to find distance.
        :param primary_position: Position of the primary subhalo at this
            snapshot for the current zoom-in. Must be a 3D vector.
        :param particle_indices: The list of indices into the array of
            all particles for the traced particles.
        :return: None, distances saved to file.
        """
        # Step 1: get particle positions
        particle_positions = self._particle_positions(snap_num, zoom_id)
        # Step 2: select only traced particles
        traced_positions = particle_positions[particle_indices[snap_num]]
        # Step 3: get the distance to the MP
        distances = compute.get_distance_periodic_box(
            traced_positions,
            primary_position,
            box_size=constants.BOX_SIZES[self.config.sim_name],
        )
        # Step 4: save to intermediate file
        filename = f"{self.quantity}_z{zoom_id:03d}s{snap_num:02d}.npy"
        np.save(self.tmp_dir / filename, distances)
