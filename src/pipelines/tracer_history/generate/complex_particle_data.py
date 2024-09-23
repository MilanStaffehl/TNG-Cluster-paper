"""
Pipelines to trace back more complex quantities.
"""
from __future__ import annotations

import abc
import dataclasses
import logging
from typing import TYPE_CHECKING, ClassVar

import h5py
import multiprocess as mp
import numpy as np

from library import compute, constants
from library.data_acquisition import halos_daq, particle_daq, sublink_daq
from library.processing import membership
from pipelines import base
from pipelines.tracer_history.generate.mixin import ArchiveMixin

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclasses.dataclass
class TraceComplexQuantityPipeline(base.Pipeline, abc.ABC, ArchiveMixin):
    """
    Trace more complex quantities back in time.

    For multiprocessing, the function which saves the data to intermediate
    files must be explicitly specified as class variable ``working_method``.
    It is the job of the implementations of this ABC to ensure that the
    returned tuples from the methods that prepare the multiprocessing
    args actually match the signature of that method.

    Since some jobs require such large amounts of data per snapshot,
    this class also allows multiprocessing on a per-snapshot basis, where
    the specified number of processes is sequentially started for every
    snapshot. The pool is joined and closed before the one for the next
    snapshot is started, which allows discarding the data for the previous
    snapshot, keeping load on memory minimal. For this purpose, the
    class variable ``split_by_snap`` must be overwritten to be True. If
    this is done, then the method to create args for only one snapshot
    but all zoom-ins :meth:`_prepare_multiproc_by_snap_args` must be
    implemented and the other two methods :meth:`_prepare_multiproc_args`
    and :meth:`_prepare_multiproc_args_single` are never called, and can
    therefore be left empty.
    """

    unlink: bool = False  # delete intermediate files after archiving?
    force_overwrite: bool = False  # overwrite intermediate files?
    zoom_id: int | None = None  # process only one zoom-in or all?
    archive_single: bool = False  # archive data even for a single zoom?

    quantity: ClassVar[str] = "unset"  # overwritten by implementations
    n_clusters: ClassVar[int] = 352
    n_snaps: ClassVar[int] = 100 - constants.MIN_SNAP

    # replace this with the name of the method that saves the intermediate
    # files to disk in implementations of this ABC
    working_method: ClassVar[str | None] = None

    # For jobs where each snap requires a lot of data, the multiprocessing
    # step can be split to run on a per-snapshot basis. Set this to True
    # to run one multiprocessing pool per snapshot.
    split_by_snap: ClassVar[bool] = False

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
        # Step 0: prepare directories, check attributes
        if not self._setup():
            return 3
        if not self._check_working_method():
            return 4
        self._create_directories(
            subdirs=[f"intermediate/{self.quantity.lower()}"], force=True
        )

        # Step 1: process data:
        if self.processes > 1:
            if self.split_by_snap:
                self._multiprocess_by_snap()
            else:
                self._multiprocess()
        elif self.zoom_id is None:
            self._sequential()
        else:
            self._sequential_single()

        # Step 2: archive data
        archive_file = h5py.File(self.config.cool_gas_history, "r+")
        if self.zoom_id is None:
            logging.info(f"Archiving {self.quantity} for all zoom-ins.")
            for zoom_id in range(self.n_clusters):
                self._archive_zoom_in(zoom_id, archive_file)
        elif self.archive_single:
            logging.info(
                f"Archiving {self.quantity} for zoom-in {self.zoom_id}."
            )
            self._archive_zoom_in(self.zoom_id, archive_file)
        else:
            logging.info("Was instructed to not archive data. Skipping.")

        # Step 3: clean-up
        if not self.unlink:
            logging.info("Nothing to delete, done!")
            return 0
        self._clean_up()

        logging.info("Pipeline completed successfully!")
        return 0

    def _check_working_method(self):
        """
        Check that the method in ``self.working_method`` is valid.

        :return: Bool, whether method is valid.
        """
        logging.debug(f"Checking working method {self.working_method}.")
        if self.processes <= 1:
            return True  # no multiprocessing, so it doesn't matter
        if self.working_method is None:
            logging.fatal(
                "Working function for creating intermediate files with "
                "multiprocessing was not specified, cannot proceed."
            )
            return False
        elif self.processes > 1 and not hasattr(self, self.working_method):
            logging.fatal(
                f"Working function for multiprocessing "
                f"'{self.working_method}' does not exist on this pipeline "
                f"instance."
            )
            return False
        elif not callable(getattr(self, self.working_method)):
            logging.fatal(
                f"Working method '{self.working_method}' is not callable."
            )
            return False
        else:
            return True

    def _multiprocess(self) -> None:
        """
        Find quantity using multiple processes.

        Method must first load all required data to reduce the load on
        the file system during parallel execution, which can take quite
        some time. For this purpose, the arguments to the method or
        function ``self.working_method`` are loaded first in the implementations
        of the abstract methods :meth:`_prepare_multiproc_args` and
        :meth:`_prepare_multiproc_args_single` and then fed to the
        working function in a multirpocessing pool.

        :return: None, saves intermediate results to file.
        """
        logging.info("Start preparing args for multiprocessing.")
        if self.zoom_id is None:
            args = self._prepare_multiproc_args()
        else:
            args = self._prepare_multiproc_args_single()

        # open a pool for all arguments
        chunksize = round(len(args) / self.processes / 4, 1)
        chunksize = max(chunksize, 1)
        logging.info(
            f"Starting {self.processes} processes with auto-determined "
            f"chunksize {chunksize} to find {self.quantity}."
        )
        working_method = getattr(self, self.working_method)
        with mp.Pool(processes=self.processes) as pool:
            pool.starmap(working_method, args, chunksize=int(chunksize))
            pool.close()
            pool.join()
        logging.info(
            f"Finished calculating {self.quantity} for all particles!"
        )

    def _multiprocess_by_snap(self):
        """
        Find quantity using multiple processes, one snap at a time.

        This is effectively the same as the :meth:`_multiprocess` method,
        but it will start one pool per snapshot. This is useful for those
        jobs that must load data on a per-snapshot basis that is very
        large and cannot be loaded all at the same time.

        :return: None, saves intermediate results to file.
        """
        logging.info("Starting per-snapshot multiprocessing.")
        working_method = getattr(self, self.working_method)
        for snap_num in range(constants.MIN_SNAP, 100):
            logging.info(f"Loading args for snapshot {snap_num}.")
            if self.zoom_id is None:
                args = self._prepare_multiproc_by_snap_args(snap_num)
            else:
                logging.fatal(
                    "Multiprocessing for a single zoom-in split by snapshot "
                    "is functionally equivalent to running a single process "
                    "for one zoom-in without multiprocessing. This is not "
                    "supported."
                )
                raise RuntimeError("Unnecessary multiprocessing instruction")

            # open a pool for all arguments
            chunksize = round(len(args) / self.processes / 2, 0)
            chunksize = max(chunksize, 1)
            logging.info(
                f"Starting {self.processes} processes with auto-determined "
                f"chunksize {chunksize} to find {self.quantity}."
            )
            with mp.Pool(processes=self.processes) as pool:
                pool.starmap(working_method, args, chunksize=int(chunksize))
                pool.close()
                pool.join()
            logging.info(
                f"Finished calculating {self.quantity} for snapshot {snap_num}."
            )

    @abc.abstractmethod
    def _sequential(self):
        """
        Find quantity sequentially for all particles in all zoom-ins.

        Must save results to file per snapshot and per zoom-in with the
        filename ``f"{self.quantity}_z{zoom_id:03d}s{snap_num:02d}.npy"``.

        :return:  None, saves intermediate results to file.
        """
        pass

    @abc.abstractmethod
    def _sequential_single(self) -> None:
        """
        Find quantity for all particles only for current zoom-in.

        Must save results to file per snapshot and per zoom-in with the
        filename
        ``f"{self.quantity}_z{self.zoom_id:03d}s{snap_num:02d}.npy"``.

        :return: None, saves intermediate results to file.
        """
        pass

    @abc.abstractmethod
    def _prepare_multiproc_args(self) -> list[tuple[NDArray | int]]:
        """
        Load all info required for multiprocessing and arrange it.

        The method creates all possible combinations of zoom-in ID and
        snapshot number, and adds to each pair of other arguments that
        the function stored in ``self.working_method`` requires.

        :return: List of tuples, containing zoom-in ID, snapshot number
            and other arguments required for ``self.working_method``.
        """
        pass

    @abc.abstractmethod
    def _prepare_multiproc_args_single(self) -> list[tuple[NDArray | int]]:
        """
        Load info required for multiprocessing a single zoom-in.

        The method creates a list of snapshot numbers and a list of
        constant zoom-in ID, namely the selected current one, and adds
        to each pair of zoom-in ID and snapshot number the corresponding
        arguments required for the function stored in ``self.working_method``.

        :return: List of tuples, containing zoom-in ID, snapshot number
            and the other arguments required for ``self.working_method``.
        """
        pass

    @abc.abstractmethod
    def _prepare_multiproc_by_snap_args(
        self, snap_num: int
    ) -> list[tuple[NDArray | int]]:
        """
        Load all info required for multiprocessing by snap and arrange it.

        The method creates all combinations of the zoom-id with the given
        snap number, and adds to each pair of other arguments that
        the function stored in ``self.working_method`` requires.

        .. note:: If processing by snap is not required, i.e. unless
            ``self.split_by_snap`` is set to True, this method should
            be implemented to raise an exception.

        :return: List of tuples, containing zoom-in ID, snapshot number
            and other arguments required for ``self.working_method``.
        """
        pass


@dataclasses.dataclass
class TraceDistancePipeline(TraceComplexQuantityPipeline):
    """
    Trace distance to main progenitor back in time.
    """

    quantity: ClassVar[str] = "DistanceToMP"
    working_method: ClassVar[str] = "_save_particle_distances"

    def _sequential(self) -> None:
        """
        Find distance to MBP of every particle sequentially.

        :return: None, saves intermediate results to file.
        """
        logging.debug("Started processing clusters sequentially.")
        group_primaries = self._get_group_primaries()
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
                    particle_indices[snap_num],
                )
        archive_file.close()

    def _sequential_single(self) -> None:
        """
        Find distance to MBP of every particle for one zoom-in sequentially.

        :return: None, saves intermediate results to file.
        """
        logging.info(f"Processing zoom-in {self.zoom_id} sequentially.")
        archive_file = h5py.File(self.config.cool_gas_history, "r")

        # Step 1: get primary positions for this zoom-id
        group_primaries = self._get_group_primaries()
        primary_subhalo_id = group_primaries[self.zoom_id]
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
                particle_indices[snap_num],
            )

        archive_file.close()

    def _prepare_multiproc_args(self) -> list[tuple[NDArray | int]]:
        """
        Load all info required for multiprocessing and arrange it.

        The method creates all possible combinations of zoom-in ID and
        snapshot number, and adds to each pair of zoom-in Id and snapshot
        number the corresponding primary subhalo position and the array
        of particle indices pointing to the traced particles.

        :return: List of tuples, containing zoom-in ID, snapshot number
            and the corresponding MBP primary position and list of traced
            particle indices.
        """
        # We must construct a list of tuples, containing zoom-ID, snap
        # num, group primary position at that snap for that zoom-in, and
        # the array of particle indices at that snapshot for that zom-in.

        # Zoom-IDs and snap nums
        snap_nums = np.arange(constants.MIN_SNAP, 100, step=1, dtype=np.uint64)
        zoom_ids = np.arange(0, self.n_clusters, step=1, dtype=np.uint64)
        snap_nums = np.broadcast_to(
            snap_nums[:, None],
            (self.n_snaps, self.n_clusters),
        ).transpose().flatten()
        zoom_ids = np.broadcast_to(
            zoom_ids[:, None],
            (self.n_clusters, self.n_snaps),
        ).flatten()

        # primary positions at every zoom-in
        group_primaries = self._get_group_primaries()
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

    def _prepare_multiproc_args_single(self) -> list[tuple[NDArray | int]]:
        """
        Load info required for multiprocessing a single zoom-in.

        The method creates a list of snapshot numbers and a list of
        constant zoom-in ID, namely the selected current one, and adds
        to each pair of zoom-in ID and snapshot number the corresponding
        primary subhalo position and the array of particle indices
        pointing to the traced particles.

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
        group_primaries = self._get_group_primaries()
        primary_subhalo_id = group_primaries[self.zoom_id]
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

    def _prepare_multiproc_by_snap_args(
        self, snap_num: int
    ) -> list[tuple[NDArray | int]]:
        """
        Not implemented, raises exception.

        :param snap_num: Snapshot.
        :return: Never returns, raises exception.
        """
        raise NotImplementedError(
            "Pipeline does not support by-snap multiprocessing."
        )

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
            all particles for the traced particles. This should be only
            the indices for the current snapshot, i.e. it must be a 1D
            array of indices, **not** a shape (100, N) array!
        :return: None, distances saved to file.
        """
        # Step 0: skip if not required
        filename = f"{self.quantity}_z{zoom_id:03d}s{snap_num:02d}.npy"
        if (self.tmp_dir / filename).exists() and not self.force_overwrite:
            if self.processes <= 1:
                logging.debug(
                    f"Data file for zoom-in {zoom_id} at snapshot {snap_num} "
                    f"exists and overwrite was not forced. Skipping."
                )
            return
        # Step 1: get particle positions
        particle_positions = self._particle_positions(snap_num, zoom_id)
        # Step 2: select only traced particles
        traced_positions = particle_positions[particle_indices]
        # Step 3: get the distance to the MP
        distances = compute.get_distance_periodic_box(
            traced_positions,
            primary_position,
            box_size=constants.BOX_SIZES[self.config.sim_name],
        )
        # Step 4: save to intermediate file:
        # Python has a bad habit of flushing file buffers as late as
        # possible - which causes huge memory pile-up. So we force Python
        # to flush and close immediately to prevent this.
        with open(self.tmp_dir / filename, "wb") as file:
            np.save(file, distances)
        # Step 5: just for good measure: clean-up
        del distances, traced_positions, particle_positions

    def _get_group_primaries(self) -> NDArray:
        """
        Load and return a list of primary subhalo IDs at redshift 0.

        :return: Array of the IDs of the primary subhalo for all 352
            clusters.
        """
        group_primaries = halos_daq.get_halo_properties(
            self.config.base_path,
            self.config.snap_num,
            ["GroupFirstSub"],
            cluster_restrict=True,
        )["GroupFirstSub"]
        return group_primaries

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
        mpb = sublink_daq.get_mpb_properties(
            self.config.base_path,
            self.config.snap_num,
            primary_id_at_snap99,
            fields=["SubhaloPos"],
            start_snap=constants.MIN_SNAP,
            log_warning=True,
        )
        return mpb["SubhaloPos"]

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


class TraceParentHaloPipeline(TraceComplexQuantityPipeline):
    """
    Trace the parent halo index of every particle back in time.
    """

    quantity: ClassVar[str] = "ParentHaloIndex"
    working_method: ClassVar[str] = "_save_parent_halo_index"
    split_by_snap: ClassVar[bool] = True  # process on per-snap basis!

    def _sequential(self):
        """
        Find parent halo index for every particle in all zoom-ins.

        :return: None, saves intermediate results to file.
        """
        logging.debug("Started processing clusters sequentially.")
        archive_file = h5py.File(self.config.cool_gas_history, "r")
        for snap_num in range(self.n_snaps):
            logging.info(f"Processing snapshot {snap_num}.")

            # Step 1: get offsets and group lengths
            fof_offsets, fof_lens, _, _ = membership.load_offsets_and_lens(
                self.config.base_path, snap_num, group_only=True,
            )

            # Step 2: loop over zoom-ins
            for zoom_id in range(constants.MIN_SNAP, 100):
                logging.debug(
                    f"Processing zoom-in {zoom_id}, snap {snap_num}."
                )
                # Step 2.1: get particle indices and type flags
                dataset = f"ZoomRegion_{zoom_id:03d}"
                pind = archive_file[dataset]["particle_indices"][snap_num]
                ptfs = archive_file[dataset]["particle_type_flags"][snap_num]
                # Step 2.2: save parent halo index
                self._save_parent_halo_index(
                    zoom_id, snap_num, fof_offsets, fof_lens, pind, ptfs
                )

            # Step 3: clean-up
            del fof_offsets, fof_lens
        archive_file.close()

    def _sequential_single(self) -> None:
        """
        Find parent halo index for every particle in only one zoom-in.

        :return: None, saves intermediate results to file.
        """
        logging.debug(f"Processing zoom-in {self.zoom_id} sequentially.")
        archive_file = h5py.File(self.config.cool_gas_history, "r")
        for snap_num in range(constants.MIN_SNAP, 100):
            logging.info(f"Processing snapshot {snap_num}.")

            # Step 1: get offsets and group lengths
            fof_offsets, fof_lens, _, _ = membership.load_offsets_and_lens(
                self.config.base_path, snap_num, group_only=True,
            )

            # Step 2: find parent halo indices
            dataset = f"ZoomRegion_{self.zoom_id:03d}"
            pind = archive_file[dataset]["particle_indices"][snap_num]
            ptfs = archive_file[dataset]["particle_type_flags"][snap_num]
            # Step 2.2: save parent halo index
            self._save_parent_halo_index(
                self.zoom_id, snap_num, fof_offsets, fof_lens, pind, ptfs
            )

            # Step 3: clean-up
            del fof_offsets, fof_lens
        archive_file.close()

    def _prepare_multiproc_args(self) -> list[tuple[NDArray | int]]:
        """
        Not implemented.

        Loading all args required for multiprocessing at once creates
        too much data to handle reliably, so this is not allowed. Raises
        an exception when called, but shouldn't ever be called by the
        pipeline anyway.

        :return: Never returns, raises ``NotImplementedError``.
        """
        raise NotImplementedError(
            f"Multiprocessing over all snaps and zoom-ins not supported "
            f"for {self.quantity}."
        )

    def _prepare_multiproc_args_single(self) -> list[tuple[NDArray | int]]:
        """
        Not implemented.

        Loading all args required for multiprocessing at once creates
        too much data to handle reliably, so this is not allowed. Raises
        an exception when called, but shouldn't ever be called by the
        pipeline anyway.

        :return: Never returns, raises ``NotImplementedError``.
        """
        raise NotImplementedError(
            f"Multiprocessing over all snaps not supported for "
            f"{self.quantity}."
        )

    def _prepare_multiproc_by_snap_args(
        self, snap_num: int
    ) -> list[tuple[NDArray | int]]:
        """
        Load info required for multiprocessing a single snapshot.

        The method creates a list of zoom-in IDs and a list of constant
        snapshot number, namely the selected current one, and adds
        to each pair of zoom-in ID and snapshot number the corresponding
        FoF offsets and FoF lengths, as well as the particle indices and
        particle type flags for the traced particles.

        :return: List of tuples, containing zoom-in ID, snapshot number
            and the corresponding FoF offsets and FoF lengths and particle
            indices and type flags.
        """
        # Zoom-IDs and snap nums
        zoom_ids = np.arange(0, self.n_clusters, step=1, dtype=np.uint64)
        snap_nums = np.empty_like(zoom_ids, dtype=np.uint64)
        snap_nums[:] = snap_num

        # FoF offsets and lengths
        logging.debug(f"Loading FoF offsets and lengths for snap {snap_num}.")
        offsets, lengths, _, _ = membership.load_offsets_and_lens(
            self.config.base_path, snap_num, group_only=True
        )
        fof_offsets = [offsets for _ in range(self.n_clusters)]
        fof_lengths = [lengths for _ in range(self.n_clusters)]

        # particle indices and tpe flags
        particle_indices = []
        particle_type_flags = []
        archive_file = h5py.File(self.config.cool_gas_history, "r")
        for zoom_id in zoom_ids:
            ds_indices = f"ZoomRegion_{zoom_id:03d}/particle_indices"
            particle_indices.append(archive_file[ds_indices][snap_num])
            ds_flags = f"ZoomRegion_{zoom_id:03d}/particle_type_flags"
            particle_type_flags.append(archive_file[ds_flags][snap_num])
        archive_file.close()

        # combine lists/arrays into list of argument tuples
        args = list(
            zip(
                zoom_ids,
                snap_nums,
                fof_offsets,
                fof_lengths,
                particle_indices,
                particle_type_flags
            )
        )
        logging.info(
            f"Finished constructing list of arguments for snap {snap_num}."
        )
        return args

    def _save_parent_halo_index(
        self,
        zoom_id: int,
        snap_num: int,
        fof_offsets: NDArray,
        fof_lens: NDArray,
        particle_indices: NDArray,
        particle_type_flags: NDArray,
    ) -> None:
        """
        Find parent halo index for every particle and save them to file.

        :param zoom_id: ID of the zoom-in region to process.
        :param snap_num: The snapshot to process.
        :param fof_offsets: Array of FoF group offsets for this zoom-in
            and snapshot. Must be split by particle type and have shape
            (6, N).
        :param fof_lens: Array of FoF group lengths for this zoom-in
            and snapshot. Must be split by particle type and have shape
            (6, N).
        :param particle_indices: The list of indices into the array of
            all particles for the traced particles. This should be only
            the indices for the current snapshot, i.e. it must be a 1D
            array of indices, **not** a shape (100, N) array!
        :param particle_type_flags: The list of particle type flags
            assigning a particle type (0, 4, or 5) to every particle.
            This must have the same shape as ``particle_indices``.
        :return: None, parent halo indices are saved to file.
        """
        # Step 0: skip if not required
        filename = f"{self.quantity}_z{zoom_id:03d}s{snap_num:02d}.npy"
        if (self.tmp_dir / filename).exists() and not self.force_overwrite:
            if self.processes <= 1:
                logging.debug(
                    f"Data file for zoom-in {zoom_id} at snapshot {snap_num} "
                    f"exists and overwrite was not forced. Skipping."
                )
            return

        # Step 1: get particle IDs
        particle_ids = self._particle_ids(snap_num, zoom_id)

        # Step 2: select only traced particles
        traced_ids = particle_ids[particle_indices]

        # Step 3: get the parent halo indices for every particle
        parent_halo_indices = np.empty(traced_ids.shape, dtype=np.int64)
        parent_halo_indices[:] = -1  # fill with sentinel value
        for part_type in [0, 4, 5]:
            where = (particle_type_flags == part_type)
            parent_halo_indices[where] = membership.find_parent(
                traced_ids[where],
                fof_offsets[:, part_type],
                fof_lens[:, part_type],
            )

        # Step 4: save to file, forcing immediate flushing of buffer
        with open(self.tmp_dir / filename, "wb") as file:
            np.save(file, parent_halo_indices)

        # Step 5: clean-up
        del particle_ids, traced_ids, parent_halo_indices

    def _particle_ids(self, snap_num: int, zoom_id: int) -> NDArray:
        """
        Load and return the array of particle indices.

        The returned array is the array of particle indices of type 0,
        4 and 5 in that order for the specified zoom-in at the specified
        snapshot.

        :param snap_num: Snapshot to load from.
        :param zoom_id: The ID of the zoom-in for which to load particle
            IDs.
        :return: Array of contiguous particle IDs of particles of type
            0, 4, and 5, in that order.
        """
        particle_ids_list = []
        for part_type in [0, 4, 5]:
            pids = particle_daq.get_particle_ids(
                self.config.base_path,
                snap_num,
                part_type=part_type,
                zoom_id=zoom_id,
            )
            if pids.size == 0:
                continue  # no particles of this type exist
            particle_ids_list.append(pids)

        # concatenate particle positions
        part_ids = np.concatenate(particle_ids_list, axis=0)
        return part_ids
