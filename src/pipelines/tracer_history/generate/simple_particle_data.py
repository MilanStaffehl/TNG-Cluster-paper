"""
Trace back some simple quantities of the tracer particles.
"""
from __future__ import annotations

import abc
import dataclasses
import logging
from typing import TYPE_CHECKING, ClassVar

import h5py
import numpy as np

from library import constants
from library.data_acquisition import gas_daq, particle_daq
from library.processing import parallelization
from pipelines import base
from pipelines.tracer_history.generate.mixin import ArchiveMixin

if TYPE_CHECKING:
    from numpy.typing import NDArray


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
        if not self._setup():
            return 3
        self._create_directories(
            subdirs=[f"intermediate/{self.quantity.lower()}"], force=True
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
