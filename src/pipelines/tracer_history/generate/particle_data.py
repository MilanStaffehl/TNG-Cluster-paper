"""
Trace back some simple quantities of the tracer particles.
"""
from __future__ import annotations

import abc
import dataclasses
import logging
from typing import TYPE_CHECKING, ClassVar

import h5py
import illustris_python as il
import numpy as np

from library import constants, units
from library.data_acquisition import gas_daq, halos_daq, particle_daq
from library.processing import parallelization
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

    quantity: ClassVar[str] = "unset"  # name of the field in the archive
    n_clusters: ClassVar[int] = 352
    n_snaps: ClassVar[int] = 100 - constants.MIN_SNAP

    def __post_init__(self):
        super().__post_init__()
        self.tmp_dir = self.paths["data_dir"] / "intermediate"

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
        self._create_directories(subdirs=["intermediate"], force=True)
        logging.info(f"Tracing {self.quantity} of gas back in time.")

        # Step 1: Load cluster primary
        group_primaries = halos_daq.get_halo_properties(
            self.config.base_path,
            self.config.snap_num,
            ["GroupFirstSub"],
            cluster_restrict=True,
        )["GroupFirstSub"]

        # Step 2: Loop through snapshots and zooms to get quantity
        if self.processes > 1:
            # create combinations of args
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
        else:
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

        # Step 3: archive data
        tracer_file = h5py.File(self.config.cool_gas_history, "r+")
        for zoom_id in range(self.n_clusters):
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

            # fill with data from intermediate files
            for snap_num in range(100):
                if snap_num < constants.MIN_SNAP:
                    data = np.empty(shape, dtype=dtype)
                    data[:] = np.nan
                else:
                    fn = f"{self.quantity}z{zoom_id:03d}s{snap_num:02d}.npy"
                    data = np.load(self.tmp_dir / fn)
                tracer_file[group][self.quantity][snap_num, :] = data

        # Step 4: clean-up
        if self.unlink:
            logging.info("Cleaning up temporary intermediate files.")
            for file in self.tmp_dir.iterdir():
                file.unlink()
            self.tmp_dir.rmdir()
            logging.info("Successfully cleaned up all intermediate files.")

        tracer_file.close()
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
        :return: None
        """
        # Step 0: skip if file exists
        if not self.force_overwrite:
            filename = f"{self.quantity}z{zoom_id:03d}s{snap_num:02d}.npy"
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

        # Step 2: Get particle quantity
        quantity = self._load_quantity(snap_num, zoom_id, primary_subhalo_id)

        # Step 3: Find gas particle indices
        group = f"ZoomRegion_{zoom_id:03d}"
        indices = tracer_file[f"{group}/particle_indices"][snap_num, :]
        flags = (tracer_file[f"{group}/particle_type_flags"][snap_num, :])

        # Step 4: Create an array for the results
        traced_quantity = np.empty(indices.shape, dtype=quantity.dtype)
        traced_quantity[:] = np.nan  # set all to NaN

        # Step 5: Mask data and fill array with results
        if np.max(indices) > quantity.shape[0]:
            # gas only
            traced_quantity[flags == 0] = quantity[indices[flags == 0]]
        else:
            # all particles available
            traced_quantity[:] = quantity[indices]

        # Step 6: Save to intermediate file
        filename = f"{self.quantity}z{zoom_id:03d}s{snap_num:02d}.npy"
        np.save(self.tmp_dir / filename, traced_quantity)

        if multiprocessing:
            tracer_file.close()

    @abc.abstractmethod
    def _load_quantity(
        self, snap_num: int, zoom_id: int, primary_subhalo_id: int
    ) -> NDArray:
        """
        Abstract method to load a cluster quantity.

        Subclasses to this class must implement this method in such a
        way that it returns the array of the desired quantity for all
        gas particles of the given zoom at the given snapshot.

        :param snap_num: The snapshot to query. Must be a number between
            0 and 99.
        :param zoom_id: The zoom-in region ID. Must be a number between
            0 and 351.
        :param primary_subhalo_id: Array of IDs of primary subhalo of every
            cluster at snapshot 99. Useful to trace back cluster
            progenitors through time.
        :return: The gas quantity for every gas cell in the zoom-in at
            that snapshot, such that it can be indexed by the indices
            saved by the generation pipeline.
        """
        pass


# -----------------------------------------------------------------------------
# CONCRETE CLASSES:


class TraceDistancePipeline(TraceSimpleQuantitiesBackABC):
    """
    Trace distance of all particles to cluster with time.
    """

    quantity: ClassVar[str] = "DistanceToMP"

    def _load_quantity(
        self, snap_num: int, zoom_id: int, primary_subhalo_id: int
    ) -> NDArray:
        """
        Find the distance of every gas particle to the cluster.

        The distance must be computed to the current position of the
        main progenitor of the clusters primary subhalo.

        :param snap_num: Snapshot to find the distances at.
        :param zoom_id: The ID of the zoom-in region.
        :return: Array of the distances of all particle cells to the
            cluster center.
        """
        # Step 1: find the cluster center (MPB progenitor position)
        mpb = il.sublink.loadTree(
            self.config.base_path,
            self.config.snap_num,
            primary_subhalo_id,
            fields=["SubhaloPos", "SnapNum"],
            onlyMPB=True,
        )
        positions = mpb["SubhaloPos"]
        snaps = mpb["SnapNum"]
        primary_pos_code_units = positions[snaps == snap_num][0]
        primary_pos = units.UnitConverter.convert(
            primary_pos_code_units, "SubhaloPos", snap_num=snap_num
        )

        # Step 2: Load particles coordinates
        positions_list = []
        for part_type in [0, 4, 5]:
            data = particle_daq.get_particle_properties(
                self.config.base_path,
                snap_num,
                part_type=part_type,
                fields=["Coordinates"],
                zoom_id=zoom_id,
            )
            positions_list.append(data["Coordinates"])

        # Step 3: concatenate particle positions
        part_positions = np.concatenate(positions_list, axis=0)

        # Step 4: Calculate distances
        distances = np.linalg.norm(primary_pos - part_positions, axis=1)
        return distances


class TraceTemperaturePipeline(TraceSimpleQuantitiesBackABC):
    """
    Trace temperature of gas particles with time.
    """

    quantity: ClassVar[str] = "Temperature"

    def _load_quantity(
        self, snap_num: int, zoom_id: int, primary_subhalo_id: int
    ) -> NDArray:
        """
        Find the temperature of gas particles in the zoom-in.

        This loads the temperature of all gas cells in the zoom-in region
        at the given snapshot and returns it, such that it can be indexed
        with the pre-saved indices.

        :param snap_num: The snap for which to load temperatures.
        :param zoom_id: The ID of the zoom-in region.
        :param primary_subhalo_id: Dummy var, not used.
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

    def _load_quantity(
        self, snap_num: int, zoom_id: int, primary_subhalo_id: int
    ) -> NDArray:
        """
        Load the density of all gas cells in the zoom-in.

        Loads only the density of the gas cells, even if there is a
        density field available for black holes.

        :param snap_num: The snap for which to load densities.
        :param zoom_id: The ID of the zoom-in region.
        :param primary_subhalo_id: Dummy var, not used.
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
