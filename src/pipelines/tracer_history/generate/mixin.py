"""
Mixin for tracing particle quantities pipelines.
"""
from __future__ import annotations

import contextlib
import logging
from pathlib import Path
from typing import Protocol

import h5py
import numpy as np

from library import constants


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

    def _setup(self: TracePipelineProtocol) -> bool:
        """
        Set-up pipeline, check directories and config.

        :return: Exit code.
        """
        if self.quantity == "unset":
            logging.fatal(
                "Quantity name unset. Does your pipeline implementation "
                "overwrite the class variable `quantity` with a proper name "
                "for the quantity?"
            )
            return False
        if self.zoom_id is None:
            logging.info(f"Tracing {self.quantity} of particles back in time.")
        else:
            logging.info(
                f"Tracing {self.quantity} of particles back in time for "
                f"zoom-in {self.zoom_id} only."
            )
        return True

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
        else:
            # check shapes match
            dset_shape = (tracer_file[group][self.quantity].shape[1], )
            if dset_shape != shape:
                logging.error(
                    f"Existing group in archive has different shape than the "
                    f"data I am currently attempting to archive: dataset has "
                    f"shape {dset_shape}, but intermediate files contain data "
                    f"of shape {shape}. Aborting archiving."
                )
                raise ValueError(
                    "Archive dataset and intermediate data have mismatched "
                    f"shapes. hdf5 dataset: {dset_shape}, intermediate file: "
                    f"{shape}"
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
