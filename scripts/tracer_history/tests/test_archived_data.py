"""
Test the validity of the archived data.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING, Iterator

import h5py
import numpy as np
import pytest

root_dir = Path(__file__).parents[3].resolve()
sys.path.insert(0, str(root_dir / "src"))

from library.config import config

if TYPE_CHECKING:
    from pytest_subtests import SubTests

TNG_CLUSTER_BASE_PATH = config.get_simulation_base_path("TNG-Cluster")
DATA_FILE = (
    root_dir / "data/tracer_history/particle_ids/TNG_Cluster"
    / "particle_ids_from_snapshot_99.hdf5"
)


@pytest.fixture
def data_file() -> Iterator[h5py.File]:
    """Open the data file and yield it, close after test."""
    f = h5py.File(DATA_FILE, "r")
    yield f
    f.close()


@pytest.mark.parametrize("zoom_id", list(range(352)))
def test_indices_range(
    zoom_id: int, data_file: h5py.File, subtests: SubTests
) -> None:
    """Test that indices do not exceed possible range"""
    for snap_num in reversed(range(100)):
        with subtests.test(msg=f"Zoom-in {zoom_id}, snap {snap_num}"):
            indices = (
                data_file[f"ZoomRegion_{zoom_id:03d}/particle_indices"][
                    snap_num, :]
            )
            flags = (
                data_file[f"ZoomRegion_{zoom_id:03d}/particle_type_flags"][
                    snap_num, :]
            )
            lens = (
                data_file[f"ZoomRegion_{zoom_id:03d}/total_particle_num"][
                    snap_num, :]
            )

            # helper quantities
            bh_start_index = lens[0] + lens[1]
            bh_end_index = np.sum(lens)

            # gas
            gas_indices = indices[flags == 0]
            assert np.all((0 <= gas_indices) & (gas_indices < lens[0]))
            # stars
            stars_indices = indices[flags == 4]
            assert np.all(
                (lens[0] <= stars_indices) & (stars_indices < bh_start_index)
            )
            # BHs
            bh_indices = indices[flags == 5]
            assert np.all(
                (bh_start_index <= bh_indices) & (bh_indices < bh_end_index)
            )


@pytest.mark.parametrize("zoom_id", list(range(352)))
def test_total_particle_number(
    zoom_id: int, data_file: h5py.File, subtests: SubTests
) -> None:
    """Test that the saved number of particles is correct"""
    for snap_num in reversed(range(100)):
        with subtests.test(msg=f"Zoom-in {zoom_id}, snap {snap_num}"):
            # load archived number of particles
            lens = (
                data_file[f"ZoomRegion_{zoom_id:03d}/total_particle_num"][
                    snap_num, :]
            )

            # get actual number of particles
            snap_dir = Path(TNG_CLUSTER_BASE_PATH) / f"snapdir_{snap_num:03d}"
            halo_file = snap_dir / f"snap_{snap_num:03d}.{zoom_id}.hdf5"
            fuzz_file = snap_dir / f"snap_{snap_num:03d}.{zoom_id + 352}.hdf5"
            with h5py.File(halo_file, "r") as f:
                num_parts_halo = f["Header"].attrs["NumPart_ThisFile"]
            with h5py.File(fuzz_file, "r") as f:
                num_parts_fuzz = f["Header"].attrs["NumPart_ThisFile"]
            total_num_part = num_parts_halo + num_parts_fuzz

            # compare
            assert lens[0] == total_num_part[0]
            assert lens[1] == total_num_part[4]
            assert lens[2] == total_num_part[5]
