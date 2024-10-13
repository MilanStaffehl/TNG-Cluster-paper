"""
Tests for the cgh_archive module.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import h5py
import illustris_python as il
import numpy as np
import pytest

from library import cgh_archive as cgh
from library.config import config
from library.data_acquisition import particle_daq

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from pytest_subtests import SubTests

BASE_PATH = config.get_simulation_base_path("TNG-Cluster")
ARCHIVE_FILE = Path(
    "/vera/ptmp/gc/mista/thesisProject/data/tracer_history/TNG_Cluster/"
    "cool_gas_history.hdf5"
)


def _load_cluster_pids(zoom_in: int, snap_num: int) -> NDArray:
    """Helper to load PIDs of a single zoom-in"""
    particle_ids_list = []
    # load data and append it to lists
    for part_type in [0, 4, 5]:
        cur_particle_ids = particle_daq.get_particle_ids(
            BASE_PATH,
            snap_num=snap_num,
            part_type=part_type,
            zoom_id=zoom_in,
        )
        if cur_particle_ids.size == 0:
            continue  # no particle data available, skip
        particle_ids_list.append(cur_particle_ids)
    particle_ids = np.concatenate(particle_ids_list, axis=0)
    return particle_ids


def _load_all_pids(snap_num: int) -> NDArray:
    """Helper to load PIDs of all particles in simulation"""
    particle_ids_list = []
    # load data and append it to lists
    for part_type in [0, 4, 5]:
        cur_particle_ids = il.snapshot.loadSubset(
            BASE_PATH,
            snap_num,
            partType=part_type,
            fields=["ParticleIDs"],
        )
        particle_ids_list.append(cur_particle_ids)
    return particle_ids_list


def test_index_to_global() -> None:
    """Test the function to turn indices global"""
    # create mock file offsets, but only those needed
    mock_offsets = np.zeros((354, 6), dtype=np.int64)
    mock_offsets[1] = np.array([220, 0, 0, 0, 120, 55])
    mock_offsets[2] = np.array([460, 0, 0, 0, 210, 108])
    mock_offsets[353] = np.array([120_240, 0, 0, 0, 8200, 2300])
    mock_total_part_num = np.array([2000, 1000, 100])
    mock_indices = np.array([
        0, 110, 239, 240, 320,
        2000, 2045, 2089, 2090, 2110,
        3000, 3023, 3052, 3053, 3059,
    ], dtype=np.int64)  # yapf: disable
    mock_types = np.array([
        0, 0, 0, 0, 0,
        4, 4, 4, 4, 4,
        5, 5, 5, 5, 5,
    ], dtype=np.int64)  # yapf: disable

    # create expected data
    expected_indices = np.array([
        220, 330, 459, 120_480, 120_560,
        120, 165, 209, 8290, 8310,
        55, 78, 107, 2353, 2359,
    ], dtype=np.int64)  # yapf: disable

    # test function
    output = cgh.index_to_global(
        1, mock_indices, mock_types, mock_total_part_num, mock_offsets
    )
    np.testing.assert_array_equal(expected_indices, output)


@pytest.mark.skipif(
    not ARCHIVE_FILE.exists(),
    reason="Can only be executed if archived data exists."
)
@pytest.mark.parametrize("snap_num", list(range(100)))
def test_index_to_global_with_real_data(
    snap_num: int, subtests: SubTests
) -> None:
    """
    Check that the function works on real data.

    We load the same particle property, once using archived local
    indices, and once using calculated global indices. They must match.
    """
    # open the archive file
    archive = h5py.File(ARCHIVE_FILE, "r")
    # pre-load common quantities between zoom-regions
    all_pids = _load_all_pids(snap_num)
    offset_directory = Path(BASE_PATH).parent / "postprocessing/offsets"
    offset_file = offset_directory / f"offsets_{snap_num:03d}.hdf5"
    with h5py.File(offset_file, "r") as file:
        file_offsets = file["FileOffsets/SnapByType"][()]

    # test the indices
    for zoom_id in range(352):
        with subtests.test(msg=f"Zoom-in {zoom_id}"):
            # load PIDs using archived indices
            pids = _load_cluster_pids(zoom_id, snap_num)
            grp = f"ZoomRegion_{zoom_id:03d}"
            particle_indices = archive[grp]["particle_indices"][snap_num]
            selected_pids = pids[particle_indices]

            # find global indices
            flags = archive[grp]["particle_type_flags"][snap_num]
            part_num = archive[grp]["total_particle_num"][snap_num]
            global_indices = cgh.index_to_global(
                zoom_id,
                particle_indices,
                flags,
                part_num,
                file_offsets=file_offsets,
            )
            derived_pids = np.zeros_like(global_indices)
            derived_pids[flags == 0] = all_pids[0][global_indices[flags == 0]]
            derived_pids[flags == 4] = all_pids[1][global_indices[flags == 4]]
            derived_pids[flags == 5] = all_pids[2][global_indices[flags == 5]]

            # test validity of data derived from global indices
            np.testing.assert_array_equal(selected_pids, derived_pids)

    archive.close()
