"""
Tests for the cgh_archive module.
"""
from __future__ import annotations

import contextlib
from pathlib import Path
from typing import TYPE_CHECKING

import h5py
import numpy as np
import pytest

from library import cgh_archive as cgh
from library import constants
from library.config import config
from library.data_acquisition import particle_daq

if TYPE_CHECKING:
    from numpy.typing import NDArray

SNAP_NUMS = reversed(list(range(constants.MIN_SNAP, 100)))
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


def test_index_to_global() -> None:
    """Test the function to turn indices global"""
    # create mock file offsets, but only those needed
    mock_offsets = np.zeros((354, 6), dtype=np.int64)
    # lengths: 240, 0, 0, 0, 90, 53
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
        220, 330, 459, 120_240, 120_320,
        120, 165, 209, 8200, 8220,
        55, 78, 107, 2300, 2306,
    ], dtype=np.int64)  # yapf: disable

    # test function
    output = cgh.index_to_global(
        1, mock_indices, mock_types, mock_total_part_num, mock_offsets
    )
    np.testing.assert_array_equal(expected_indices, output)


@pytest.mark.skipif(
    not ARCHIVE_FILE.exists() or not Path(BASE_PATH).exists(),
    reason="Can only be executed if data exists."
)
@pytest.mark.parametrize("snap_num", SNAP_NUMS)
def test_global_indices_with_real_data(snap_num: int) -> None:
    """Test the function with real data, but only for one cluster."""
    zoom_id = 2  # change manually for different tests

    # open the archive file
    archive = h5py.File(ARCHIVE_FILE, "r")

    # pre-load common quantities between zoom-regions
    offset_directory = Path(BASE_PATH).parent / "postprocessing/offsets"
    offset_file = offset_directory / f"offsets_{snap_num:03d}.hdf5"
    with h5py.File(offset_file, "r") as file:
        file_offsets = file["FileOffsets/SnapByType"][()]

    # load PIDs using archived indices
    pids = _load_cluster_pids(zoom_id, snap_num)
    grp = f"ZoomRegion_{zoom_id:03d}"
    particle_indices = archive[grp]["particle_indices"][snap_num]
    flags = archive[grp]["particle_type_flags"][snap_num]
    part_num = archive[grp]["total_particle_num"][snap_num]
    selected_pids = pids[particle_indices]

    # create a safety copy of the indices for later verification
    safety_copy_indices = particle_indices.copy()

    # load mocked list of "all" particles
    global_pids_t0 = np.zeros(file_offsets[zoom_id + 353, 0])
    global_pids_t4 = np.zeros(file_offsets[zoom_id + 353, 4])
    global_pids_t5 = np.zeros(file_offsets[zoom_id + 353, 5])
    snap_dir = BASE_PATH + f"/snapdir_{snap_num:03d}/"
    fof_file = f"snap_{snap_num:03d}.{zoom_id}.hdf5"
    with h5py.File(snap_dir + fof_file, "r") as f:
        j = zoom_id
        with contextlib.suppress(KeyError):
            pids_t0 = f["PartType0/ParticleIDs"][()]
            global_pids_t0[file_offsets[j, 0]:file_offsets[j + 1, 0]] = pids_t0
        with contextlib.suppress(KeyError):
            pids_t4 = f["PartType4/ParticleIDs"][()]
            global_pids_t4[file_offsets[j, 4]:file_offsets[j + 1, 4]] = pids_t4
        with contextlib.suppress(KeyError):
            pids_t5 = f["PartType5/ParticleIDs"][()]
            global_pids_t5[file_offsets[j, 5]:file_offsets[j + 1, 5]] = pids_t5
    fuzz_file = f"snap_{snap_num:03d}.{int(zoom_id + 352)}.hdf5"
    with h5py.File(snap_dir + fuzz_file, "r") as f:
        i = zoom_id + 352
        with contextlib.suppress(KeyError):
            pids_t0 = f["PartType0/ParticleIDs"][()]
            global_pids_t0[file_offsets[i, 0]:file_offsets[i + 1, 0]] = pids_t0
        with contextlib.suppress(KeyError):
            pids_t4 = f["PartType4/ParticleIDs"][()]
            global_pids_t4[file_offsets[i, 4]:file_offsets[i + 1, 4]] = pids_t4
        with contextlib.suppress(KeyError):
            pids_t5 = f["PartType5/ParticleIDs"][()]
            global_pids_t5[file_offsets[i, 5]:file_offsets[i + 1, 5]] = pids_t5

    # find global indices
    global_indices = cgh.index_to_global(
        zoom_id,
        particle_indices,
        flags,
        part_num,
        file_offsets=file_offsets,
    )
    derived_pids = np.zeros_like(global_indices)
    derived_pids[flags == 0] = global_pids_t0[global_indices[flags == 0]]
    derived_pids[flags == 4] = global_pids_t4[global_indices[flags == 4]]
    derived_pids[flags == 5] = global_pids_t5[global_indices[flags == 5]]

    # test validity of data derived from global indices
    np.testing.assert_array_equal(selected_pids, derived_pids)

    # test that originally loaded indices were not altered in place
    np.testing.assert_array_equal(safety_copy_indices, particle_indices)

    archive.close()
