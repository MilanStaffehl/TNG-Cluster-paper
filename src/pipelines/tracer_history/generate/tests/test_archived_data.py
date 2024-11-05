"""Tests for archived data, performed with real sim data."""
from __future__ import annotations

import itertools
from pathlib import Path

import h5py
import pytest

from library.processing import membership

ARCHIVE_FILE = Path(
    "/vera/ptmp/gc/mista/thesisProject/data/tracer_history/TNG_Cluster/"
    "cool_gas_history.hdf5"
)
BASE_PATH = "/virgotng/mpia/TNG-Cluster/TNG-Cluster/output"

# skip module unless executed on the VERA cluster
if not ARCHIVE_FILE.exists():
    pytest.skip(
        "Can only be executed if cool gas history data exists.",
        allow_module_level=True,
    )

PARAM_LIST = list(itertools.product(range(352), reversed(range(8, 100))))


@pytest.mark.parametrize("zoom_in, snap_num", PARAM_LIST)
def test_parent_halo_index_for_group_catalogue(
    zoom_in: int, snap_num: int
) -> None:
    """Test that all particles in a group have the correct parent."""
    limit = 1000  # limits the number of particles to test

    # Load parent halo indices
    with h5py.File(ARCHIVE_FILE, "r") as f:
        grp = f"ZoomRegion_{zoom_in:03d}"
        particle_indices = f[grp]["particle_indices"][()]
        particle_types = f[grp]["particle_type_flags"][()]
        total_part_num = f[grp]["total_particle_num"][()]
        parent_halos = f[grp]["ParentHaloIndex"][()]

    # load offsets of current
    offsets, lengths, _, _ = membership.load_offsets_and_lens(
        BASE_PATH, snap_num, group_only=True
    )
    offset_dir = Path(BASE_PATH).parent / "postprocessing/offsets"
    with h5py.File(offset_dir / f"offsets_{snap_num:03d}.hdf5", "r") as f:
        file_offsets = f["FileOffsets/SnapByType"][()]

    # offsets of every particle in archive-style
    type_offsets = [
        0,
        0,
        0,
        0,
        total_part_num[snap_num][0],
        total_part_num[snap_num][0] + total_part_num[snap_num][1],
    ]

    # go through particles
    fof_file_lengths = file_offsets[zoom_in + 1] - file_offsets[zoom_in]
    for i, particle_index in enumerate(particle_indices[snap_num][:limit]):
        host_halo = parent_halos[snap_num][i]
        part_type = particle_types[snap_num][i]

        if host_halo == -1:
            # check particle isn't in any FoF file (but using archive-style
            # indices)
            fuzz_start = fof_file_lengths[part_type] + type_offsets[part_type]
            assert particle_index >= fuzz_start
        else:
            # find group offsets and transform
            group_offset = offsets[host_halo][part_type]
            start = group_offset - file_offsets[zoom_in][part_type]
            start += type_offsets[part_type]
            end = start + lengths[host_halo][part_type]
            # check that particle falls into the group range
            assert start <= particle_index < end
