"""
Tests to validate the real, archived postprocessed data.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import h5py
import numpy as np
import pytest

from library import constants
from library.data_acquisition import halos_daq, sublink_daq

ARCHIVE_FILE = Path(
    "/vera/ptmp/gc/mista/thesisProject/data/tracer_history/TNG_Cluster/"
    "cool_gas_history.hdf5"
)
BASE_PATH = "/virgotng/mpia/TNG-Cluster/TNG-Cluster/output"
if ARCHIVE_FILE.exists():
    PRIMARIES = halos_daq.get_halo_properties(
        BASE_PATH,
        99,
        ["GroupFirstSub"],
        cluster_restrict=True,
    )["GroupFirstSub"]
else:
    PRIMARIES = None

# skip module unless executed on the VERA cluster
if not ARCHIVE_FILE.exists():
    pytest.skip(
        "Can only be executed if cool gas history data exists.",
        allow_module_level=True,
    )


@pytest.mark.parametrize("zoom_id", list(range(352)))
def test_archived_crossing_times(zoom_id: int) -> None:
    """Sanity check for archived times: do they point to a crossing?"""
    assert PRIMARIES is not None
    # load crossing times and distances
    with h5py.File(ARCHIVE_FILE, "r") as file:
        grp = f"ZoomRegion_{zoom_id:03d}"
        first_crossing_z = file[grp]["FirstCrossingRedshift"][()]
        last_crossing_z = file[grp]["LastCrossingRedshift"][()]
        particle_distances = file[grp]["DistanceToMP"][()]
    # load ID of primary subhalo at redshift zero for MPB tree
    primary = PRIMARIES[zoom_id]
    # load virial radii (shape (92, N))
    virial_radii = sublink_daq.get_mpb_properties(
        BASE_PATH, 99, primary, ["Group_R_Crit200"], constants.MIN_SNAP
    )["Group_R_Crit200"]

    # turn the redshifts into snap nums
    first_crossing = 99 - np.searchsorted(
        np.flip(constants.REDSHIFTS), first_crossing_z
    )
    last_crossing = 99 - np.searchsorted(
        np.flip(constants.REDSHIFTS), last_crossing_z
    )

    # Actual test: check that before crossing, the particle is outside
    # the cluster, and inside after.
    n = particle_distances.shape[1]
    ms = constants.MIN_SNAP

    radius_before_first_crossing = virial_radii[first_crossing - ms]
    dist_before_first_crossing = np.array(
        [particle_distances[first_crossing[i], i] for i in range(n)]
    )
    assert np.all(
        dist_before_first_crossing > 2 * radius_before_first_crossing
    )

    radius_after_first_crossing = virial_radii[first_crossing - ms + 1]
    dist_after_first_crossing = np.array(
        [particle_distances[first_crossing[i] + 1, i] for i in range(n)]
    )
    assert np.all(dist_after_first_crossing < 2 * radius_after_first_crossing)

    radius_before_last_crossing = virial_radii[last_crossing - ms]
    dist_before_last_crossing = np.array(
        [particle_distances[last_crossing[i], i] for i in range(n)]
    )
    assert np.all(dist_before_last_crossing > 2 * radius_before_last_crossing)

    radius_after_last_crossing = virial_radii[last_crossing - ms + 1]
    dist_after_last_crossing = np.array(
        [particle_distances[last_crossing[i] + 1, i] for i in range(n)]
    )
    assert np.all(dist_after_last_crossing < 2 * radius_after_last_crossing)


def _verify_parent_category(
    parent_category: int,
    parent_halo_id: int,
    parent_subhalo_id: int,
    primary_halo_id: int,
    primary_subhalo_id: int,
    snap_num,
    zoom_in
) -> None:
    """Verify that the parent categories are valid"""
    match parent_category:
        case 0:
            # unbound particle
            assert parent_halo_id == -1
            assert parent_subhalo_id == -1
        case 1:
            # bound to other halo
            assert parent_halo_id != -1
            assert parent_halo_id != primary_halo_id
        case 2:
            # inner fuzz of primary halo
            assert parent_halo_id == primary_halo_id
            assert parent_subhalo_id == -1
        case 3:
            # bound to primary subhalo of the primary halo
            assert parent_halo_id == primary_halo_id
            assert parent_subhalo_id == primary_subhalo_id
        case 4:
            # bound to any satellite of primary halo
            assert parent_halo_id == primary_halo_id
            assert parent_subhalo_id != -1
            assert parent_subhalo_id != primary_subhalo_id
        case 255:
            warnings.warn(
                UserWarning(
                    f"Tested particle of parent category 255. This should not "
                    f"have happened. Snap {snap_num} zoom-in {zoom_in}"
                ),
                stacklevel=2,
            )
        case _:
            pytest.fail(f"Unexpected parent category {parent_category}.")


@pytest.mark.parametrize("zoom_id", list(range(352)))
def test_archived_parent_categories(zoom_id: int, subtests) -> None:
    """Check archived parent categories: are they plausible?"""
    assert PRIMARIES is not None
    # load the mpb of the current zoom-in
    mpb = sublink_daq.get_mpb_properties(
        BASE_PATH,
        99,
        PRIMARIES[zoom_id],
        start_snap=constants.MIN_SNAP,
        fields=["SubfindID", "SubhaloGrNr"],
        interpolate=False,  # cannot interpolate IDs
    )
    # load parent category and parent halo and subhalo index
    with h5py.File(ARCHIVE_FILE, "r") as f:
        grp = f"ZoomRegion_{zoom_id:03d}"
        parent_category = f[grp]["ParentCategory"][()]
        parent_halo = f[grp]["ParentHaloIndex"][()]
        parent_subhalo = f[grp]["ParentSubhaloIndex"][()]

    # pick a few particles at random
    n_particles = parent_category.shape[1]
    indices = np.random.randint(0, n_particles, 8)

    # verify per snap
    for snap_num in range(constants.MIN_SNAP, 100):
        index = np.nonzero(mpb["SnapNum"] == snap_num)[0]
        if mpb["SubfindID"][index] == -1:
            continue  # sublink misses this snapshot
        with subtests.test(msg=f"Snap num {snap_num}"):
            primary_halo = mpb["SubhaloGrNr"][index]
            primary_subhalo = mpb["SubfindID"][index]
            for i in indices:
                _verify_parent_category(
                    parent_category[snap_num][i],
                    parent_halo[snap_num][i],
                    parent_subhalo[snap_num][i],
                    primary_halo,
                    primary_subhalo,
                    snap_num,
                    zoom_id
                )
