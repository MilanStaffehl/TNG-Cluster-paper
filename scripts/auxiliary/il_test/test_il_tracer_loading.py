"""
Test loading tracers from TNG-Cluster with illustris_python.
"""
import h5py
import illustris_python as il
import numpy as np
import pytest


@pytest.fixture
def setup():
    """Get necessary data ready"""
    base_path = "/virgotng/mpia/TNG-Cluster/TNG-Cluster/output"
    yield base_path


def test_tracer_ids(setup, subtests):
    """Tracer IDs must be the same in every snapshot"""
    base_path = setup

    for halo_id in [0, 252455, 476245]:
        with subtests.test(msg=f"Cluster {halo_id}"):
            tracer_ids_99 = il.snapshot.loadOriginalZoom(
                base_path, 99, halo_id, 3, ["TracerID"]
            )
            tracer_ids_98 = il.snapshot.loadOriginalZoom(
                base_path, 98, halo_id, 3, ["TracerID"]
            )
            tracer_ids_97 = il.snapshot.loadOriginalZoom(
                base_path, 97, halo_id, 3, ["TracerID"]
            )
            assert tracer_ids_99.shape == tracer_ids_98.shape
            assert tracer_ids_99.shape == tracer_ids_98.shape
            assert tracer_ids_98.shape == tracer_ids_97.shape
            np.testing.assert_equal(tracer_ids_99, tracer_ids_98)
            np.testing.assert_equal(tracer_ids_99, tracer_ids_98)
            np.testing.assert_equal(tracer_ids_98, tracer_ids_97)


def test_parent_ids(setup, subtests):
    """Parent IDs should not be equal throughout snaps, but have same length"""
    base_path = setup

    for halo_id in [0, 252455, 476245]:
        with subtests.test(msg=f"Cluster {halo_id}"):
            parent_ids_99 = il.snapshot.loadOriginalZoom(
                base_path, 99, halo_id, 3, ["ParentID"]
            )
            parent_ids_98 = il.snapshot.loadOriginalZoom(
                base_path, 98, halo_id, 3, ["ParentID"]
            )
            parent_ids_97 = il.snapshot.loadOriginalZoom(
                base_path, 97, halo_id, 3, ["ParentID"]
            )
            assert parent_ids_99.shape == parent_ids_98.shape
            assert parent_ids_99.shape == parent_ids_98.shape
            assert parent_ids_98.shape == parent_ids_97.shape
            assert np.any(np.not_equal(parent_ids_99, parent_ids_98))
            assert np.any(np.not_equal(parent_ids_99, parent_ids_98))
            assert np.any(np.not_equal(parent_ids_98, parent_ids_97))


def load_tracers_directly(base_path, snap_num, halo_id, parent_id=False):
    """Load the tracers of the given halo directly from file"""
    id_type = "ParentID" if parent_id else "TracerID"

    # determine file number
    offset_path = "/../postprocessing/offsets/offsets_099.hdf5"
    with h5py.File(base_path + offset_path) as f:
        file_offset_groups = np.array(f["FileOffsets"]["Group"]).tolist()
    i = file_offset_groups.index(halo_id)

    # open file for fof of halo
    subdir = f"/snapdir_{snap_num:03d}/snap_{snap_num:03d}.{i}.hdf5"
    with h5py.File(base_path + subdir, "r") as f:
        tracer_ids_fof = np.array(f["PartType3"][id_type])
    # open file for fuzz of halo
    subdir = f"/snapdir_{snap_num:03d}/snap_{snap_num:03d}.{i + 352}.hdf5"
    with h5py.File(base_path + subdir, "r") as f:
        tracer_ids_fuzz = np.array(f["PartType3"][id_type])

    # concatenate all tracers and return them
    return np.concatenate([tracer_ids_fof, tracer_ids_fuzz])


def test_tracer_ids_direct_load(setup, subtests):
    """Same as test_tracer_ids but loading tracers directly from file"""
    base_path = setup

    for halo_id in [0, 252455, 476245]:
        with subtests.test(msg=f"Cluster {halo_id}"):
            tracer_ids_99 = load_tracers_directly(base_path, 99, halo_id)
            tracer_ids_98 = load_tracers_directly(base_path, 98, halo_id)
            tracer_ids_97 = load_tracers_directly(base_path, 97, halo_id)
            assert tracer_ids_99.shape == tracer_ids_98.shape
            assert tracer_ids_99.shape == tracer_ids_98.shape
            assert tracer_ids_98.shape == tracer_ids_97.shape
            np.testing.assert_equal(tracer_ids_99, tracer_ids_98)
            np.testing.assert_equal(tracer_ids_99, tracer_ids_98)
            np.testing.assert_equal(tracer_ids_98, tracer_ids_97)


def test_parent_ids_direct_load(setup, subtests):
    """Same as test_parent_ids but loading tracers directly from file"""
    base_path = setup

    for halo_id in [0, 252455, 476245]:
        with subtests.test(msg=f"Cluster {halo_id}"):
            parent_ids_99 = load_tracers_directly(base_path, 99, halo_id, True)
            parent_ids_98 = load_tracers_directly(base_path, 98, halo_id, True)
            parent_ids_97 = load_tracers_directly(base_path, 97, halo_id, True)
            assert parent_ids_99.shape == parent_ids_98.shape
            assert parent_ids_99.shape == parent_ids_98.shape
            assert parent_ids_98.shape == parent_ids_97.shape
            assert np.any(np.not_equal(parent_ids_99, parent_ids_98))
            assert np.any(np.not_equal(parent_ids_99, parent_ids_98))
            assert np.any(np.not_equal(parent_ids_98, parent_ids_97))
