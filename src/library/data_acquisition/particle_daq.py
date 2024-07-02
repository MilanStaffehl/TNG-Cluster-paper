"""
Function for data acquisition of all particle cells.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import h5py
import illustris_python as il
import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


def get_particle_ids(
    base_path: str,
    snap_num: int,
    part_type: int,
    *,
    cluster: int | None = None,
    zoom_id: int | None = None,
) -> NDArray:
    """
    Load and return the unique particle IDs of all particles of ``part_type``.

    This function loads and returns the particle IDs of all particles
    of the given type from the given snapshot and returns them as an
    array of dtype uint64.

    :param base_path: The base path of the simulation to use.
    :param snap_num: The snapshot number from which to load the data.
    :param part_type: The particle type as integer.
    :param cluster: When loading data from TNG-Cluster, set this to the
        ID of the halo of whose original zoom to load particle IDs, to
        avoid loading filler particles and particles from other zooms.
        If not set, even when using TNG-Cluster, this function will load
        all particles of the simulation. Setting this to anything other
        than None for any simulation except TNG-Cluster will cause an
        error.
    :param zoom_id: Alternative to ``cluster``: This must be an integer
        between 0 and 351, giving the index of the original zoom-in
        region. The return is functionally identical to using a halo ID
        with ``cluster``, except that the index of the zoom-in region is
        constant across all snapshots, whereas halos have different IDs
        in different snapshots. This means, setting ``zoom_id=1`` will
        _always_ load the particles of the second zoom-in region, while
        ``cluster=252455`` only loads this region for snapshot 99 and
        will load a different zoom-in for other snapshots, due to halos
        having different IDs in every snapshot. If ``cluster`` is given,
        ``zoom_id`` is ignored.
    :return: A list of particle IDs of all particles type ``part_type``.
    """
    # load gas particle data
    fields = ["ParticleIDs"]
    if cluster is not None:
        particle_ids = il.snapshot.loadOriginalZoom(
            base_path, snap_num, cluster, partType=part_type, fields=fields
        )
    elif zoom_id is not None:
        particle_ids = _load_original_zoom_particle_ids(
            base_path, snap_num, zoom_id=zoom_id, part_type=part_type
        )
    else:
        particle_ids = il.snapshot.loadSubset(
            base_path,
            snap_num,
            partType=part_type,
            fields=fields,
        )
    return particle_ids


def _load_original_zoom_particle_ids(
    base_path: str,
    snap_num: int,
    zoom_id: int,
    part_type: int,
) -> NDArray:
    """
    Load all particle IDs from one of the original zoom-ins of TNG-Cluster.

    Functions loads all particle IDs of the specified original zoom-in
    (halo, inner and outer fuzz), and returns them as an array.

    :param base_path: Base path of TNG-Cluster.
    :param snap_num: The snapshot to load.
    :param zoom_id: The index/ID of the zoom-in region to load. Must be
        a number between 0 and 351.
    :param part_type: The particle type as integer.
    :return: An array of particle IDs.
    """
    if zoom_id < 0 or zoom_id > 351:
        logging.error(f"Invalid zoom-in region ID: {zoom_id}.")
        return np.empty((0, ), dtype=np.uint64)

    # check if we are actually working with TNG-Cluster
    test_data = il.groupcat.loadSingle(base_path, snap_num, haloID=0)
    if "GroupOrigHaloID" not in test_data.keys():
        logging.error(
            "Tried loading original zoom-in of a simulation that is not "
            "TNG-Cluster. Returning empty data."
        )
        return np.empty((0, ), dtype=np.uint64)

    # locate and load files
    snapshot_path = base_path + f"/snapdir_{snap_num:03d}/"
    fof_file = f"snap_{snap_num:03d}.{zoom_id}.hdf5"
    with h5py.File(str(snapshot_path + fof_file), "r") as file:
        try:
            particle_ids_fof = file[f"PartType{part_type}"]["ParticleIDs"][()]
        except KeyError:
            particle_ids_fof = np.empty((0, ), dtype=np.uint64)

    fuzz_file = f"snap_{snap_num:03d}.{zoom_id + 352}.hdf5"
    with h5py.File(str(snapshot_path + fuzz_file), "r") as file:
        try:
            particle_ids_fuzz = file[f"PartType{part_type}"]["ParticleIDs"][()]
        except KeyError:
            particle_ids_fuzz = np.empty((0, ), dtype=np.uint64)

    # concatenate data
    return np.concatenate([particle_ids_fof, particle_ids_fuzz], axis=0)
