"""
Function for data acquisition of all particle cells.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal, Sequence

import h5py
import illustris_python as il
import numpy as np

from library import units

if TYPE_CHECKING:
    from numpy.typing import NDArray


def get_particle_ids(
    base_path: str,
    snap_num: int,
    part_type: Literal[0, 4, 5],
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


def get_particle_properties(
    base_path: str,
    snap_num: int,
    part_type: Literal[0, 4, 5],
    fields: list[str],
    *,
    zoom_id: int | None = None,
) -> dict[str, NDArray] | None:
    """
    Return array of properties of the given particle type.

    :param base_path: Base path of the simulation.
    :param snap_num: Snapshot from which to load the data.
    :param part_type: The particle type to load. Must be either 0 for gas,
        4 for stars, or 5 for black holes.
    :param fields: List of fields to load. Must be available for the
        given particle type.
    :param zoom_id: When loading data from TNG-Cluster, set this to the
        ID of the original zoom-in region to load only particles from
        this zoom-in.
    :return: Dictionary mapping field names to arrays of values in
        physical units. If loading fails, returns None instead.
    """
    if not isinstance(fields, Sequence):
        fields = list(fields)

    if zoom_id is None:
        part_data = il.snapshot.loadSubset(
            base_path, snap_num, part_type, fields=fields, float32=True
        )
        if isinstance(part_data, np.ndarray):
            part_data = {fields[0]: part_data}
    else:
        part_data = _load_original_zoom_particle_properties(
            base_path, snap_num, part_type, zoom_id, fields
        )
        if part_data is None:
            return None

    # convert units
    part_data_physical = {}
    for field, value in part_data.items():
        part_data_physical[field] = units.UnitConverter.convert(
            value, field, snap_num
        )
    return part_data_physical


def _load_original_zoom_particle_properties(
    base_path: str,
    snap_num: int,
    part_type: Literal[0, 4, 5],
    zoom_id: int,
    fields: list[str],
) -> dict[str, NDArray | int] | None:
    """
    Load particle properties from one of the original zoom-ins of TNG-Cluster.

    Functions loads all particles of the specified original zoom-in
    (halo, inner and outer fuzz), and returns the specified list of
    fields as values in a dictionary mapping the field names to an array
    of values. The values are in code units.

    :param base_path: Base path of TNG-Cluster.
    :param snap_num: The snapshot to load.
    :param zoom_id: The index/ID of the zoom-in region to load. Must be
        a number between 0 and 351.
    :param part_type: The particle type as integer.
    :param fields: A list of fields to load.
    :return: Dictionary mapping field names to values in code units.
    """
    if zoom_id < 0 or zoom_id > 351:
        logging.error(f"Invalid zoom-in region ID: {zoom_id}.")
        return None

    # check if we are actually working with TNG-Cluster
    test_data = il.groupcat.loadSingle(base_path, snap_num, haloID=0)
    if "GroupOrigHaloID" not in test_data.keys():
        logging.error(
            "Tried loading original zoom-in of a simulation that is not "
            "TNG-Cluster. Returning empty data."
        )
        return None

    # temporary dict
    temp = {field: [] for field in fields}

    # locate and load files
    snapshot_path = base_path + f"/snapdir_{snap_num:03d}/"
    fof_file = f"snap_{snap_num:03d}.{zoom_id}.hdf5"
    with h5py.File(str(snapshot_path + fof_file), "r") as file:
        for field in fields:
            try:
                particles_fof = file[f"PartType{part_type}"][field][()]
            except KeyError:
                pass  # simply skip this one (sometimes they are empty)
            else:
                temp[field].append(particles_fof)

    fuzz_file = f"snap_{snap_num:03d}.{zoom_id + 352}.hdf5"
    with h5py.File(str(snapshot_path + fuzz_file), "r") as file:
        for field in fields:
            try:
                particles_fuzz = file[f"PartType{part_type}"][field][()]
            except KeyError:
                pass
            else:
                temp[field].append(particles_fuzz)

    # concatenate data
    try:
        data = {f: np.concatenate(v, axis=0) for f, v in temp.items()}
        return data
    except ValueError:
        logging.error(
            f"At least one field of {fields} did not exist for TNG-Cluster "
            f"particles of type {part_type}."
        )
        return None
