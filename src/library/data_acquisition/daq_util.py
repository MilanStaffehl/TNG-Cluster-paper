"""
Common utility functions for DAQ modules.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal

import h5py
import illustris_python as il
import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


def load_original_zoom_particle_properties(
    base_path: str,
    snap_num: int,
    part_type: Literal[0, 4, 5],
    zoom_id: int,
    fields: list[str],
) -> dict[str, NDArray | int]:
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
    :return: Dictionary mapping field names to values in code units. If
        loading fails for any reason, the return value is a data dictionary
        representing empty data, i.e. ``{"count": 0}``.
    """
    if zoom_id < 0 or zoom_id > 351:
        logging.error(f"Invalid zoom-in region ID: {zoom_id}.")
        return {"count": 0}

    # check if we are actually working with TNG-Cluster
    test_data = il.groupcat.loadSingle(base_path, snap_num, haloID=0)
    if "GroupOrigHaloID" not in test_data.keys():
        logging.error(
            "Tried loading original zoom-in of a simulation that is not "
            "TNG-Cluster. Returning empty data."
        )
        return {"count": 0}

    # create a dictionary for the data
    raw_data = {f: list() for f in fields}

    # locate and load files
    snapshot_path = base_path + f"/snapdir_{snap_num:03d}/"
    fof_file = f"snap_{snap_num:03d}.{zoom_id}.hdf5"
    with h5py.File(str(snapshot_path + fof_file), "r") as file:
        for field in fields:
            try:
                raw_data[field].append(file[f"PartType{part_type}"][field][()])
            except KeyError:
                raw_data[field].append(np.empty((0, )))

    fuzz_file = f"snap_{snap_num:03d}.{zoom_id + 352}.hdf5"
    with h5py.File(str(snapshot_path + fuzz_file), "r") as file:
        for field in fields:
            try:
                raw_data[field].append(file[f"PartType{part_type}"][field][()])
            except KeyError:
                raw_data[field].append(np.empty((0, )))

    # concatenate data
    data = {k: np.concatenate(v, axis=0) for k, v in raw_data.items()}
    data["count"] = data[fields[0]].shape[0]
    if data["count"] == 0:
        logging.warning(
            f"At least one particle property of {fields} did not exist for "
            f"TNG-Cluster particles of type {part_type} at snapshot "
            f"{snap_num}; the 'count' field of the returned data dict may "
            f"incorrectly be set to 0 or the dict might contain empty data "
            f"fields."
        )
    return data
