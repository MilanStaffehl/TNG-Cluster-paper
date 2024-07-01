"""
Data acquisition functions for tracer particles
"""
from __future__ import annotations

import logging
from pathlib import Path

import h5py
import illustris_python as il
import numpy as np
from numpy.typing import NDArray
from typing_extensions import NotRequired, TypedDict


class TracerInfo(TypedDict):
    """Dictionary containing tracer info. Has at most two fields."""
    count: int
    ParentID: NotRequired[NDArray]
    TracerID: NotRequired[NDArray]


def load_tracers(
    base_path: str,
    snap_num: int,
    fields: list[str] | None = None,
    cluster_id: int = -1,
    zoom_id: int = -1,
) -> TracerInfo:
    """
    Load tracer info from the given simulation and snapshot.

    Function loads the tracer fields specified, or - if none are given -
    the two main fields, namely ``TracerID`` and ``ParentID``. It returns
    them as a dictionary mapping the field name to an array of integers
    for each. The integer tye of the data is enforced, meaning if the
    data is loaded with a different data format (such as a floating point
    format), the type of the array is cast to ``uint64``.

    Optionally, the function supports loading tracers only for the
    original zoom-in regions of TNG-Cluster, by specifying the ID of the
    cluster. If this is attempted for any simulation that is not
    TNG-Cluster or equivalent (i.e. any simulation that does not provide
    the group catalogue field ``GroupOrigHaloID``), the function logs
    an error message and raises a KeyError.

    :param base_path: Path to the simulation data.
    :param snap_num: The snapshot number from which to load the tracers.
    :param fields: The fields to load. Optional, defaults to loading
        both ``TracerID`` and ``ParentID``.
    :param cluster_id: The ID of the original zoom-in cluster, onto which
        to restrict the tracer data. Optional, defaults to -1 which is
        equivalent to loading all tracers from the simulation. Will not
        work for any simulation except TNG-Cluster and equivalent sims.
    :param zoom_id: The ID of the zoom-in region to load. When specified,
        this takes precedence over the ``cluster_id`` argument, i.e. if
        specified, this zoom-in region will be loaded.
    :raises KeyError: When attempting to restrict tracer data to a
        zoom-in region for a simulation that does not have zoom-ins.
    :return: The tracer data as a dictionary mapping the field names to
        the data of integer type. By default, this is a dictionary of
        only two keys, namely ``TracerID`` and ``ParentID``.
    """
    if fields is None:
        fields = ["TracerID", "ParentID"]

    if cluster_id > -1 or zoom_id > -1:
        tracer_data = _load_original_zoom_tracers(
            base_path, snap_num, cluster_id, zoom_id, fields
        )
    else:
        tracer_data = il.snapshot.loadSubset(
            base_path, snap_num, 3, fields=fields
        )

    # repackage single arrays into the expected dict structure
    if not isinstance(tracer_data, dict):
        tracer_data = {fields[0]: tracer_data}

    # coerce type
    for key, data in tracer_data.items():
        if key == "count":
            continue
        if not np.issubdtype(data.dtype, np.uint64):
            logging.debug(
                f"Tracer field {key} has dtype {data.dtype}. "
                f"Correcting to uint64."
            )
            tracer_data[key] = data.astype(np.uint64)

    return tracer_data


def _load_original_zoom_tracers(
    base_path: str,
    snap_num: int,
    cluster_id: int,
    zoom_id: int,
    fields: list[str],
) -> TracerInfo:
    """
    Load tracer data only for original zoom-in region of TNG-Cluster.

    Function is a helper function that attempts to load the tracer info
    required by the ``fields`` argument for only one of the 352 original
    zoom-in regions of the TNG-Cluster simulation. If the simulation base
    path belongs to a different simulation (i.e. one that does not
    provide the ``GroupOrigHaloID`` field for halos), an error message
    is logged and a KeyError raised.

    :param base_path: Base path of TNG-Cluster.
    :param snap_num: The snapshot to load.
    :param cluster_id: The ID of the cluster whose zoom-in to load.
    :param zoom_id: The ID of the zoom-in region to load. Takes precedence
        over ``cluster_id``.
    :param fields: List of fields to load.
    :return: Dictionary of tracer data.
    """
    base_path = Path(base_path)
    offset_file = base_path / "../postprocessing/offsets/offsets_099.hdf5"

    # check if we are actually working with TNG-Cluster
    test_data = il.groupcat.loadSingle(str(base_path), snap_num, haloID=0)
    if "GroupOrigHaloID" not in test_data.keys():
        logging.error(
            "Tried loading original zoom-in of a simulation that is not "
            "TNG-Cluster. Returning empty data."
        )
        return {"count": 0}

    if zoom_id > -1:
        file_index = zoom_id
    else:
        # load cluster IDs to get to file index
        with h5py.File(str(offset_file.resolve()), "r") as file:
            cluster_ids = np.array(file["FileOffsets"]["Group"]).tolist()
        file_index = cluster_ids.index(cluster_id)

    # load the tracer data from the two files
    snapshot_path = (base_path / f"snapdir_{snap_num:03d}/")

    fof_file = f"snap_{snap_num:03d}.{file_index}.hdf5"
    with h5py.File(str(snapshot_path / fof_file), "r") as file:
        tracer_ids_fof = np.array(file["PartType3"]["TracerID"], np.uint64)
        parent_ids_fof = np.array(file["PartType3"]["ParentID"], np.uint64)

    fuzz_file = f"snap_{snap_num:03d}.{file_index + 352}.hdf5"
    with h5py.File(str(snapshot_path / fuzz_file), "r") as file:
        tracer_ids_fuzz = np.array(file["PartType3"]["TracerID"], np.uint64)
        parent_ids_fuzz = np.array(file["PartType3"]["ParentID"], np.uint64)

    # construct and return data array
    tracer_data = {
        "count": len(tracer_ids_fof) + len(tracer_ids_fuzz),
    }
    if "TracerID" in fields:
        tracer_data.update(
            {"TracerID": np.concatenate([tracer_ids_fof, tracer_ids_fuzz])}
        )
    if "ParentID" in fields:
        tracer_data.update(
            {"ParentID": np.concatenate([parent_ids_fof, parent_ids_fuzz])}
        )
    return tracer_data
