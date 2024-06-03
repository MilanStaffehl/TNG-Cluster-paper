"""
Data acquisition functions for tracer particles
"""
from __future__ import annotations

import logging

import illustris_python as il
import numpy as np
from numpy.typing import NDArray
from typing_extensions import NotRequired, TypedDict


class TracerInfo(TypedDict):
    """Dictionary containing tracer info. Has at most two fields."""
    ParentID: NotRequired[NDArray]
    TracerID: NotRequired[NDArray]


def load_tracers(
    base_path: str,
    snap_num: int,
    fields: list[str] | None = None,
    cluster_id: int = -1
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
    :raises KeyError: When attempting to restrict tracer data to a
        zoom-in region for a simulation that does not have zoom-ins.
    :return: The tracer data as a dictionary mapping the field names to
        the data of integer type. By default, this is a dictionary of
        only two keys, namely ``TracerID`` and ``ParentID``.
    """
    if fields is None:
        fields = ["TracerID", "ParentID"]

    if cluster_id > -1:
        logging.info(
            f"Loading tracer data for cluster {cluster_id} of TNG-Cluster."
        )
        tracer_data = _load_original_zoom_tracers(
            base_path, snap_num, cluster_id, fields
        )
    else:
        logging.info("Loading tracer data for full simulation.")
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
    base_path: str, snap_num: int, cluster_id: int, fields: list[str]
) -> TracerInfo:
    """
    Load tracer data only for original zoom-in region of TNG-Cluster.

    Function is a helper function that attempts to load the tracer info
    required by the ``fields`` argument for only one of the 352 original
    zoom-in regions of the TNG-Cluster simulation. If the simulation base
    path belongs to a different simulation (i.e. one that does not
    provide the ``GroupOrigHaloID`` field for halos), an error message
    is logged and a KeyError raised.

    :param base_path:
    :param snap_num:
    :param cluster_id:
    :param fields:
    :return:
    """
    try:
        tracer_data = il.snapshot.loadOriginalZoom(
            base_path, snap_num, cluster_id, 3, fields=fields
        )
    except AssertionError:
        logging.fatal(
            "Attempted to restrict tracer data to a zoom-in region for a "
            "simulation that is not TNG Cluster, exception will be raised."
        )
        raise KeyError(
            "The simulation group catalogue does not provide the field "
            "'GroupPrimaryZoomTarget'."
        )
    return tracer_data
