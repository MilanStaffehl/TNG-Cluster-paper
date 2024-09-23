"""
DAQ functions for the sublink merger tree.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import illustris_python as il
import numpy as np

from library import units

if TYPE_CHECKING:
    from numpy.typing import NDArray


def get_mpb_properties(
    base_path: str | Path,
    snap_num: int,
    subhalo_id: int,
    fields: list[str],
    start_snap: int | None = None,
    log_warning: bool = False,
) -> dict[str, NDArray | int]:
    """
    Load specified fields along main progenitor branch of the given subhalo.

    Function loads the specified fields of the main progenitor branch of
    the subhalo given by the combination of ID and snapshot number (i.e.
    the ID must be the ID of the subhalo at the specified snapshot). The
    quantities are ordered by snapshot and converted into physical units.
    They are returned in the form of a dictionary, mapping field names to
    the array of values. Each value array is of shape (S, ...) where S is
    the number of loaded snapshots and the values or other axes are
    ordered from lowest to highest snapshot. Note that some MPBs do not
    reach back as far as others, so the exact value of S can vary from
    subhalo to subhalo.

    Optionally, the starting snapshot can be set. In such a case, only
    results starting from that snapshot and later are returned for all
    fields.

    :param base_path: Simulation base path.
    :param snap_num: The snapshot number at which the subhalo ID is valid.
    :param subhalo_id: ID of the subhalo whose main progenitor branch to
        follow.
    :param fields: List of sublink fields to load. ``SnapNum`` is always
        loaded, so it can be omitted.
    :param start_snap: The snapshot from which to start loading the
        data. Optional, if set, only data from this snapshot onward is
        loaded.
    :param log_warning: Whether to emit a warning when values need to
        be interpolated. This can be useful especially for fields where
        interpolation is not sensible (such as IDs or particle numbers).
    :return: Mapping of field names to the corresponding sublink values
        for all snapshots in the MPB history, optionally limited to start
        only at ``start_snap``.
    """
    # correct input
    if isinstance(base_path, Path):
        base_path = str(base_path.resolve())
    if isinstance(fields, str):
        fields = [fields]
    if "SnapNum" not in fields:
        fields.append("SnapNum")

    mpb = il.sublink.loadTree(
        base_path, snap_num, subhalo_id, fields=fields, onlyMPB=True
    )

    # find start snap if not given or invalid
    snaps = mpb["SnapNum"]
    if start_snap is None or start_snap < np.min(snaps):
        start_snap = np.min(snaps)
    # limit snaps to only ones after the start snap
    snaps = snaps[snaps >= start_snap]
    snap_indices = snaps - start_snap  # turn snaps into array indices

    final_results = {}
    warnings = []
    for field, value in mpb.items():
        if field == "SnapNum":
            final_results[field] = np.arange(start_snap, snap_num + 1, step=1)
            continue

        # convert units and limit to desired snap range
        values_raw = units.UnitConverter.convert(value, field, snaps)
        values_raw = values_raw[mpb["SnapNum"] >= start_snap]

        # create array for results and fill with sentinel value
        shape = (snap_num - start_snap + 1, )
        if len(value.shape) > 1:
            shape += value.shape[1:]  # account for ND arrays
        if np.issubdtype(value.dtype, np.integer):
            sentinel = -1
            dtype = np.int64
        else:
            sentinel = np.nan
            dtype = value.dtype
        final_value = np.empty(shape, dtype=dtype)
        final_value[:] = sentinel

        # assign existing values to allocated array
        final_value[snap_indices] = values_raw

        # fill missing entries with interpolated values
        if sentinel == -1:
            where_empty = np.argwhere(final_value == -1)
        else:
            where_empty = np.argwhere(np.isnan(final_value))

        # skip if there is no hole to fill
        if where_empty.size == 0:
            final_results[field] = final_value
            continue

        # 3D vectors produce every index three times, so we limit it to one:
        where_empty = np.unique(where_empty[:, 0])
        if log_warning:
            warnings = where_empty + start_snap
        for index in where_empty:
            before = final_value[index - 1]
            after = final_value[index + 1]
            final_value[index] = (before + after) / 2

        final_results[field] = final_value

    if log_warning and warnings:
        logging.warning(
            f"Interpolated missing main branch progenitor properties for "
            f"subhalo of ID {subhalo_id} (defined at snapshot {snap_num}) at "
            f"snapshots {', '.join(warnings.astype(str))}."
        )
    return final_results
