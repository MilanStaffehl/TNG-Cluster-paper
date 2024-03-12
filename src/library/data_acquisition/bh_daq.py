"""
Functions to obtain black hole data.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Sequence

import illustris_python as il
import numpy as np

from library import units

if TYPE_CHECKING:
    from numpy.typing import NDArray


def get_most_massive_blackhole(
    base_path: str,
    snap_num: int,
    halo_id: int,
    fields: Sequence[str],
) -> dict[str, float | NDArray]:
    """
    Return data for the most massive black hole of the selected halo.

    The function returns a data dictionary as the illustris_python
    helper scripts would, but limited only to the most massive black
    hole of the halo of the given ID. The quantities are converted to
    have physical units.

    :param base_path: Base path of the simulation to use.
    :param snap_num: The number of the snapshot to look up.
    :param halo_id: The ID of the halo for which to find the most massive
        BH of.
    :param fields: A list of field names to load and to return for the
        most massive halo.
    :return: The data dictionary, limited to only the most massive BH.
        Note that scalar quantities will be set as scalars in the
        values of the dict, not arrays! Similarly, vector quantitites
        will be the vector itself, not a vector nested inside a length
        one array.
    """
    # create field list
    if not isinstance(fields, list):
        fields = list(fields)
    if "BH_Mass" not in fields:
        fields.append("BH_Mass")
    logging.debug(
        f"Restricting black hole fields {', '.join(fields)} to most massive "
        f"black hole of halo {halo_id}."
    )

    # load all required data
    data = il.snapshot.loadHalo(
        base_path,
        snap_num,
        halo_id,
        5,
        fields=fields,
    )
    if data["count"] == 0:
        logging.warning(
            f"Halo {halo_id} has no black holes, cannot provide data for "
            f"fields {', '.join(fields)}! Will return NaNs instead for all "
            f"fields."
        )
        return data.update({field: np.nan for field in fields})

    # determine index of most massive BH, extract its data
    central_idx = np.argmax(data["BH_Mass"])
    if central_idx != 0:
        logging.debug(
            f"Most massive black hole does not have index 0 for halo "
            f"{halo_id}. Most massive black hole has index {central_idx}."
        )

    # create restricted dict and convert units
    central_data = {}
    for field, data in data.items():
        if field == "count":
            continue
        central_data[field] = units.UnitConverter.convert(
            data[central_idx], field
        )
    return central_data
