"""
Functions to obtain halo data.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Sequence

import illustris_python as il
import numpy as np
import numpy.ma as ma

from library import units
from library.processing import statistics

if TYPE_CHECKING:
    from numpy.typing import NDArray


def get_halo_properties(
    base_path: str,
    snap_num: int,
    fields: list[str],
) -> dict[str, NDArray]:
    """
    Load halo properties and return them plus a list of IDs.

    Function takes a list of halo fields and loads them for all halos in
    the given simulation. Units are converted into physical units. Only
    for specific data fields, unit conversion exists, so only these
    fields may be supplied.

    :param base_path: The base path of the simulation to use.
    :param snap_num: The snapshot number from which to load the data.
    :param fields: The list of fields to load. Must match the name of the
        field in the simulation.
    :return: A dictionary of the field values for every halo, including
        a list of halo IDs.
    """
    # verify and correct input types
    if not isinstance(fields, list):
        logging.warning(
            "Received a string instead of a list of fields for halo data "
            "acquistion. Please use a list of fields."
        )
        fields = [fields]

    logging.info("Loading halo properties.")
    # verify units can be converted
    supported = units.UnitConverter.supported_fields()
    for field in fields:
        if field not in supported:
            raise units.UnsupportedUnitError(field)

    # load halo properties
    halo_data = il.groupcat.loadHalos(base_path, snap_num, fields=fields)
    # turn arrays into dictionaries as expected
    if not isinstance(halo_data, dict):
        halo_data = {fields[0]: halo_data}

    # create ids
    num_halos = len(halo_data[fields[0]])
    ids = np.indices([num_halos], sparse=True)[0]

    # convert units
    halo_data_physical = {}
    for field, data in halo_data.items():
        halo_data_physical[field] = units.UnitConverter.convert(data, field)
    halo_data_physical["IDs"] = ids
    logging.info("Finished loading halo masses & radii.")
    return halo_data_physical


def select_halo_data_subset(
    base_path: str,
    snap_num: int,
    fields: list[str],
    mass_bins: Sequence[float],
    min_select: int,
    mass_field: str = "Group_M_Crit200"
) -> dict[str, NDArray]:
    """
    Load and bin the halos from the simulation, save binned masses.

    The method loads data for all halos as specified by ``fields`` and
    bins the halos by mass into the specified mass bins. For every mass
    bin it then selects twice the number of halos specified by
    ``min_select`` and returns a reduced halo data dictionary with
    entries for only those halos. The resulting arrays inside the dict
    values are sorted by mass bin, i.e. the first 2 * ``min_select``
    entries belong to the first mass bin, the next 2 * ``min_select``
    entries belong to the second mass bin etc.

    :param base_path: The base path of the simulation to use.
    :param snap_num: The snapshot number from which to load the data.
    :param fields: The list of fields to load. Must match the name of the
        field in the simulation.
    :param mass_bins: A sequence of mass bin edges in units of solar
        masses.
    :param min_select: The minimum required number of halos selected
        per mass bin. This method will select twice as many halos in
        order to avoid empty halos polluting the sample.
    :param mass_field: The name of the masss field. Defaults to
        'Group_M_Crit200', i.e. the virial radius.
    :return: A halo data dictionary with the keys being the fields
        specified plus an additional field 'IDs', and the corresponding
        values being the values of these fields for the selected and
        binned halos. Entries are ordered such that every chunk of N
        entries belongs to one mass bin, where N = 2 * ``min_select``.
        As such, every value is an array of length N * M where M is the
        number of mass bins: ``M = len(mass_bins) - 1``.
    """
    halo_data = get_halo_properties(base_path, snap_num, fields)
    mass_bin_mask = statistics.sort_masses_into_bins(
        halo_data[mass_field], mass_bins
    )

    # for every mass bin, select twice as many halos as needed (to
    # have a backup when empty halos are selected by accident)
    logging.info("Selecting subset of halos for gallery.")
    select_ids = np.zeros((len(mass_bins) - 1) * 2 * min_select, dtype=int)
    for bin_num in range(len(mass_bins) - 1):
        mask = np.where(mass_bin_mask == bin_num + 1, 1, 0)
        masked_ids = ma.masked_array(halo_data["IDs"]).compress(mask)
        masked_ids = masked_ids.compressed()
        n = 2 * min_select  # number of halos to select per bin
        # choose entries randomly
        rng = np.random.default_rng()
        select_ids[bin_num * n:(bin_num + 1) * n] = rng.choice(
            masked_ids, size=n, replace=False
        )
    # mask fields and construct dict
    select_halo_data = {}
    for field in fields:
        select_halo_data[field] = halo_data[field][select_ids]
    logging.info("Finished loading and selecting halo masses & radii.")
    return select_halo_data
