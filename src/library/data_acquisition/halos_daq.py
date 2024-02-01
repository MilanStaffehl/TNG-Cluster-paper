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
from library.processing import selection

if TYPE_CHECKING:
    from numpy.typing import NDArray


def get_halo_properties(
    base_path: str,
    snap_num: int,
    fields: list[str],
    cluster_restrict: bool = False,
) -> dict[str, NDArray | int]:
    """
    Load halo properties and return them plus a list of IDs.

    Function takes a list of halo fields and loads them for all halos in
    the given simulation. Units are converted into physical units. Only
    for specific data fields, unit conversion exists, so only these
    fields may be supplied.

    For TNG Cluster, the option to restrict the halo data to only the
    352 zoom-in clusters exists.

    .. warning:: Do not set ``cluster_restrict`` to True for any other
        simulation! The function will not attempt to check whether the
        base path actually belongs to TNG Cluster! Attempting to restrict
        data from a simulation other than TNG Cluster will raise a
        KeyError due to the respective field for restriction not existing
        in the group catalogue.

    :param base_path: The base path of the simulation to use.
    :param snap_num: The snapshot number from which to load the data.
    :param fields: The list of fields to load. Must match the name of the
        field in the simulation.
    :param cluster_restrict: When loading data from TNG Cluster, set this
        option to True in order to restrict the returned data to only
        the halo data of the original zoom-in clusters (352 clusters in
        total). Has no effect when using other simulations.
    :raises UnsupportedUnitError: If one of the fields has a unit that
        cannot be converted into physical units.
    :raises KeyError: When attempting to restrict halo data for any
        simulation that is not TNG Cluster.
    :return: A dictionary of the field values for every halo, including
        a list of halo IDs.
    """
    # verify and correct input types
    if isinstance(fields, str):
        fields = [fields]
    elif not isinstance(fields, list):
        # attempt cast
        fields = list(fields)

    logging.info(f"Loading halo properties {', '.join(fields)}.")
    # verify units can be converted (done first to avoid loading data
    # that cannot be converted later anyway)
    supported = units.UnitConverter.supported_fields()
    for field in fields:
        if field not in supported:
            logging.fatal(f"No unit conversion fo field {field} available.")
            raise units.UnsupportedUnitError(field)

    # load halo properties
    halo_data = il.groupcat.loadHalos(base_path, snap_num, fields=fields)
    # turn arrays into dictionaries as expected
    if not isinstance(halo_data, dict):
        halo_data = {fields[0]: halo_data}  # There's only one field

    # create ids
    num_halos = len(halo_data[fields[0]])
    ids = np.indices([num_halos], sparse=True)[0]
    halo_data["IDs"] = ids

    # restrict halo data
    if cluster_restrict:
        halo_data = _restrict_cluster_data(base_path, snap_num, halo_data)

    # convert units
    halo_data_physical = {}
    for field, data in halo_data.items():
        halo_data_physical[field] = units.UnitConverter.convert(data, field)
    logging.info("Finished loading halo properties.")
    return halo_data_physical


def _restrict_cluster_data(
    base_path: str,
    snap_num: int,
    data: dict[str, NDArray | int],
) -> dict[str, NDArray | int]:
    """
    Helper function, restricts TNG Cluster halo data to zoom-in clusters.

    The function will take the halo data dictionary and mask all values
    in its value fields that do not belong to one of the 352 original
    zoom-in halos. This will drastically reduce the size of the data
    and allows to handle only the relevant data.

    If any simulation base path other than that of TNG Cluster is given,
    the function will raise a KeyError due to the missing group catalogue
    field ``GroupPrimaryZoomTarget``.

    :param base_path: The base path of the simulation to use.
    :param snap_num: The snapshot number from which to load the data.
    :param data: The mapping of field names to halo data in arrays as
        returned by the ``illustris_python`` groupcat helper scripts.
    :raises KeyError: When attempting to restrict halo data for any
        simulation that is not TNG Cluster.
    :return: The mapping of field names to halo data, but containing
        only the halo data of the original 352 primary zoom-in clusters.
    """
    logging.info("Restricting halo data to original zoom-in clusters only.")
    # create a mask for the data
    try:
        is_primary = il.groupcat.loadHalos(
            base_path, snap_num, fields=["GroupPrimaryZoomTarget"]
        )
    except Exception as e:
        errmsg = (
            "Group catalog does not have requested field [GroupPrimaryZoomTarget]!"
        )
        if str(e) == errmsg:
            logging.fatal(
                "Attempted to restrict halo data for a simulation that "
                "is not TNG Cluster, exception will be raised."
            )
            raise KeyError(
                "The simulation group catalogue does not provide the field "
                "'GroupPrimaryZoomTarget'."
            )
        else:
            raise

    # verify the data is from TNG Cluster
    n_clusters_found = np.count_nonzero(is_primary)
    if n_clusters_found != 352:
        logging.warning(
            "The restricted halo data will not have the expected 352 halos, "
            f"but {n_clusters_found} instead. This might indicate a problem."
        )

    # mask data
    restricted_data = {}
    for field, value in data.items():
        if field == "count":
            restricted_data[field] = n_clusters_found
        else:
            restricted_data[field] = selection.mask_quantity(
                value, is_primary, index=1
            )
    return restricted_data


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
    mass_bin_mask = np.digitize(halo_data[mass_field], mass_bins)

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
