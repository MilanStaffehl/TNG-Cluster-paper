"""
Functions for data acquisition of clusters in TNG300-1 and TNG-Cluster.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Callable

import h5py
import numpy as np

from library import units
from library.config import config
from library.data_acquisition import bh_daq, halos_daq
from library.processing import selection

if TYPE_CHECKING:
    from numpy.typing import NDArray

N_CLUSTERS = 632  # total number of clusters
N_TNG300 = 280  # number of clusters in TNG300-1


class UnsupportedFieldError(KeyError):
    """Custom exception for unsupported fields."""
    pass


def get_cluster_property(
    field_name: str,
    snap_num: int,
    mass_field: str = "Group_M_Crit200"
) -> NDArray:
    """
    Return the chosen property for all clusters.

    The function supports a fixed set of fields. For these fields, it
    will load the given quantity (e.g. FoF SFR) for the 280 clusters
    in TNG300-1 and the 352 clusters in TNG-Cluster and return them in
    an array of shape (632, ) with the first entries containing the
    values for the TNG300 clusters in order of halo ID, and the latter
    352 entries for the original zoom-in clusters of TNG-Cluster, in
    order of halo ID.

    When asked to load a field that is not supported, the function
    raises a custom exception (UnsupportedFieldError).

    :param field_name: The name of the field to load. Must be one of the
        supported fields for the function.
    :param snap_num: The snapshot number from which to load the data.
    :param mass_field: The name of the field to use as halo mass.
        Required for restriction of TNG300 halos to clusters. Defaults
        to R_200c.
    :raises UnsupportedFieldError: When a field is requested that is not
        supported by the function.
    :return: Array of the requested quantity for all clusters in TNG300-1
        and TNG-Cluster, in that order, of shape (632, ).
    """
    special_fields = {
        "SFRCore": _get_sfr_core,
        "TotalBHMass": _get_total_bh_mass,
        "TotalBHMdot": _get_total_bh_mdot,
        "CentralBHMass": _get_central_bh_mass,
        "CentralBHMode": _get_central_bh_mode,
        "BHCumEnergyFraction": _get_bh_cumulative_energy_fraction,
        "BHCumMassFraction": _get_bh_cumulative_mass_fraction,
        "RelaxednessDist": _get_relaxedness_dist,
        "RelaxednessMass": _get_relaxedness_mass,
        "FormationRedshift": _get_formation_redshift,
        "CentralCoolingTime": _get_central_cooling_time,
    }
    if field_name in special_fields.keys():
        # Field has a specialised acquisition function, so call it
        return special_fields[field_name](snap_num, mass_field)
    # all other fields are either groupcat fields or unsupported
    logging.debug(
        f"Field {field_name} is not in the set of special fields known. "
        f"Attempting to load field from group catalogue."
    )
    try:
        quantity = get_cluster_groupcat_field(field_name, snap_num, mass_field)
    except units.UnsupportedUnitError:
        raise UnsupportedFieldError(f"Unsupported field {field_name}")
    return quantity


def get_cluster_groupcat_field(
    field: str, snap_num: int, mass_field: str = "Group_M_Crit200"
) -> NDArray:
    """
    Get a quantity from the group catalogue for all clusters.

    .. tip:: Prefer the :func:`get_cluster_property` function over this
        function for most cases. It will automatically call this function
        for all fields that require no special treatment and is therefore
        more general-purpose than this function.

    Function loads the given quantity from the group catalogue for the
    280 clusters from TNG300-1 and the 352 clusters from TNG-Cluster.
    It returns them in an array of shape (632, ) with the first 280
    entries being from the TNG300 clusters in order of theit halo ID,
    and the remaining 352 entries from the TNG-Cluster original zoom-in
    clusters, also in order of their halo ID.

    If a field is supplied that is not part of the group catalogue, the
    function raises an UnsupportedFieldError. In such a case, the field
    might be retrievable with the :func:`get_cluster_property` function
    which offers more fields.

    :param field: The field name as it is in the TNG group catalogue.
    :param snap_num: The snapshot from which to laod the data.
    :param mass_field: The name of the field to use as halo mass.
        Required for restriction of TNG300 halos to clusters. Defaults
        to R_200c.
    :return: The quantity loaded from the group catalogue, converted
        into physical units, as an array of shape (632, ?) with the
        TNG300 clusters first. The question mark refers to the possible
        secondary and higher axes of fields containing vectors or
        multiple entries (such as quantities per particle type).
    """
    logging.info(
        f"Loading group catalogue field '{field}' for TNG300-1 and TNG-Cluster."
    )
    quantity = np.zeros(N_CLUSTERS)

    # load and restrict TNG300-1 data
    halo_data = halos_daq.get_halo_properties(
        config.get_simulation_base_path("TNG300-1"),
        snap_num,
        [field, mass_field],
    )
    cluster_data = selection.select_clusters(
        halo_data, mass_field, expected_number=N_TNG300
    )
    quantity[:N_TNG300] = cluster_data[field]

    # load and restrict TNG-Cluster data
    halo_data = halos_daq.get_halo_properties(
        config.get_simulation_base_path("TNG-Cluster"),
        snap_num,
        [field],
        cluster_restrict=True,
    )
    quantity[N_TNG300:] = halo_data[field]

    return quantity


def _get_sfr_core(
    snap_num: int, mass_field: str = "Group_M_Crit200"
) -> NDArray:
    raise NotImplementedError("Core SFR implementation pending.")


def _get_central_bh_mass(
    snap_num: int, mass_field: str = "Group_M_Crit200"
) -> NDArray:
    """
    Return mass of the most massive BH per clusters.

    :return: Array of shape (632, ) of mass of the most massive BH
        in every cluster.
    """
    bh_masses = np.zeros(N_CLUSTERS)
    tng_300_base_path = config.get_simulation_base_path("TNG300-1")
    tng_cluster_basepath = config.get_simulation_base_path("TNG-Cluster")

    # load and restrict TNG300-1 mass data (required for restriction)
    halo_data = halos_daq.get_halo_properties(
        tng_300_base_path,
        snap_num,
        [mass_field],
    )
    cluster_data = selection.select_clusters(
        halo_data, mass_field, expected_number=N_TNG300
    )
    # load the black hole data for every halo
    fields = ["BH_Mass"]
    i = 0
    for halo_id in cluster_data["IDs"]:
        bh_data = bh_daq.get_most_massive_blackhole(
            tng_300_base_path, snap_num, halo_id, fields
        )
        bh_masses[i] = bh_data["BH_Mass"]
        i += 1

    # load TNG-Cluster IDs
    halo_data = halos_daq.get_halo_properties(
        tng_cluster_basepath,
        snap_num,
        [mass_field],
        cluster_restrict=True,
    )
    # load the black hole data for every halo
    for halo_id in halo_data["IDs"]:
        bh_data = bh_daq.get_most_massive_blackhole(
            tng_cluster_basepath, snap_num, halo_id, fields
        )
        bh_masses[i] = bh_data["BH_Mass"]
        i += 1

    return bh_masses


def _get_total_bh_mass(
    snap_num: int, mass_field: str = "Group_M_Crit200"
) -> NDArray:
    """
    Return the total black hole mass per clusters.

    :return: Array of shape (632, ) of black hole masses per cluster.
    """
    bh_masses = np.zeros(N_CLUSTERS)

    # load and restrict TNG300-1 BH masses
    halo_data = halos_daq.get_halo_properties(
        config.get_simulation_base_path("TNG300-1"),
        snap_num,
        ["GroupMassType", mass_field],
    )
    cluster_data = selection.select_clusters(
        halo_data, mass_field, expected_number=N_TNG300
    )
    bh_masses[:N_TNG300] = cluster_data["GroupMassType"][:, 5]

    # load TNG-Cluster BH masses
    halo_data = halos_daq.get_halo_properties(
        config.get_simulation_base_path("TNG-Cluster"),
        snap_num,
        ["GroupMassType"],
        cluster_restrict=True,
    )
    bh_masses[N_TNG300:] = halo_data["GroupMassType"][:, 5]

    return bh_masses


def _get_total_bh_mdot(
    snap_num: int, mass_field: str = "Group_M_Crit200"
) -> NDArray:
    """
    Return mass accretion of all clusters.

    :return: Array of shape (632, ) of total black hole mass accretion
        rate per cluster.
    """
    bh_mdots = np.zeros(N_CLUSTERS)
    tng_300_basepath = config.get_simulation_base_path("TNG300-1")
    tng_cluster_basepath = config.get_simulation_base_path("TNG-Cluster")

    # load and restrict TNG300-1 mass data (required for restriction)
    halo_data = halos_daq.get_halo_properties(
        tng_300_basepath,
        snap_num,
        [mass_field],
    )
    cluster_data = selection.select_clusters(
        halo_data, mass_field, expected_number=N_TNG300
    )
    # load the black hole data for every halo
    fields = ["BH_Mdot"]
    i = 0
    for halo_id in cluster_data["IDs"]:
        bh_data = bh_daq.get_most_massive_blackhole(
            tng_300_basepath, snap_num, halo_id, fields
        )
        bh_mdots[i] = bh_data["BH_Mdot"]
        i += 1

    # load TNG-Cluster IDs
    halo_data = halos_daq.get_halo_properties(
        tng_cluster_basepath,
        snap_num,
        [mass_field],
        cluster_restrict=True,
    )
    # load the black hole data for every halo
    for halo_id in halo_data["IDs"]:
        bh_data = bh_daq.get_most_massive_blackhole(
            tng_cluster_basepath, snap_num, halo_id, fields
        )
        bh_mdots[i] = bh_data["BH_Mdot"]
        i += 1

    return bh_mdots


def _get_central_bh_mode(
    snap_num: int, mass_field: str = "Group_M_Crit200"
) -> NDArray:
    """
    Return central BH mode of all clusters.

    :return: Array of shape (632, ) of black hole mass accretion
        rate per cluster.
    """

    # helper func
    def get_black_hole_mode_index(base_path: str, hid: int) -> float:
        """
        Return the black hole mode index.

        Index is given as the difference between the accretion rate
        MDot and the threshold at which the BH switches over from
        kinetic to thermal mode (the threshold is mass dependent).
        See Weinberger et al. (2017) for details.

        :param base_path: Sim base path.
        :param hid: Halo ID.
        :return: The ratio of the Eddington ratio over the mode
            switchover threshold: (Mdor / Mdot_EDdd) / chi.
        """
        logging.debug(f"Finding black hole mode for halo {hid}.")
        # load all required data
        fields = ["BH_Mass", "BH_Mdot", "BH_MdotEddington"]
        bh_data = bh_daq.get_most_massive_blackhole(
            base_path,
            snap_num,
            hid,
            fields=fields,
        )
        mass = bh_data["BH_Mass"]
        mdot = bh_data["BH_Mdot"]
        eddington_limit = bh_data["BH_MdotEddington"]
        # calculate the threshold for mode switchover
        chi = min(0.002 * (mass / 1e8)**2, 0.1)
        # calculate actual ratio
        eddington_ratio = mdot / eddington_limit
        return eddington_ratio / chi

    return _acquire_cluster_quantity(
        snap_num, get_black_hole_mode_index, "BH mode", mass_field
    )


def _get_bh_cumulative_energy_fraction(
    snap_num: int, mass_field: str = "Group_M_Crit200"
) -> NDArray:
    """
    Return the cumulative energy fraction of most massive BH.

    The fraction is the fraction of the cumulative energy injected
    in kinetic mode over the total energy injected (kinetic + thermal).

    :return: Array of cumulative kinetic energy fraction of most
        massive BH for every cluster.
    """

    # helper func
    def get_black_hole_kinetic_fraction(base_path: str, hid: int) -> float:
        """
        Return the black hole cumulative kinetic energy fraction.

        This fraction is the ratio of the cumulative energy injected
        in kinetic mode over the total cumulative energy injected.

        :param base_path: Sim base path.
        :param hid: Halo ID.
        :return: The ratio of the Eddington ratio over the mode
            switchover threshold: (Mdor / Mdot_EDdd) / chi.
        """
        # load all required data
        fields = ["BH_CumEgyInjection_QM", "BH_CumEgyInjection_RM"]
        bh_data = bh_daq.get_most_massive_blackhole(
            base_path,
            snap_num,
            hid,
            fields=fields,
        )
        total_energy_injected = (
            bh_data["BH_CumEgyInjection_RM"] + bh_data["BH_CumEgyInjection_RM"]
        )
        return bh_data["BH_CumEgyInjection_RM"] / total_energy_injected

    return _acquire_cluster_quantity(
        snap_num,
        get_black_hole_kinetic_fraction,
        "BH kinetic energy injection fraction",
        mass_field,
    )


def _get_bh_cumulative_mass_fraction(
    snap_num: int, mass_field: str = "Group_M_Crit200"
) -> NDArray:
    """
    Return the cumulative mass accretion fraction of most massive BH.

    The fraction is the fraction of the cumulative mass accreted in
    kinetic mode over the total mass accreted (kinetic + thermal).

    :return: Array of cumulative mass accretion fraction of most
        massive BH for every cluster.
    """

    # helper func
    def get_black_hole_kinetic_fraction(base_path: str, hid: int) -> float:
        """
        Return the black hole cumulative kinetic accretion fraction.

        This fraction is the ratio of the cumulative mass accreted
        in kinetic mode over the total cumulative mass accreted.

        :param base_path: Sim base path.
        :param hid: Halo ID.
        :return: The ratio of the Eddington ratio over the mode
            switchover threshold: (Mdor / Mdot_EDdd) / chi.
        """
        # load all required data
        fields = ["BH_CumMassGrowth_QM", "BH_CumMassGrowth_RM"]
        bh_data = bh_daq.get_most_massive_blackhole(
            base_path,
            snap_num,
            hid,
            fields=fields,
        )
        total_mass_accreted = (
            bh_data["BH_CumMassGrowth_QM"] + bh_data["BH_CumMassGrowth_RM"]
        )
        return bh_data["BH_CumMassGrowth_RM"] / total_mass_accreted

    return _acquire_cluster_quantity(
        snap_num,
        get_black_hole_kinetic_fraction,
        "BH kinetic mass accretion fraction",
        mass_field,
    )


def _get_relaxedness_dist(
    snap_num: int, mass_field: str = "Group_M_Crit200"
) -> NDArray:
    """
    Return the relaxedness of the clusters for TNG-Cluster only.

    :return: Array of relaxedness according to distance criterion.
    """
    relaxedness = np.zeros(N_CLUSTERS)
    relaxedness[:N_TNG300] = np.nan
    path = (
        Path(config.get_simulation_base_path("TNG-Cluster")).parent
        / "postprocessing" / "released" / "Relaxedness.hdf5"
    )
    with h5py.File(path, "r") as file:
        relaxedness[N_TNG300:] = np.array(
            file["Halo"]["Distance_Criterion"][99]
        )
    return relaxedness


def _get_relaxedness_mass(
    snap_num: int, mass_field: str = "Group_M_Crit200"
) -> NDArray:
    """
    Return the relaxedness of the clusters for TNG-Cluster only.

    :return: Array of relaxedness according to mass criterion.
    """
    relaxedness = np.zeros(N_CLUSTERS)
    relaxedness[:N_TNG300] = np.nan
    path = (
        Path(config.get_simulation_base_path("TNG-Cluster")).parent
        / "postprocessing" / "released" / "Relaxedness.hdf5"
    )
    with h5py.File(path, "r") as file:
        relaxedness[N_TNG300:] = np.array(file["Halo"]["Mass_Criterion"][99])
    return relaxedness


def _get_formation_redshift(
    snap_num: int, mass_field: str = "Group_M_Crit200"
) -> NDArray:
    """
    Return array of formation redshifts.

    :return: Formation redshifts only for TNG-Cluster clusters.
    """
    # Load data for TNG-Cluster
    formation_z = np.zeros(N_CLUSTERS)
    formation_z[:N_TNG300] = np.nan
    path = (
        Path(config.get_simulation_base_path("TNG-Cluster")).parent
        / "postprocessing" / "released" / "FormationHistories.hdf5"
    )
    with h5py.File(path, "r") as file:
        formation_z[N_TNG300:] = np.array(
            file["Halo"]["Redshift_formation"][:, 99]
        )
    return formation_z


def _get_central_cooling_time(
    snap_num: int, mass_field: str = "Group_M_Crit200"
) -> NDArray:
    """
    Return the central cooling time of clusters in TNG Cluster.

    :return: Central cooling time in Gyr.
    """
    cct = np.zeros(N_CLUSTERS)
    cct[:N_TNG300] = np.nan
    path = (
        Path(config.get_simulation_base_path("TNG-Cluster")).parent
        / "postprocessing" / "released" / "CCcriteria.hdf5"
    )
    with h5py.File(path, "r") as file:
        cct[N_TNG300:] = np.array(file["centralCoolingTime"][:, 99])
    return cct


def _acquire_cluster_quantity(
    snap_num: int,
    call_func: Callable[[str, int], float],
    quantity_descr: str,
    mass_field: str = "Group_M_Crit200",
) -> NDArray:
    """
    Load a quantity for every cluster.

    Given a function ``call_func`` that loads a quantity for a halo
    of a given ID and a simulation base path, return an array of the
    quantities that this function returns for *one* halo for all
    clusters in both TNG300-1 and TNG-Cluster.

    For example, if the function ``call_func`` returns the SFR of
    a halo from a simulation under base path X with ID Y, then the
    return value of this method will be an array of all 632 values
    of the SFR for the clusters in TNG300-1 and TNG-Cluster.

    :param snap_num: The number of the snapshot from which to get data.
    :param call_func: A function of method that takes only two
        arguments namely a simulation base path and a halo ID and
        returns a single quantity for the given halo as float.
    :param quantity_descr: A description of the quantity. Used for
        logging.
    :param mass_field: Name of the mass field to use as halo mass. Used
        to determine which TNG300-1 halos are clusters.
    :return: An array of the quantity returned by ``call_func`` for
        all clusters in TNG300-1 and TNG-Cluster.
    """
    logging.info(f"Loading {quantity_descr} for TNG300-1 and TNG-Cluster.")
    quantity = np.zeros(N_CLUSTERS)
    tng_300_basepath = config.get_simulation_base_path("TNG300-1")
    tng_cluster_basepath = config.get_simulation_base_path("TNG-Cluster")

    # load and restrict TNG300-1 SFRs
    halo_data = halos_daq.get_halo_properties(
        tng_300_basepath,
        snap_num,
        [mass_field],
    )
    cluster_data = selection.select_clusters(
        halo_data, mass_field, expected_number=N_TNG300
    )
    # assign return value of quantity getter to the array
    quantity[:N_TNG300] = np.array(
        [call_func(tng_300_basepath, hid) for hid in cluster_data["IDs"]]
    )

    # load TNG-Cluster IDs
    halo_data = halos_daq.get_halo_properties(
        tng_cluster_basepath,
        snap_num,
        ["GroupLen"],  # dummy field name
        cluster_restrict=True,
    )
    # assign return value of quantity getter to the array
    quantity[N_TNG300:] = np.array(
        [call_func(tng_cluster_basepath, hid) for hid in halo_data["IDs"]]
    )

    return quantity
