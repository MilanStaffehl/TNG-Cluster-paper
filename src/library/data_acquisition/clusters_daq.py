"""
Functions for data acquisition of clusters in TNG300-1 and TNG-Cluster.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Callable

import h5py
import illustris_python as il
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
        "GasMetallicityCore": _get_metallicity_core,
        "MassiveSatellites": _get_number_of_massive_satellites,
        "TotalBHMass": _get_total_bh_mass,
        "TotalBHMdot": _get_total_bh_mdot,
        "CentralBHMass": _get_central_bh_mass,
        "CentralBHMode": _get_central_bh_mode,
        "BHCumEnergyFraction": _get_bh_cumulative_energy_fraction,
        "BHCumTotalEnergy": _get_bh_total_cumulative_energy,
        "BHCumKineticEnergy": _get_bh_cumulative_kinetic_energy,
        "BHCumThermalEnergy": _get_bh_cumulative_thermal_energy,
        "BHCumMassFraction": _get_bh_cumulative_mass_fraction,
        "BHCumTotalMass": _get_bh_total_cumulative_mass,
        "BHCumKineticMass": _get_bh_cumulative_kinetic_mass,
        "BHCumThermalMass": _get_bh_cumulative_thermal_mass,
        "BHProgenitors": _get_bh_number_of_progenitors,
        "TotalStellarMass": _get_total_stellar_mass,
        "CentralStellarMass": _get_central_stellar_mass,
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
    """
    Return SFRs of the primary subhalo of all clusters.

    :param snap_num: Snapshot number to load.
    :param mass_field: Mass field name.
    :return: Array of shape (632, ) of SFR in primary subhalo in every
        cluster.
    """

    def get_cluster_primary_subhalo_sfr(base_path: str, halo_id: int) -> float:
        """
        Helper function; returns the SFR of the primary subhalo.

        :param base_path: Base path of the simulation.
        :param halo_id: ID of the halo for which to load the data.
        :return:
        """
        halo_data = il.groupcat.loadSingle(base_path, snap_num, haloID=halo_id)
        primary_subhalo_id = halo_data["GroupFirstSub"]
        # load subhalo data
        subhalo_data = il.groupcat.loadSingle(
            base_path, snap_num, subhaloID=primary_subhalo_id
        )
        return units.UnitConverter.convert(
            subhalo_data["SubhaloSFR"], "SubhaloSFR"
        )

    return _acquire_cluster_quantity(
        snap_num,
        get_cluster_primary_subhalo_sfr,
        "primary subhalo SFR",
        mass_field
    )


def _get_metallicity_core(
    snap_num: int, mass_field: str = "Group_M_Crit200"
) -> NDArray:
    """
    Return metallicities of the primary subhalo of all clusters.

    :param snap_num: Snapshot number to load.
    :param mass_field: Mass field name.
    :return: Array of shape (632, ) of metallicity in primary subhalo in every
        cluster in solar units.
    """

    def get_cluster_primary_subhalo_z(base_path: str, halo_id: int) -> float:
        """
        Helper function; returns the metallicity of the primary subhalo.

        :param base_path: Base path of the simulation.
        :param halo_id: ID of the halo for which to load the data.
        :return:
        """
        halo_data = il.groupcat.loadSingle(base_path, snap_num, haloID=halo_id)
        primary_subhalo_id = halo_data["GroupFirstSub"]
        # load subhalo data
        subhalo_data = il.groupcat.loadSingle(
            base_path, snap_num, subhaloID=primary_subhalo_id
        )
        return units.UnitConverter.convert(
            subhalo_data["SubhaloGasMetallicity"], "SubhaloGasMetallicity"
        )

    return _acquire_cluster_quantity(
        snap_num,
        get_cluster_primary_subhalo_z,
        "primary subhalo metallicity",
        mass_field
    )


def _get_number_of_massive_satellites(
    snap_num: int, mass_field: str = "Group_M_Crit200"
) -> NDArray:
    """
    Return number of satellites above 10^9 solar masses of all clusters.

    :param snap_num: Snapshot number to load.
    :param mass_field: Mass field name.
    :return: Array of shape (632, ) of the number of satellites with
        mass > 10^9 solar masses per cluster.
    """
    n_massive_satellites = np.zeros(N_CLUSTERS)
    tng_300_base_path = config.get_simulation_base_path("TNG300-1")
    tng_cluster_basepath = config.get_simulation_base_path("TNG-Cluster")

    # load mass for all subhalos in TNG300-1
    logging.info("Loading subhalo masses for TNG300-1 subhalos.")
    satellite_masses = il.groupcat.loadSubhalos(
        tng_300_base_path, snap_num, fields=["SubhaloMass"]
    )
    satellite_masses = units.UnitConverter.convert(
        satellite_masses, "SubhaloMass"
    )

    # load cluster data for TNG300-1
    halo_data = halos_daq.get_halo_properties(
        tng_300_base_path,
        snap_num,
        fields=["GroupFirstSub", "GroupNsubs", mass_field],
    )
    cluster_data = selection.select_clusters(
        halo_data, mass_field, expected_number=N_TNG300
    )

    # for every cluster, find the number of massive satellites
    logging.debug("Start processing TNG300-1")
    for i, n_satellites in enumerate(cluster_data["GroupNsubs"]):
        # get ID/index of the first subhalo, excluding the primary subhalo
        first_idx = cluster_data["GroupFirstSub"][i] + 1
        # select masses of subhalos belonging to current cluster
        cur_masses = satellite_masses[first_idx:first_idx + n_satellites - 1]
        # select from those only massive subhalos
        massive_satellites = cur_masses[cur_masses > 1e9]
        # count number of massive subhalos
        n_massive_satellites[i] = len(massive_satellites)

    # load mass for all subhalos in TNG-Cluster
    logging.info("Loading subhalo masses for TNG-Cluster subhalos.")
    satellite_masses = il.groupcat.loadSubhalos(
        tng_cluster_basepath, snap_num, fields=["SubhaloMass"]
    )
    satellite_masses = units.UnitConverter.convert(
        satellite_masses, "SubhaloMass"
    )

    # load cluster data for TNG-Cluster
    halo_data = halos_daq.get_halo_properties(
        tng_cluster_basepath,
        snap_num,
        fields=["GroupFirstSub", "GroupNsubs"],
        cluster_restrict=True,
    )

    # find number of massive satellites per cluster
    logging.info("Start processing TNG-Cluster")
    for i, n_satellites in enumerate(halo_data["GroupNsubs"]):
        # get ID/index of the first subhalo, excluding the primary subhalo
        first_idx = halo_data["GroupFirstSub"][i] + 1
        # select masses of subhalos belonging to current cluster
        cur_masses = satellite_masses[first_idx:first_idx + n_satellites - 1]
        # select from those only massive subhalos
        massive_satellites = cur_masses[cur_masses > 1e9]
        # count number of massive subhalos
        n_massive_satellites[i + N_TNG300] = len(massive_satellites)

    return n_massive_satellites


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
            bh_data["BH_CumEgyInjection_QM"] + bh_data["BH_CumEgyInjection_RM"]
        )
        return bh_data["BH_CumEgyInjection_RM"] / total_energy_injected

    return _acquire_cluster_quantity(
        snap_num,
        get_black_hole_kinetic_fraction,
        "BH kinetic energy injection fraction",
        mass_field,
    )


def _get_bh_total_cumulative_energy(
    snap_num: int, mass_field: str = "Group_M_Crit200"
) -> NDArray:
    """
    Return the cumulative energy ejected of most massive BH.

    The total is the sum of the cumulative energy injected
    in kinetic mode and the energy injected in thermal mode.

    :return: Array of cumulative total energy of most massive BH for
        every cluster.
    """

    # helper func
    def get_black_hole_total_energy(base_path: str, hid: int) -> float:
        """
        Return the black hole cumulative total energy.

        :param base_path: Sim base path.
        :param hid: Halo ID.
        :return: The sum of the energy in thermal mode and the energy in
            kinetic mode.
        """
        # load all required data
        fields = ["BH_CumEgyInjection_QM", "BH_CumEgyInjection_RM"]
        bh_data = bh_daq.get_most_massive_blackhole(
            base_path,
            snap_num,
            hid,
            fields=fields,
        )
        return (
            bh_data["BH_CumEgyInjection_RM"] + bh_data["BH_CumEgyInjection_RM"]
        )

    return _acquire_cluster_quantity(
        snap_num,
        get_black_hole_total_energy,
        "BH total energy",
        mass_field,
    )


def _get_bh_cumulative_kinetic_energy(
    snap_num: int, mass_field: str = "Group_M_Crit200"
) -> NDArray:
    """
    Return the cumulative kinetic energy ejected of most massive BH.

    :return: Array of cumulative kinetic energy of most massive BH for
        every cluster.
    """

    # helper func
    def get_black_hole_kinetic_energy(base_path: str, hid: int) -> float:
        """
        Return the black hole cumulative kinetic energy.

        :param base_path: Sim base path.
        :param hid: Halo ID.
        :return: The kinetic energy injected over the lifetime of the BH.
        """
        # load all required data
        bh_data = bh_daq.get_most_massive_blackhole(
            base_path,
            snap_num,
            hid,
            fields=["BH_CumEgyInjection_RM"],
        )
        return bh_data["BH_CumEgyInjection_RM"]

    return _acquire_cluster_quantity(
        snap_num,
        get_black_hole_kinetic_energy,
        "BH kinetic energy injection",
        mass_field,
    )


def _get_bh_cumulative_thermal_energy(
    snap_num: int, mass_field: str = "Group_M_Crit200"
) -> NDArray:
    """
    Return the cumulative thermal energy ejected of most massive BH.

    :return: Array of cumulative thermal energy of most massive BH for
        every cluster.
    """

    # helper func
    def get_black_hole_thermal_energy(base_path: str, hid: int) -> float:
        """
        Return the black hole cumulative thermal energy.

        :param base_path: Sim base path.
        :param hid: Halo ID.
        :return: The kinetic energy injected over the lifetime of the BH.
        """
        # load all required data
        bh_data = bh_daq.get_most_massive_blackhole(
            base_path,
            snap_num,
            hid,
            fields=["BH_CumEgyInjection_QM"],
        )
        return bh_data["BH_CumEgyInjection_QM"]

    return _acquire_cluster_quantity(
        snap_num,
        get_black_hole_thermal_energy,
        "BH thermal energy injection",
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


def _get_bh_total_cumulative_mass(
    snap_num: int, mass_field: str = "Group_M_Crit200"
) -> NDArray:
    """
    Return the cumulative mass accreted of most massive BH.

    The total is the sum of the cumulative mass accreted
    in kinetic mode and the mass accreted in thermal mode.

    :return: Array of cumulative total mass of most massive BH for
        every cluster.
    """

    # helper func
    def get_black_hole_total_mass(base_path: str, hid: int) -> float:
        """
        Return the black hole cumulative total energy.

        :param base_path: Sim base path.
        :param hid: Halo ID.
        :return: The sum of the energy in thermal mode and the energy in
            kinetic mode.
        """
        # load all required data
        fields = ["BH_CumMassGrowth_QM", "BH_CumMassGrowth_RM"]
        bh_data = bh_daq.get_most_massive_blackhole(
            base_path,
            snap_num,
            hid,
            fields=fields,
        )
        return (
            bh_data["BH_CumMassGrowth_QM"] + bh_data["BH_CumMassGrowth_RM"]
        )

    return _acquire_cluster_quantity(
        snap_num,
        get_black_hole_total_mass,
        "BH total mass accreted",
        mass_field,
    )


def _get_bh_cumulative_kinetic_mass(
    snap_num: int, mass_field: str = "Group_M_Crit200"
) -> NDArray:
    """
    Return the cumulative mass accreted in kinetic mode.

    :return: Array of cumulative mass accreted in kinetic mode of most
        massive BH for every cluster.
    """

    # helper func
    def get_black_hole_kinetic_mass(base_path: str, hid: int) -> float:
        """
        Return the black hole cumulative mass accreted in kinetic mode.

        :param base_path: Sim base path.
        :param hid: Halo ID.
        :return: The mass accreted in kinetic mode.
        """
        # load all required data
        bh_data = bh_daq.get_most_massive_blackhole(
            base_path,
            snap_num,
            hid,
            fields=["BH_CumMassGrowth_RM"],
        )
        return bh_data["BH_CumMassGrowth_RM"]

    return _acquire_cluster_quantity(
        snap_num,
        get_black_hole_kinetic_mass,
        "BH kinetic mass accretion",
        mass_field,
    )


def _get_bh_cumulative_thermal_mass(
    snap_num: int, mass_field: str = "Group_M_Crit200"
) -> NDArray:
    """
    Return the cumulative mass accreted in thermal mode.

    :return: Array of cumulative mass accreted in thermal mode of most
        massive BH for every cluster.
    """

    # helper func
    def get_black_hole_thermal_mass(base_path: str, hid: int) -> float:
        """
        Return the black hole cumulative thermal mass accreted.

        :param base_path: Sim base path.
        :param hid: Halo ID.
        :return: The mass accreted in thermal mode.
        """
        # load all required data
        bh_data = bh_daq.get_most_massive_blackhole(
            base_path,
            snap_num,
            hid,
            fields=["BH_CumMassGrowth_QM"],
        )
        return bh_data["BH_CumMassGrowth_QM"]

    return _acquire_cluster_quantity(
        snap_num,
        get_black_hole_thermal_mass,
        "BH thermal mass accretion",
        mass_field,
    )


def _get_bh_number_of_progenitors(
    snap_num: int, mass_field: str = "Group_M_Crit200"
) -> NDArray:
    """
    Return the number of progenitors of the most massive BH per cluster.

    :return: Array of number of progenitors of most massive BH.
    """

    # helper func
    def get_black_hole_number_of_progenitors(
        base_path: str, hid: int
    ) -> float:
        """
        Return the black hole number of progenitors.

        :param base_path: Sim base path.
        :param hid: Halo ID.
        :return: The number of progenitor BHs.
        """
        # load all required data
        bh_data = bh_daq.get_most_massive_blackhole(
            base_path,
            snap_num,
            hid,
            fields=["BH_Progs"],
        )
        return bh_data["BH_Progs"]

    return _acquire_cluster_quantity(
        snap_num,
        get_black_hole_number_of_progenitors,
        "BH number of progenitors",
        mass_field,
    )


def _get_total_stellar_mass(
    snap_num: int, mass_field: str = "Group_M_Crit200"
) -> NDArray:
    """
    Return the total stellar mass per cluster in the FoF.

    :return: Array of shape (632, ) of stellar masses per cluster.
    """
    stellar_masses = np.zeros(N_CLUSTERS)

    # load and restrict TNG300-1 stellar masses
    halo_data = halos_daq.get_halo_properties(
        config.get_simulation_base_path("TNG300-1"),
        snap_num,
        ["GroupMassType", mass_field],
    )
    cluster_data = selection.select_clusters(
        halo_data, mass_field, expected_number=N_TNG300
    )
    stellar_masses[:N_TNG300] = cluster_data["GroupMassType"][:, 4]

    # load TNG-Cluster stellar masses
    halo_data = halos_daq.get_halo_properties(
        config.get_simulation_base_path("TNG-Cluster"),
        snap_num,
        ["GroupMassType"],
        cluster_restrict=True,
    )
    stellar_masses[N_TNG300:] = halo_data["GroupMassType"][:, 4]

    return stellar_masses


def _get_central_stellar_mass(
    snap_num: int, mass_field: str = "Group_M_Crit200"
) -> NDArray:
    """
    Return the stellar mass of the central galaxy.

    :return: Array of shape (632, ) of stellar masses of central subhalo
        per cluster.
    """
    stellar_masses = np.zeros(N_CLUSTERS)
    tng_300_base_path = config.get_simulation_base_path("TNG300-1")
    tng_cluster_base_path = config.get_simulation_base_path("TNG-Cluster")

    # load and restrict TNG300-1 primary subhalo IDs
    halo_data = halos_daq.get_halo_properties(
        tng_300_base_path,
        snap_num,
        ["GroupFirstSub", mass_field],
    )
    cluster_data = selection.select_clusters(
        halo_data, mass_field, expected_number=N_TNG300
    )

    # load the stellar mass from all subhalos
    for i, subhalo_id in enumerate(cluster_data["GroupFirstSub"]):
        fields = il.groupcat.loadSingle(
            tng_300_base_path, snap_num, subhaloID=subhalo_id
        )
        stellar_masses[i] = units.UnitConverter.convert(
            fields["SubhaloMassInRadType"][4], "SubhaloMassInRadType"
        )

    # load TNG-Cluster primary subhalos
    halo_data = halos_daq.get_halo_properties(
        tng_cluster_base_path,
        snap_num,
        ["GroupFirstSub"],
        cluster_restrict=True,
    )

    # load the stellar mass from all subhalos
    for i, subhalo_id in enumerate(halo_data["GroupFirstSub"]):
        fields = il.groupcat.loadSingle(
            tng_cluster_base_path, snap_num, subhaloID=subhalo_id
        )
        stellar_masses[i + N_TNG300] = units.UnitConverter.convert(
            fields["SubhaloMassInRadType"][4], "SubhaloMassInRadType"
        )

    return stellar_masses


def _get_relaxedness_dist(
    snap_num: int, mass_field: str = "Group_M_Crit200"
) -> NDArray:
    """
    Return the relaxedness of the clusters for TNG-Cluster only.

    :return: Array of relaxedness according to distance criterion.
    """
    relaxedness = np.zeros(N_CLUSTERS)

    # TNG300-1
    base_path = Path(config.get_simulation_base_path("TNG300-1"))
    path = (
        base_path.parent / "postprocessing" / "Relaxedness"
        / f"relaxedness_{snap_num}.hdf5"
    )
    with h5py.File(path, "r") as file:
        all_relaxedness = file["Halo"]["Distance_Criterion"][()]
    # index only desired entries
    halo_data = halos_daq.get_halo_properties(
        str(base_path),
        snap_num,
        [mass_field],
    )
    cluster_data = selection.select_clusters(halo_data, mass_field)
    relaxedness[:N_TNG300] = all_relaxedness[cluster_data["IDs"]]

    # TNG-Cluster
    path = (
        Path(config.get_simulation_base_path("TNG-Cluster")).parent
        / "postprocessing" / "released" / "Relaxedness.hdf5"
    )
    with h5py.File(path, "r") as file:
        relaxedness[N_TNG300:] = np.array(
            file["Halo"]["Distance_Criterion"][snap_num]
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

    # TNG300-1
    base_path = Path(config.get_simulation_base_path("TNG300-1"))
    path = (
        base_path.parent / "postprocessing" / "Relaxedness"
        / f"relaxedness_{snap_num}.hdf5"
    )
    with h5py.File(path, "r") as file:
        all_relaxedness = file["Halo"]["Mass_Criterion"][()]
    # index only desired entries
    halo_data = halos_daq.get_halo_properties(
        str(base_path),
        snap_num,
        [mass_field],
    )
    cluster_data = selection.select_clusters(halo_data, mass_field)
    relaxedness[:N_TNG300] = all_relaxedness[cluster_data["IDs"]]

    # TNG-Cluster
    path = (
        Path(config.get_simulation_base_path("TNG-Cluster")).parent
        / "postprocessing" / "released" / "Relaxedness.hdf5"
    )
    with h5py.File(path, "r") as file:
        relaxedness[N_TNG300:] = np.array(
            file["Halo"]["Mass_Criterion"][snap_num]
        )
    return relaxedness


def _get_formation_redshift(
    snap_num: int, mass_field: str = "Group_M_Crit200"
) -> NDArray:
    """
    Return array of formation redshifts.

    :return: Formation redshifts only for TNG-Cluster clusters.
    """
    formation_z = np.zeros(N_CLUSTERS)

    # TNG300-1
    formation_z[:N_TNG300] = np.nan  # no data yet

    # TNG-Cluster
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

    # TNG300-1
    path = (
        Path(config.get_simulation_base_path("TNG300-1")).parent
        / "postprocessing" / "released" / "CCcriteria.hdf5"
    )
    with h5py.File(path, "r") as file:
        ids = file["HaloIDs"]
        sorting_indices = np.argsort(ids)
        # need to slice the array of CCs as it is accidentally 352 entries long
        unsorted_ccts = file["centralCoolingTime"][:N_TNG300, snap_num]
        cct[:N_TNG300] = unsorted_ccts[sorting_indices]

    # TNG-Cluster
    path = (
        Path(config.get_simulation_base_path("TNG-Cluster")).parent
        / "postprocessing" / "released" / "CCcriteria.hdf5"
    )
    with h5py.File(path, "r") as file:
        ids = file["HaloIDs"]
        sorting_indices = np.argsort(ids)
        unsorted_ccts = np.array(file["centralCoolingTime"][:, snap_num])
        cct[N_TNG300:] = unsorted_ccts[sorting_indices]
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
