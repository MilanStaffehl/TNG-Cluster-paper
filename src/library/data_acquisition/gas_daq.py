"""
Function for data acquisition of gas cells.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Callable, Sequence

import illustris_python as il

from library import compute, units

if TYPE_CHECKING:
    from numpy.typing import NDArray


def get_halo_temperatures(
    halo_id: int,
    base_path: str,
    snap_num: int,
    additional_fields: list[str] | None = None,
    skip_condition: Callable[..., bool] = lambda x: False,
    skip_args: Sequence[Any] = None,
) -> dict[str, NDArray | int]:
    """
    Calculate temperatures for a single halo, return gas data.

    This method loads the gas cell data for a single halo and from
    it calculates the temperatures of the gas cells. It then adds the
    temperature array to the gas data dictionary and returns the dict.
    If the halo can be skipped or if it does not contain gas, the
    dictionary ``{"count": 0}`` is returned.

    Whether halos are skipped can be determined by the ``skip_condition``
    callable.

    The gas data dictionary will contain temperatures, internal energy,
    electron abundance, masses and star formation rate of every gas cell.
    To retrieve additional fields, the ``additional_fields`` argument
    can be used.

    .. attention:: All returned quantities are in computational units as
        described by the data spevification of the simulation.

    :param halo_id: The ID of the halo to process.
    :param base_path: Base path of the simulation.
    :param snap_num: Snapshot number at which to load the data.
    :param additional_fields: A list of gas data fields to load in
        addition to the required fields. Leave empty if no further gas
        data is required.
    :param skip_condition: A callable that can take a halo ID plus any
        number of additional positional arguments and returns as bool
        whether the halo of that ID may be skipped. Defaults to an
        expression that skips no halo.
    :param skip_args: List or arguments to pass to ``skip_condition``
        after the halo ID. Must be None if ``skip_condition`` takes
        only the halo ID as positional argument.
    :return: A dictionary with the gas data, including gas cell
        temperatures. If additional fields were specified, they are added
        to this dictionary. Dictionary will always contain key-value-pairs
        for the keys "Temperature", "InternalEnergy", "ElectronAbundance",
        "Masses" and "StarFormationRate". Note that all quantities are
        in computational units as specified by the simulation data specs.
        If the halo has no gas particles or is skipped, the return value
        is the dictionary ``{"count": 0}``.
    """
    if additional_fields is None:
        additional_fields = []
    if skip_args is None:
        skip_args = []

    # optionally skip a halo under specific conditions
    if skip_condition(halo_id, *skip_args):
        return {"count": 0}

    fields = [
        "InternalEnergy", "ElectronAbundance", "Masses", "StarFormationRate"
    ]
    gas_data = il.snapshot.loadHalo(
        base_path,
        snap_num,
        halo_id,
        partType=0,  # gas
        fields=fields + additional_fields,
    )

    # some halos do not contain gas
    if gas_data["count"] == 0:
        logging.debug(
            f"Halo {halo_id} contains no gas. Returning a fallback array."
        )
        return {"count": 0}

    # calculate temperatures
    temperatures = compute.get_temperature(
        gas_data["InternalEnergy"],
        gas_data["ElectronAbundance"],
        gas_data["StarFormationRate"],
    )

    gas_data["Temperature"] = temperatures
    return gas_data


def get_cluster_temperature(
    halo_id: int,
    base_path: str,
    snap_num: int,
) -> NDArray:
    """
    Get temperatures for a full TNG Cluster zoom.

    Loads the particle data of a full original TNG Cluster zoom and
    calculates the temperature for all particles. The array of these
    temperatures is then returned, without any of the accompanying gas
    data.

    .. attention:: This function is named similar to
        :func:`get_halo_temperatures`, but works fundamentally different:
        This function loads all particles from an original TNG Cluster
        zoom, including non-FoF particles, while the other only loads
        particles associated with the FoF-group.

    :param halo_id: The ID of the halo for which to find temperatures.
        This must be the actual ID, not the original halo ID.
    :param base_path: Base directory of the TNG-Cluster simulation data.
    :param snap_num: The snapshot from which to load the data.
    :return: Temperature of all particles in the original zoom-in region,
        in units of Kelvin.
    """
    fields = ["InternalEnergy", "ElectronAbundance", "StarFormationRate"]
    # acquire the necessary data from the zoom simulation
    gas_data = il.snapshot.loadOriginalZoom(
        base_path, snap_num, halo_id, partType=0, fields=fields
    )
    # calculate temperature
    return compute.get_temperature(
        gas_data["InternalEnergy"],
        gas_data["ElectronAbundance"],
        gas_data["StarFormationRate"],
    )


def get_gas_properties(
    base_path: str,
    snap_num: int,
    fields: Sequence[str],
    cluster: int | None = None,
) -> dict[str, NDArray]:
    """
    Load and return properties of all gas cells in the simulation.

    The function will convert units as far as they are known into physical
    units. The data is returned as a dictionary. The keys are identical
    to the field names given by ``fields`` and contain as values the
    loaded and unit-converted data for the gas particles.

    Note that all data existing in float64 format will be downcast to
    float32 type to save memory.

    :param base_path: The base path of the simulation to use.
    :param snap_num: The snapshot number from which to load the data.
    :param fields: The list of fields to load. Must match the name of
        the field in the simulation.
    :param cluster: When loading data from TNG-Cluster, set this to the
        ID of the halo of whose original zoom to load gas particles, to
        avoid loading filler particles and particles from other zooms.
        If not set, even when using TNG-Cluster, this function will load
        all particles of the simulation. Setting this to anything other
        than None for any simulation except TNG-Cluster will cause an
        error.
    :raises UnsupportedUnitError: If one of the fields has a unit that
        cannot be converted into physical units.
    :return: A dictionary of the field values for every gas cell,
        converted into physical units.
    """
    if not isinstance(fields, Sequence):
        fields = [fields]

    if cluster is None:
        logging.info(f"Loading gas particle properties: {', '.join(fields)}.")
    # verify units (done first to avoid loading time if conversion would fail)
    supported = units.UnitConverter.supported_fields()
    for field in fields:
        if field not in supported:
            raise units.UnsupportedUnitError(field)

    # load gas particle data
    if cluster is not None:
        gas_data = il.snapshot.loadOriginalZoom(
            base_path, snap_num, cluster, partType=0, fields=fields
        )
    else:
        gas_data = il.snapshot.loadSubset(
            base_path, snap_num, partType=0, fields=fields, float32=True
        )

    # turn arrays into dictionaries to comply with expected return type
    if not isinstance(gas_data, dict):
        gas_data = {fields[0]: gas_data}  # only one field exists

    # convert units
    gas_data_physical = {}
    for field, data in gas_data.items():
        gas_data_physical[field] = units.UnitConverter.convert(data, field)
    del gas_data  # memory clean-up
    if cluster is None:
        logging.info("Finished loading gas particle properties.")
    return gas_data_physical


def get_gas_temperatures(
    base_path: str,
    snap_num: int,
) -> NDArray:
    """
    Return the temperatures for all gas cells of a simulation.

    Function loads gas cell data required for temperature calculation
    (internal energy, electron abundance, SFR) for all gas particles in
    a simulation and calculates the temperature for all cells. Returns
    the array of temperature in units of Kelvin for all gas cells, in
    the order they were loaded in.

    :param base_path: Base path of the simulation.
    :param snap_num: The snapshot number from which to load data.
    :return: The array of gas cell temperatures in Kelvin. Ordered the
        same way as the gas data when loaded from a snapshot.
    """
    logging.info(
        "Loading gas cell data required for temperature calculation for "
        "all gas particles."
    )
    fields = ["InternalEnergy", "ElectronAbundance", "StarFormationRate"]
    gas_data = il.snapshot.loadSubset(
        base_path, snap_num, partType=0, fields=fields
    )

    # Step 5: Calculate temperature of every gas cell
    part_shape = gas_data["InternalEnergy"].shape
    logging.info(f"Calculating temperature for {part_shape[0]:,} gas cells.")
    temperatures = compute.get_temperature(
        gas_data["InternalEnergy"],
        gas_data["ElectronAbundance"],
        gas_data["StarFormationRate"],
    )
    # clean up unneeded data
    del gas_data
    return temperatures
