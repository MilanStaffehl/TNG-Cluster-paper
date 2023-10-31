"""
Function for data acquisition of gas cells.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Callable, Sequence

import illustris_python as il

from library import compute

if TYPE_CHECKING:
    from numpy.typing import NDArray


def get_halo_temperatures(
    halo_id: int,
    base_path: str,
    snap_num: int,
    additional_fields: list[str] | None = None,
    skip_condition: Callable[..., bool] = lambda x: False,
    skip_args: Sequence[Any] = None,
) -> dict[str, NDArray]:
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
    :param fields: A list of gas data fields to load in addition to the
        required fields. Leave empty if no further gas data is required.
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
