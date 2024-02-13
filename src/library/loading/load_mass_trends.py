"""
Loading functions for mass trends plots.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


def load_mass_trend_data(
    filepath: Path | str, n_mass_bins: int | None = None
) -> tuple | None:
    """
    Load and verify the data from the mass trend plots.

    Data is expected to have three temperature regimes. Returned arrays
    for gas data have shape (H, 3) where H is the number of halos and
    the three values are the mean/median and the lower and upper error
    on the value respectively.

    :param filepath: Path to the data file to load.
    :param n_mass_bins: Number of mass bins, used for verification of
        data. Set to None for minimal verification.
    :return: The arrays of data, in the following order:

        - Gas fraction/masses for the three regimes for all halos (shape
          is (H, 2, 3))
        - Data array for cool gas fraction, shape (H, 3)
        - Data array for warm gas fraction, shape (H, 3)
        - Data array for hot gas fraction, shape (H, 3)
        - Data array for cool gas mass, shape (H, 3)
        - Data array for warm gas mass, shape (H, 3)
        - Data array for hot gas mass, shape (H, 3)
        - Array for binned halo masses, shape (H, 3)

        If the given file is invalid or data verification fails, the
        function returns None instead.
    """
    logging.info("Loading saved mass trend data from file.")

    if not isinstance(filepath, Path):
        filepath = Path(filepath)

    if not filepath.is_file():
        logging.error(f"The given file {str(filepath)} is not a valid file.")
        return

    # load the file
    with np.load(filepath) as data:
        gas_data_points = data["gas_data_points"]
        avg_masses = data["avg_masses"]
        cold_by_frac = data["cold_by_frac"]
        warm_by_frac = data["warm_by_frac"]
        hot_by_frac = data["hot_by_frac"]
        cold_by_mass = data["cold_by_mass"]
        warm_by_mass = data["warm_by_mass"]
        hot_by_mass = data["hot_by_mass"]

    if gas_data_points.shape[1] != 2 or gas_data_points.shape[2] != 3:
        logging.error(
            f"Halo data has unexpected shape: {gas_data_points.shape}."
        )
        return None

    to_verify = [
        cold_by_frac,
        warm_by_frac,
        hot_by_frac,
        cold_by_mass,
        warm_by_mass,
        hot_by_mass,
        avg_masses
    ]

    if n_mass_bins is None:
        logging.info("Loaded unverified mass trend data.")
        return (
            gas_data_points,
            avg_masses,
            cold_by_frac,
            warm_by_frac,
            hot_by_frac,
            cold_by_mass,
            warm_by_mass,
            hot_by_mass,
        )

    # in-depth verification of data shapes
    for dataset in to_verify:
        if dataset.shape != (3, n_mass_bins):
            logging.error(
                f"At least one data set has the wrong shape: got shape "
                f"{dataset.shape} but expected shape (3, {n_mass_bins})."
            )
            return None

    logging.info("Succesfully loaded and verified mass trend data.")
    return (
        gas_data_points,
        avg_masses,
        cold_by_frac,
        warm_by_frac,
        hot_by_frac,
        cold_by_mass,
        warm_by_mass,
        hot_by_mass,
    )


def load_cool_gas_mass_trend_data(
    filepath: str | Path, n_halos: int | None
) -> tuple[NDArray, NDArray, NDArray | None] | None:
    """
    Load and verify the data for cool gas fraction mass trends in clusters.

    :param filepath:
    :param n_halos:
    :return: Tuple containing the arrays for halo masses, cool gas
        fraction and color quantity respectively. If the colored
        quantity is not set, the last entry is None instead of an
        array.
    """
    logging.info("Loading saved cool gas fraction mass trend data from file.")

    if not isinstance(filepath, Path):
        filepath = Path(filepath)

    if not filepath.is_file():
        logging.error(f"The given file {str(filepath)} is not a valid file.")
        return

    # load the file
    with np.load(filepath) as data:
        halo_masses = data["halo_masses"]
        gas_fraction = data["cool_gas_fracs"]
        colored_quantity = data["color_quantity"]

    # check if the colored quantity is set
    if np.all(colored_quantity == 1):
        colored_quantity = None

    if n_halos is None:
        logging.info("Returning unverified cool gas mass trend data.")
        return halo_masses, gas_fraction, colored_quantity

    # verify data
    results = halo_masses, gas_fraction, colored_quantity
    for arr in results:
        if arr is None:
            continue  # skip empty colored quantity results
        if not arr.shape == (n_halos, ):
            logging.error(
                f"One of the loaded arrays has unexpected shape: {arr.shape}."
            )
            return

    return results
