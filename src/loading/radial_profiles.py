"""
Data loading for radial temperature profiles.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


def load_radial_profile_data(
    filepath: str | Path,
    n_mass_bins: int | None = None,
    n_radial_bins: int | None = None,
    n_temperature_bins: int | None = None
) -> tuple[NDArray, NDArray] | None:
    """
    Return the radial profile data loaded from the given file.

    Function also additionally allows for verifying the shape of the
    data. If the number of either of the bins is given, that part of the
    data shape will be verified. If none of them are given, the data
    will be returned unverified.

    :param filepath: File name and path of the numpy data file.
    :param n_mass_bins: The number of mass bins. Used for verification.
        Optional.
    :param n_radial_bins: The number of radial bins in the data, used
        for verification. Optional.
    :param n_temperature_bins: The number of temperature bins in the
        data, used for verification. Optional.
    :return: A tuple of arrays, with the first one being the histogram
        data, and the second the running averages for the histograms. If
        loading or data verification fail, returns None instead.
    """
    logging.info("Loading saved radial temperature profiles from file.")
    if not isinstance(filepath, Path):
        filepath = Path(filepath)

    if not filepath.is_file():
        logging.error(f"The given file {str(filepath)} is not a valid file.")
        return

    # load the file
    with np.load(filepath) as data:
        histograms = data["hist_mean"]
        averages = data["running_avg"]

    if not n_radial_bins and not n_temperature_bins and not n_mass_bins:
        logging.info("Returning loaded data wihout verification.")
        return histograms, averages

    if n_mass_bins:
        if histograms.shape[0] != n_mass_bins:
            logging.error(
                f"Histogram data does not have the expected length: expected "
                f"{n_mass_bins} mass bins, but found {histograms.shape[0]} "
                f"instead."
            )
            return
        if averages.shape[0] != n_mass_bins:
            logging.error(
                f"Running averages do not have the expected length: expected "
                f"{n_mass_bins} mass bins, but found {averages.shape[0]} "
                f"instead."
            )
            return

    if n_radial_bins:
        if histograms.shape[2] != n_radial_bins:
            logging.error(
                f"Histogram data does not have the expected number of radial "
                f"bins: expected {n_radial_bins} mass bins, but found "
                f"{histograms.shape[2]} instead."
            )
            return
        if averages.shape[1] != n_radial_bins:
            logging.error(
                f"Running averages do not have the expected number of radial "
                f"bins: expected {n_radial_bins} mass bins, but found "
                f"{averages.shape[1]} instead."
            )
            return

    if n_temperature_bins:
        if histograms.shape[1] != n_temperature_bins:
            logging.error(
                f"Histogram data does not have the expected number of "
                f"temperature bins: expected {n_temperature_bins} mass bins, "
                f"but found {histograms.shape[1]} instead."
            )
            return

    logging.info("Successfully loaded verified data.")
    return histograms, averages
