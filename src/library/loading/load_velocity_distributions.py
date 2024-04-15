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


def load_velocity_distributions(
    filepath: str | Path,
    n_velocity_bins: int | None = None,
    n_clusters: int | None = 352,
) -> tuple[NDArray, NDArray, NDArray, NDArray, NDArray] | None:
    """
    Return the velocity distribution histograms for all clusters.

    Loads the velocity distribution histograms of all clusters in TNG300
    and TNG-Cluster and returns them alongside the cluster masses, the
    mass bin mask to sort them into mass bins of 0.2 dex and the edges
    of the histogram bins in the following order:

    - Histograms in shape (``n_clusters``, ``n_velocity_bins``)
    - Bin edges (length ``n_velocity_bins + 1``)
    - Cluster masses (length ``n_clusters``)
    - Cluster mass bin mask (length ``n_clusters``)

    Optionally, if the number of velocity bins and/or clusters is given,
    the shape of the returned data is verified. If verification fails,
    function returns None.

    :param filepath: The path and filename of the file with the data.
    :param n_velocity_bins: The number of velocity bins in the histogram.
        Optional. If given, data will be verified to have this number of
        bins.
    :param n_clusters: The number of clusters. Optional. If given, the
        data will be verified to have this number of clusters.
    :return: The histograms, bin edges, cluster masses and mass bin mask
        (see above).
    """
    logging.info("Loading velocity distribution of clusters from file.")
    if not isinstance(filepath, Path):
        filepath = Path(filepath)

    if not filepath.is_file() or not filepath.exists():
        logging.error(f"The given file {filepath} is not a valid file.")
        return

    # load file
    with np.load(filepath) as data_file:
        histograms = data_file["histograms"]
        edges = data_file["edges"]
        halo_masses = data_file["halo_masses"]
        virial_velocities = data_file["virial_velocities"]
        mass_mask = data_file["mass_mask"]

    if n_clusters is None and n_velocity_bins is None:
        logging.info("Returning loaded data without verification.")
        return histograms, edges, halo_masses, virial_velocities, mass_mask

    # verify data shapes
    if n_clusters is not None:
        if histograms.shape[0] != n_clusters:
            logging.error(
                f"The loaded data does not have the expected number of "
                f"clusters: expected {n_clusters} but loaded "
                f"{histograms.shape[0]}."
            )
            return
        if len(halo_masses) != n_clusters:
            logging.error(
                f"Halo masses array has wrong number of entries: "
                f"Halo masses has length {len(halo_masses)}; expected "
                f"{n_clusters} entries."
            )
            return
        if len(mass_mask) != n_clusters:
            logging.error(
                f"Mass mask array has wrong number of entries: "
                f"Mass mask has length {len(halo_masses)}; expected "
                f"{n_clusters} entries."
            )
            return
        if len(virial_velocities) != n_clusters:
            logging.error(
                f"Virial velocity array has wrong number of entries: "
                f"Virial velocities has length {len(halo_masses)}; expected "
                f"{n_clusters} entries."
            )
            return
    if n_velocity_bins is not None:
        if histograms.shape[1] != n_velocity_bins:
            logging.error(
                f"Histograms have the wrong number of bins: expected "
                f"{n_velocity_bins} but loaded histograms with "
                f"{histograms.shape[1]} bins."
            )
            return
        if len(edges) != n_velocity_bins + 1:
            logging.error(
                f"The array of edges has the wrong number of entries. "
                f"Expected {n_velocity_bins + 1} but loaded {len(edges)} "
                f"entries."
            )
            return

    logging.info(
        "Successfully loaded and verified velocity distribution data."
    )
    return histograms, edges, halo_masses, virial_velocities, mass_mask
