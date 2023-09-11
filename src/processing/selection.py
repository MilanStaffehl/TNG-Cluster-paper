"""
Tools to select data entries from a larger data set.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import numpy.ma as ma

if TYPE_CHECKING:
    from numpy.typing import NDArray


def select_halos_from_mass_bins(
    selections_per_bin: int,
    halo_ids: NDArray,
    n_mass_bins: int,
    mass_bin_mask: NDArray
) -> NDArray:
    """
    Return a list of halo IDs selected from every mass bin.

    The method returns N = ``selections_per_bin`` halos from every mass
    bin, selected randomly, in an array.

    :param selections_per_bin: The number of halos to select in every
        mass bin.
    :param halo_ids: An array of all available halo IDs.
    :param n_mass_bins: The number of available mass bins.
    :param mass_bin_mask: An array containing the mass bin number of
        every halo in ``halo_ids``, i.e. the number of the mass bin into
        which the halo with the corresponding array index falls.
    :return: A flattened array of all halos IDs chosen. They are ordered
        by mass bin, meaning that the first N [*]_ IDs are selected from
        the first mass bin, the next N from the second and so on. This
        means the array has length N * ``n_mass_bins``.

    .. [*] N = ``selections_per_mass_bin``
    """
    logging.info("Selecting subset of halos for gallery.")
    selected_halo_ids = np.zeros(n_mass_bins * selections_per_bin, dtype=int)
    for bin_num in range(n_mass_bins):
        mask = np.where(mass_bin_mask == bin_num + 1, 1, 0)
        masked_indices = ma.masked_array(halo_ids).compress(mask)
        masked_indices = masked_indices.compressed()
        # choose entries randomly
        rng = np.random.default_rng()
        low_edge = bin_num * selections_per_bin
        upp_edge = (bin_num + 1) * selections_per_bin
        selected_halo_ids[low_edge:upp_edge] = rng.choice(
            masked_indices, size=selections_per_bin, replace=False
        )
    return selected_halo_ids
