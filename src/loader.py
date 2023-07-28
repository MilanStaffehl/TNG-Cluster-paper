from __future__ import annotations

from typing import TYPE_CHECKING

import illustris_python as il
import numpy as np
import numpy.ma as ma

import constants

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

    import config


def get_halos_binned_by_mass(
    bins: ArrayLike, 
    config: config.Config, 
) -> tuple[list[ArrayLike]]:
    """
    Return an array of halo data, sorted by mass into bins.

    The function takes an array or Sequence of bin boundaries for the
    mass in units of solar masses. It then loads all halos from the 
    simulation set in the config given and sorts them into the bins by
    mass.

    The function returns a tuple of two lists, each containing in turn
    as many arrays as there are mass bins. The first list contains
    arrays of halo IDs, the second contains arrays of the corresponding
    masses. Each list entry corresponds to one bin, i.e. the first list
    contains entries for the first bin, the second list for the second
    bin etc.

    :param bins: bin edges, must have length of n + 1 where n is the
        number of bins desired and must be monotonic
    :param config: configuration object holding at least the base path
        for the current simulation, the snapshot number for the snapshot
        to use and the field name for the halo mass field preferred 
        (e.g. 'Group_M_Crit_200')
    :return: tuple of two lists, each containing arrays. The first list
        holds arrays which hold the halo IDs, the second list holds
        arrays which hold the corresponding halo masses.
    """
    n_bins = len(bins) - 1
    # load halos from simulation
    halo_masses = il.groupcat.loadHalos(
        config.base_path, config.snap_num, fields=config.mass_field
    )
    # convert to physical unit solar masses
    halo_masses = halo_masses * 1e10 / constants.HUBBLE

    # create a list of indices for the halos
    halo_indices = np.indices([len(halo_masses)], sparse=True)[0]

    # sort halos into mass bins (returns a list of bin numbers)
    mask_indices = np.digitize(halo_masses, bins)

    index_list = []
    mass_list = []

    for b in range(n_bins):
        # mask values to only include halos in current bin
        mask = np.where(mask_indices == b + 1, 1, 0)
        masses = ma.masked_array(halo_masses, dtype=float).compress(mask)
        indices = ma.masked_array(halo_indices, dtype=int).compress(mask)

        mass_list.append(masses)
        index_list.append(indices)
    
    return index_list, mass_list


def generate_halos_binned_by_mass(
    bins: ArrayLike, 
    config: config.Config, 
) -> tuple[list[ArrayLike]]:
    """
    Yield an array of halo data, sorted by mass into bins.

    The function takes an array or Sequence of bin boundaries for the
    mass in units of solar masses. It then loads all halos from the 
    simulation set in the config given and sorts them into the bins by
    mass.

    The function yields for every bin a tuple of two arrays, containing
    the halo data for the current bin: the first contains the halo ID
    for all halos in the current bin, the second their respective masses.

    :param bins: bin edges, must have length of n + 1 where n is the
        number of bins desired and must be monotonic
    :param config: configuration object holding at least the base path
        for the current simulation, the snapshot number for the snapshot
        to use and the field name for the halo mass field preferred 
        (e.g. 'Group_M_Crit_200')
    :yield: tuple of arrays, first holding halo IDs, second holding
        halo masses in units of solar masses 
    """
    n_bins = len(bins) - 1
    # load halos from simulation
    halo_masses = il.groupcat.loadHalos(
        config.base_path, config.snap_num, fields=config.mass_field
    )
    # convert to physical unit solar masses
    halo_masses = halo_masses * 1e10 / constants.HUBBLE

    # create a list of indices for the halos
    halo_indices = np.indices([len(halo_masses)], sparse=True)[0]

    # sort halos into mass bins (returns a list of bin numbers)
    mask_indices = np.digitize(halo_masses, bins)

    for b in range(n_bins):
        # mask values to only include halos in current bin
        mask = np.where(mask_indices == b + 1, 1, 0)
        masses = ma.masked_array(halo_masses).compress(mask)
        indices = ma.masked_array(halo_indices).compress(mask)

        yield indices, masses