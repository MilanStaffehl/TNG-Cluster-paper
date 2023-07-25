from __future__ import annotations

import numpy as np
import numpy.ma as ma
from typing import TYPE_CHECKING

import illustris_python as il

if TYPE_CHECKING:
    from numpy.typing import ArrayLike
    import config


def get_halos_binned_by_mass(
    bins: ArrayLike, 
    config: config.Config, 
) -> list[ArrayLike]:
    """
    Return an array of halo data, sorted by mass into bins.

    The function takes an array or Sequence of bin boundaries for the
    mass in units of solar masses. It then loads all halos from the 
    simulation set in the config given and sorts them into the bins by
    mass.

    The function returns a list of arrays with a length equal to the 
    number of mass bins. Every entry of the tuple is a numpy array of
    shape (N, 2), which holds the information on all halos in the
    respective mass bin. The entries of the array are the halo ID and 
    the mass of the halo in computational units.

    :param bins: bin edges, must have length of n + 1 where n is the
        number of bins desired and must be monotonic
    :param config: configuration object holding at least the base path
        for the current simulation, the snapshot number for the snapshot
        to use and the field name for the halo mass field preferred 
        (e.g. 'Group_M_Crit_200')
    :return: list of arrays, every array repreenting the halos of one
        mass bin with the arrays having shape (N, 2) and holding the 
        halo ID and halo mass in computational units.
    """
    n_bins = len(bins) - 1
    # load halos from simulation
    halo_masses = il.groupcat.loadHalos(
        config.base_path, config.snap_num, fields=config.mass_field
    )
    # create a list of indices for the halos
    halo_indices = np.indices([len(halo_masses)], sparse=True)[0]
    # sort halos into mass bins (returns a list of bin numbers)
    bin_indices = np.digitize(halo_masses, bins)

    final_list = []

    for b in range(n_bins):
        # mask values to only include halos in current bin
        mask = np.where(bin_indices == b + 1, 1, 0)
        masses = ma.masked_array(halo_masses).compress(mask)
        indices = ma.masked_array(halo_indices).compress(mask)

        return_array = np.array([indices, masses]).transpose()
        final_list.append(return_array)
    
    return final_list
