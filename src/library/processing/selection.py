"""
Tools to select data entries from a larger data set.
"""
from __future__ import annotations

import logging
import warnings
from typing import TYPE_CHECKING, Iterator, Sequence

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


def sort_masses_into_bins(
    masses: NDArray, mass_bins: Sequence[float]
) -> NDArray:
    """
    Sort the given masses into the bins and return mask index array.

    The returned array is of the same length as ``masses`` and contains
    the index of the bin into which the mass of the same array index
    falls, starting from 1.

    :param masses: Array of masses.
    :param mass_bins: List of bin edges, must have same unit as ``masses``.
    :return: Array of mask indices for masses.
    """
    warnings.warn(
        "Called function 'sort_masses_into_bins', which is merely an alias "
        "for 'numpy.digitize'. Use the latter instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return np.digitize(masses, mass_bins)


def bin_quantity(quantity: NDArray,
                 bin_mask: NDArray,
                 n_bins: int = -1) -> Iterator[NDArray]:
    """
    Sort ``quantity`` into mass bins according to ``bin_mask``.

    Function is a generator that will yield, for every mass bin given
    by ``bin_mask``, all entries in ``quantity`` that fall into the
    current bin. It will start by yielding the first bin, given by the
    index 1 in the bin mask, and continue until the last bin present
    in the bin mask.

    :param quantity: The array of quantities to bin. Must have shape
        (N, S) where S can be any arbitrary shape. The array will be
        binned along the first axis.
    :param bin_mask: A mask assigning every entry in ``quantity`` to a
        bin. Must be an array of shape (N, ). Every entry must be a
        number, assigning the corresponding entry  of the same array
        index in ``quantity`` to a bin. Can be obtained for example
        from ``numpy.digitize``
    :param n_bins: The number of bins to sort into. All bins with mask
        indices higher than this will be ignored. This is useful when
        using a mask returned by ``np.digitize`` to avoid values outside
        of the value range (which are assigned a value of either zero or
        ``len(bins)``) being sorted as well but instead discarded.
        Optional, defaults to -1 which means the number of bins will be
        determined from the mask by taking the highest index in it as
        the number of bins.
    :return: A generator object that will yield arrays of shape (M, S).
        M is the number of entries inside the n-th bin. The generator
        will go through the bins in order, starting from bin index 1.
    """
    if n_bins == -1:
        n_bins = np.max(bin_mask)
    for bin_num in range(n_bins):
        mask = np.where(bin_mask == bin_num + 1, 1, 0)
        masked_indices = ma.masked_array(quantity).compress(mask)
        masked_indices = masked_indices.compressed()
        yield masked_indices


def mask_quantity(
    quantity: NDArray,
    mask: NDArray,
    index: int = 0,
    compress: bool = True
) -> NDArray:
    """
    Return an array containing only non-masked values.

    All entries in ``quantity`` that do not have a masking value equal
    to ``index`` in the corresponding position in ``mask`` are masked
    by this function. The returned array can be either a numpy masked
    array or a compressed version of the masked array, only containing
    the unmasked values and nothing else.

    The default behavior is to expect the mask to contain only zeros and
    ones, with all values in the quantity array at the positions of ones
    being masked and all values at positions of zeros remaining unmasked.
    The function returns a compressed version of the masked array by
    default. This can be changed by setting ``compressed=False``.

    A custom value for which integer in the masking array which leaves
    corresponding values unmasked can be chosen.

    Note that multidimensional ``quantity`` arrays will be masked along
    the first axis (axis 0), such that any vectors inside the array will
    retain their shape.

    :param quantity: Array of the quantity to mask.
    :param mask: Masking array. Must be an array of integers. Must have
        the same shape as ``quantity``.
    :param index: The integers value for the index in the masking array
        to keep unmasked. Defaults to 1.
    :param compress: Whether to compress the masked array into a standard
        numpy array before returning. Defaults to True.
    :return: The masked quantity array.
    """
    mask = np.where(mask == index, 1, 0)
    masked_indices = ma.masked_array(quantity).compress(mask, axis=0)
    if not compress:
        return masked_indices
    masked_indices = masked_indices.compressed()
    if len(quantity.shape) > 1:
        masked_indices = masked_indices.reshape(-1, *quantity.shape[1:])
    return masked_indices
