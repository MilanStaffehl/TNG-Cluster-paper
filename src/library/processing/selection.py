"""
Tools to select data entries from a larger data set.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Iterator, Literal

import numpy as np
import numpy.ma as ma

if TYPE_CHECKING:
    from numpy.typing import NDArray


def digitize_clusters(
    cluster_masses: NDArray,
    bins: NDArray | None = None,
) -> NDArray:
    """
    ``numpy.digitize`` for clusters in TNG300-1 and TNG-Cluster.

    Wrapper around ``numpy.digitize`` made to accommodate for the fact
    that some clusters in TNG-Cluster lie just slightly above the highest
    mass bin of 10^15 solar masses and therefore would not be considered
    in some analyses.

    This function simply calls ``numpy.digitize`` with the given masses
    and bins and returns the resulting array of indices, with one change:
    all indices higher than ``len(bins) - 1``, i.e. all indices belonging
    to masses outside the right edge of the mass bins are set to the
    index of the highest mass bin. This is so that clusters with a mass
    of for example log M = 15.4001 are also considered in analysis and
    not discarded.

    .. attention:: This function is specifically meant for a very
        narrow purpose, namely to replace the use of ``numpy.digitize``
        for TNG300-1 and TNG-Cluster clusters, when used with the
        M_200c mass field. Other mass fields or simulations probably
        won't do anything different from ``numpy.digitize`` or they
        might cause seriously misleading results. Use with caution.

    :param cluster_masses: Array of the cluster masses in units of solar
        masses.
    :param bins: Array of mass bin edges in units of solar masses.
        Optional, defaults to seven 0.2 dex mass bins from log M = 14 to
        log M = 15.4 when left empty or set to None.
    :return: An array of bin indices into which the masses fall, with
        the masses that fall to the right of the last bin edge being
        sorted into the last bin instead.
    """
    if bins is None:
        bins = 10**np.linspace(14.0, 15.4, num=8)
    mask = np.digitize(cluster_masses, bins)
    # replace index of "outside of bins" with last bin index
    mask[mask == len(bins)] -= 1
    return mask


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


def bin_quantity(
    quantity: NDArray,
    bin_mask: NDArray,
    n_bins: int = -1,
) -> Iterator[NDArray]:
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
        to keep unmasked. Defaults to 0.
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


def mask_data_dict(
    data: dict[str, NDArray],
    mask: NDArray,
    index: int = 0,
) -> dict[str, NDArray]:
    """
    Mask contents of a data dictionary.

    Function masks every entry of a data dictionary as used by the
    illustris_python helper scripts given a mask and an index from that
    mask. Only entries matching the selected index in the mask will be
    kept. The function also updates the ``count`` of the dictionary
    appropriately.

    :param data: Data dictionary; a mapping of field names as string to
        values as NDArrays.
    :param mask: The mask to use. Must be an NDArray of integers of the
        same length as the first axis of the data arrays in the dictionary.
    :param index: The index whose entries to select. See
        :func:`mask_quantity` for details.
    :return: The dictionary but with its data entries restricted to the
        selected non-masked values and with an updated ``count`` value.
    """
    restricted_data = {}
    for field, value in data.items():
        if field == "count":
            continue
        restricted_data[field] = mask_quantity(
            value, mask, index, compress=True
        )

    restricted_data["count"] = len(list(restricted_data.values())[0])
    return restricted_data


def select_clusters(
    halo_data: dict[str, NDArray],
    mass_field: str,
    mass_threshold: float = 1e14,
    expected_number: int | None = None,
) -> dict[str, NDArray]:
    """
    Return the halo data, restricted to halos above mass threshold.

    The function creates a new dictionary with the same fields, but with
    values only for those halos that are above the mass threshold. The
    halo data dictionary must contain a mass field (as specified by
    ``mass_field``) which will be used to determine which halos lie
    above the threshold.

    :param halo_data: Dictionary of halo data, consisting of field names
        as keys and arrays of corresponding values as dict values.
    :param mass_field: The name of the mass field to use for the
        restriction. Must be one of the keys in the halo data dictionary.
    :param mass_threshold: The value below which to clip the data in
        units of the given mass field.
    :param expected_number: The expected number of clusters after
        restriction. When given, the function will raise an exception
        if the restricted data does not have the expected number of
        entries.
    :return: The halo data dictionary, but with values only for those
        halos above the mass threshold.
    """
    logging.info(
        f"Restricting halo data to halos with mass log M > "
        f"{np.log10(mass_threshold)}"
    )
    mask = np.digitize(halo_data[mass_field], [0, mass_threshold, 1e25])
    restricted_data = {}
    for key, value in halo_data.items():
        if key == "count":
            continue
        restricted_data[key] = mask_quantity(
            halo_data[key], mask, index=2, compress=True
        )
    # update count
    restricted_data["count"] = len(restricted_data[mass_field])
    if expected_number is not None:
        if restricted_data["count"] != expected_number:
            raise RuntimeError(
                f"Selected an unexpected number of clusters: expected "
                f"{expected_number} clusters but got "
                f"{restricted_data['count']} instead."
            )
    return restricted_data


def select_if_in(
    a: NDArray,
    s: NDArray,
    mode: Literal["iterate", "searchsorted", "detect"] = "iterate",
    warn_if_not_unique: bool = False,
    warn_if_not_subset: bool = False,
) -> NDArray:
    """
    Return indices of entries in ``a`` that are also in ``s``.

    Function compares the entries of ``a`` with those of ``s`` and
    returns a list of array indices that point to those entries of ``a``
    that are also present in ``s``. This is equivalent to the Python
    code

    .. code:: python

        return [idx for idx, val in enumerate(a) if val in s]

    The function can determine the list of indices one of two ways and
    does so depending on the ``mode`` of operation chosen:

    - ``iterate``: This is the default option and the most robust one.
      It works for all cases, including those where entries to ``a`` are
      not unique and cases where ``s`` is not a subset of ``a``, i.e.
      where ``s`` includes values that are not in ``a``, but it is slower
      for cases where ``s`` is considerably smaller than ``a``, and it
      returns the array of indices sorted.
    - ``searchsorted``: This option is typically faster for cases where
      ``s`` is considerably smaller than ``a``, but it comes with two
      significant limitations: ``searchsorted`` only works if ``a`` has
      only unique entries _and_ only if ``s`` is a subset of ``a``. This
      function will not attempt to limit ``s`` to a subset of ``a``, so
      if this is desired, it must be done ahead of calling this function.
      This mode returns the array of indices in such an order that each
      index points to the corresponding value in ``s``, i.e.
      ``select_if_in(s, a, mode="searchsorted")[i]`` is the index which
      points to the entry of ``a`` that has the value ``s[i]``. Unless
      you know what you are doing, avoid this mode.
    - ``detect``: This option will detect whether ``s`` is a subset of
      ``a`` and whether ``a`` contains only unique entries and chooses
      the appropriate method from the two choices above accordingly.
      While convenient, this is discouraged as it requires additional
      intensive computations that will add noticeable overhead when ``a``
      and/or ``s`` are very large. If you cannot be sure ``a`` and ``s``
      will fulfill the requirements for mode ``searchsorted`` however,
      it is recommended to set the mode to ``detect``.

    .. warning:

        When using mode ``searchsorted`` with ``s`` not being a subset
        of ``a``, the function will pass execution without any error and
        will return an array of indices that might seem correct, but it
        will point to wrong entries in ``a`` due to the entries in ``s``
        that are not in ``a``. This is a side effect of the method used.
        To prevent this, check whether ``s`` is a subset of ``a`` before
        calling the function or use mode ``detect`` instead.

        Similarly, using ``searchsorted`` when ``a`` is not unique will
        not cause an error either and will produce an array of indices
        that point to valid entries, but it will miss all duplicate
        entries in ``a`` except the first in sorted order.

        Both problems can be avoided by using mode ``detect``, at the
        cost of computational overhead.

    Optionally, the function can log a warning if not all entries to
    ``a`` are unique, or if ``s`` is not  subset to ``a``, but this will
    add computational overhead, which might become noticeable for very
    large arrays ``a`` and/or ``s``.

    Note that this function assumes all entries in ``s`` to be unique.
    If they are not, mode ``searchsorted`` will return a list of indices
    that also contains duplicates. Avoid this by either making ``s``
    unique, or by making the list of indices unique after running this
    function. In mode ``iterate``, the indices into ``a`` will not
    contain duplicates.

    :param a: The array of values which is to be indexed.
    :param s: The array of values for which to search for in ``a``.
    :param mode: The mode of determining the indices. Options are
        ``iterate``, ``searchsorted``, and ``detect``. For details, see
        above. Defaults to ``iterate``.
    :param warn_if_not_unique: Whether to check ``a`` for uniqueness of
        all its entries. If set to True and ``a`` contains duplicate
        values, a warning is logged, but the method will continue to run.
        The resulting return will be wrong in mode ``searchsorted`` but
        correct in both other modes. For large ``a`` setting this to
        True might add noticeable computational overhead. Has no effect
        in mode ``detect``. Defaults to False.
    :param warn_if_not_subset: Whether to check if ``s`` is a subset of
        ``a``. If set to True and there are entries is ``s`` that are
        not in ``a``, a warning is logged, but the method will continue
        to run. The resulting return will be wrong in mode ``searchsorted``
        but correct in both other modes. For large ``a`` and ``s`` this
        will add computational overhead that might be noticeable. Has no
        effect in mode ``detect``. Defaults to False.
    :return: A list of indices into ``a`` that select all those values
        that are also in ``s``. When using ``mode=iterate``, this will
        also correctly point to multiple occurrences of the same value
        in ``a``. If an error occurs (for example due to a wrong mode),
        the function returns an array of shape (1, ) containing only
        one NaN value.
    """
    # If warnings were enabled, check if something needs to be logged.
    # Since the same checks are performed in mode `detect`, we skip them
    # here since we will need to repeat them later anyhow and mode
    # `detect` issues no warnings.
    if warn_if_not_unique and mode != "detect":
        if len(np.unique(a.copy())) != len(a):
            logging.warning("`select_if_in`: `a` contains duplicate entries!")

    if warn_if_not_subset and mode != "detect":
        if len(np.intersect1d(a, s)) < len(s):
            logging.warning("`select_if_in`: `s` is not a subset of `a`!")

    # if mode is `detect`, find out which one to use
    if mode == "detect":
        # check for uniqueness
        if len(np.unique(a.copy())) != len(a):
            logging.debug(
                "`select_if_in`: mode `detect` found duplicate entries in "
                "array `a`. Set mode to `iterate`."
            )
            mode = "iterate"
        # check for subset
        elif len(np.intersect1d(a, s)) < len(s):
            logging.debug(
                "`select_if_in`: mode `detect` found `s` to not be a subset "
                "of `a`. Set mode to `iterate`."
            )
            mode = "iterate"
        else:
            logging.debug(
                "`select_if_in1: mode `detect` found `a` has only unique "
                "entries; `s` is a subset of `a`. Set mode to `searchsorted`."
            )
            mode = "searchsorted"

    # find indices
    if mode == "searchsorted":
        a_sorted_indices = np.argsort(a)
        indices = np.searchsorted(a[a_sorted_indices], s)
        return a_sorted_indices[indices]
    elif mode == "iterate":
        return np.nonzero(np.isin(a, s))[0]
    else:
        logging.error(f"Unsupported mode {mode} for `selection.select_if_in`.")
        return np.array([np.nan])
