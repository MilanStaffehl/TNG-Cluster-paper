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
        M_200 mass field. Other mass fields or simulations probably
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
    mode: Literal["iterate", "searchsort", "intersect"] = "iterate",
    warn_if_not_unique: bool = False,
    warn_if_not_subset: bool = False,
    assume_unique: bool = False,
    assume_subset: bool = False,
) -> NDArray:
    """
    Return indices of entries in ``a`` that are also in ``s``.

    Function compares the entries of ``a`` with those of ``s`` and
    returns a list of array indices that point to those entries of ``a``
    that are also present in ``s``. Depending on the mode chosen, the
    result is a bit different. In modes ``iterate`` and ``intersect``,
    this function is the optimized equivalent to the Python code

    .. code:: python

        return [idx for idx, val in enumerate(a) if val in s]

    In mode ``searchsort`` however, the function also returns indices
    multiple times if they are present in ``s`` multiple times:

    >>> a = np.array([2, 3, 6, 8, 9, 1, 0])
    >>> s = np.array([6, 1, 1, 1])
    >>> select_if_in(a, s, mode="iterate")
    array([2, 5])
    >>> select_if_in(a, s, mode="searchsort")
    array([2, 5, 5, 5])

    The function can determine the list of indices using of three methods,
    which differ slightly in the output, performance and most importantly
    in the requirements on ``a`` and ``s`` they have. The method can be
    chosen with the ``mode`` parameter. It can have one of these values:

    - ``iterate``: This is the default option and the most robust one.
      It works for all cases, including those where entries to ``a`` are
      not unique and cases where ``s`` is not a subset of ``a``, i.e.
      where ``s`` includes values that are not in ``a``, but it is slower
      for cases where ``s`` is considerably smaller than ``a``. It only
      returns one index per unique value in ``s`` and the indices will
      be in sorted order. This also means however, that given an output
      ``output`` of this function, indexing ``a`` with it will **not**
      retain the order of ``s``, i.e. ``a[output] != s`` pair-wise.
    - ``intersect``: This mode is functionally equivalent to ``iterate``
      but can be faster when ``s`` is similar in size to ``a``. It has
      some additional constraints compared to ``intersect``: ``a`` must
      be unique, meaning no value can be in ``a`` more than once. It
      also explicitly requires both ``a`` and ``s`` to be 1D arrays.
      It returns the indices into ``a`` such that when indexing ``a``
      with them, the resulting array is sorted: ``a[output]`` is a
      sorted array when using mode ``intersect``.
    - ``searchsort``: This option has the benefit of returning an index
      multiple times if the corresponding value appears multiple times
      in ``s``, but it comes with a significant limitation: ``searchsort``
      only works if ``a`` has only unique entries! This function will
      not attempt to make ``a`` unique, so this mode can only be used
      if ``a`` is already unique. This mode returns the array of indices
      in such an order that each index is at the same position as the
      value in it points to is in ``s``, i.e.
      ``select_if_in(a, s, mode="searchsort")[i]`` is the index which
      points to the entry of ``a`` that has the value ``s[i]``. This
      also means that the order is retained when indexing ``a`` with the
      output of this function in mode ``searchsort``: ``a[output] == s``.
      Mode ``searchsort`` is typically much slower than the other modes
      due to additional computations required, but when ``assume_unique``
      is set to True, it can be faster for cases where ``s`` is much
      smaller than ``a``. However, in such a scenario, it loses its
      benefit over the other modes, since all three modes return the
      same result if ``s`` is unique. Therefore, it is recommended to
      use ``searchsort`` if and only if ``s`` is not unique _and_ one
      requires an index for every duplicate value in ``s``.

    Optionally, the function can log a warning if not all entries to
    ``a`` are unique, or if ``s`` is not  subset to ``a``, but this will
    add computational overhead, which might become noticeable for very
    large arrays ``a`` and/or ``s``.

    .. warning:

        When using mode ``searchsorted`` with ``a`` not being unique,
        i.e. with ``a`` containing the duplicate values, the function
        will pass execution without any error and will return an array
        of indices that might seem correct, but they will point to wrong
        entries in ``a``. This is a side effect of the method used.
        To prevent this, check whether ``a`` is unique before calling
        the function or use the ``warn_if_not_unique`` option.

    The function also allows the user to supply some a-priori assumption
    about the ``a`` and ``s``: If both ``a`` and ``s`` can be assumed to
    be unique, setting ``assume_unique`` to True is recommended as it can
    give considerable performance improvements in both ``iterate`` and
    ``intersect`` mode. If ``s`` can be assumed to be a subset of ``a``,
    i.e. all values in ``s`` are also in ``a``, setting ``assume_subset``
    to True is recommended when using mode ``searchsort``, which will
    shave off a lot of overhead and improve performance drastically for
    large arrays.

    .. caution::

        When setting ``assume_unique`` to True in modes ``iterate`` or
        ``intersect`` while either ``s`` or ``a`` are not actually
        unique, the function will pass execution silently and return a
        seemingly valid array of indices. These indices will however be
        wrong!

        Similarly, setting ``assume_subset`` to True in mode ``searchsort``
        while ``s`` is not in fact a subset of ``a``, the function will
        also pass with a wrong result.

        To be certain ``a`` and ``s`` are unique, you can use the option
        ``warn_if_not_unique`` which will log a warning if either is in
        fact not unique. Similarly, set ``warn_if_not_subset`` to True
        for an equivalent warning about ``s`` not being a subset of ``a``.
        Note that this option will **not** halt execution on a warning
        though!

    :param a: The array of values which is to be indexed.
    :param s: The array of values for which to search for in ``a``.
    :param mode: The mode of determining the indices. Options are
        ``iterate``, ``searchsort``, and ``intersect``. For details, see
        above. Defaults to ``iterate``.
    :param warn_if_not_unique: Whether to check ``a`` and ``s`` for
        uniqueness of all their entries. If set to True and ``a`` or
        ``s`` contain duplicate values, a warning is logged, but the
        method will continue to run. In such a case, the resulting
        return will be wrong if ``assume_unique`` is set to True, but
        correct otherwise. For large ``a`` setting this to True might
        add noticeable computational overhead. Defaults to False.
    :param warn_if_not_subset: Whether to check if ``s`` is a subset of
        ``a``. If set to True and there are entries is ``s`` that are
        not in ``a``, a warning is logged, but the method will continue
        to run. This is useful for cases where one expects the putput
        from all three modes to be identical, which is the case only if
        ``s`` . For large ``a`` and ``s`` this
        will add computational overhead that might be noticeable. Has no
        effect in mode ``detect``. Defaults to False.
    :param assume_unique: When using the ``iterate`` or ``intersect``
        modes, with both ``a`` and ``s`` assuredly being unique, this
        can be set to True to help speed up the calculation. However,
        if this is falsely set to True while either ``a`` or ``s`` are
        in fact not unique, the results returned by this function will
        be wrong.
    :param assume_subset: When using mode ``searchsort`` with ``s``
        assuredly being a subset of ``a``, this can be set to True to
        help speed up the calculation _considerably_. However, if this
        is falsely set to True while ``s`` does in fact contain values
        that are not present in ``a``, the results returned by this
        function will be wrong.
    :return: A list of indices into ``a`` that select all those values
        that are also in ``s``. When using mode ``iterate``, this will
        also correctly point to multiple occurrences of the same value
        in ``a``. When using mode ``searchsort``, this will also include
        an index multiple times if the corresponding value was included
        in ``s`` multiple times. If an error occurs (for example due to
        a wrong mode), the function returns an array of shape (1, )
        containing only one NaN value.
    """
    # If warnings were enabled, check if something needs to be logged.
    # Since the same checks are performed in mode `detect`, we skip them
    # here as we will need to repeat them later anyhow and mode `detect`
    # issues no warnings.
    if warn_if_not_unique:
        if len(np.unique(a.copy())) != len(a):
            logging.warning("`select_if_in`: `a` contains duplicate entries!")

    if warn_if_not_subset:
        if len(np.intersect1d(a, s)) < len(np.unique(s)):
            logging.warning("`select_if_in`: `s` is not a subset of `a`!")

    # find indices
    if mode == "searchsort":
        # make use of a-priori assumptions:
        if assume_subset:
            a_sorted_indices = np.argsort(a)
            indices = np.searchsorted(a[a_sorted_indices], s)
            return a_sorted_indices[indices]
        else:
            # `s` is not a subset of `a`, must mask invalid indices
            a_sorted_indices = np.argsort(a)
            a_sorted = a[a_sorted_indices]
            sorted_indices = np.searchsorted(a_sorted, s)
            selected_indices = np.take(
                a_sorted_indices, sorted_indices, mode="clip"
            )
            return selected_indices[a[selected_indices] == s]
    elif mode == "iterate":
        return np.nonzero(np.isin(a, s, assume_unique=assume_unique))[0]
    elif mode == "intersect":
        return np.intersect1d(
            a, s, assume_unique=assume_unique, return_indices=True
        )[1]
    else:
        logging.error(f"Unsupported mode {mode} for `selection.select_if_in`.")
        return np.array([np.nan])
