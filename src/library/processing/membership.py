"""
Functions to determine membership of particles and subhalos.
"""
from __future__ import annotations

from pathlib import Path
from typing import TypeAlias

import h5py
import illustris_python as il
import numpy as np
from numpy.typing import NDArray


def particle_parents(
    particle_indices: NDArray,
    particle_types: NDArray,
    snap_num: int,
    base_path: str | Path,
) -> tuple[NDArray, NDArray]:
    """
    Return the parent halo and subhalo ID of every particle given.

    Function takes either a single integer or an array of integers and
    returns a tuple of either two integers (when given a single integer)
    or two arrays of integers of the same length as the input.

    The first returned integer or array of integers is the parent halo
    of every particle passed. The second one is the parent subhalo of
    every particle passed.

    If a particle is unbound, i.e. it does not belong to a (sub)halo,
    the corresponding parent ID is set to -1.

    :param particle_indices: indices of a single particle or array of
        multiple particle IDs.
    :param particle_types: Array of particle type flag. Must be 0, 4, or
        5 for every particle. Must have same length as ``particle_ids``.
    :param snap_num: The snapshot at which to look for the particle
        parent halo and subhalo.
    :param base_path: Base path of the simulation.
    :return: Tuple of halo and subhalo ID. For a single particle, this
        is a tuple of integers, for an array of particle IDs this is a
        tuple of arrays of the same length as the input.
    """
    if not isinstance(base_path, Path):
        base_path = Path(base_path)

    # load offsets and group lens
    offsets_and_lens = load_offsets_and_lens(base_path, snap_num)
    group_offsets, group_lens = offsets_and_lens[:2]
    subhalo_offsets, subhalo_lens = offsets_and_lens[2:]

    # allocate memory for the final results
    parent_halos = np.empty_like(particle_indices, dtype=np.int64)
    parent_halos[:] = -1  # initialize to -1
    parent_subhalos = np.empty_like(parent_halos, dtype=np.int64)
    parent_subhalos[:] = -1  # initialize to -1

    # assign parents by type
    for part_type in [0, 4, 5]:
        # find parents for current type
        current_parent_halos = find_parent(
            particle_indices[particle_types == part_type],
            group_offsets[:, part_type],
            group_lens[:, part_type],
        )
        parent_halos[particle_types == part_type] = current_parent_halos
        # same for subhalos
        current_parent_subhalos = find_parent(
            particle_indices[particle_types == part_type],
            subhalo_offsets[:, part_type],
            subhalo_lens[:, part_type],
        )
        parent_subhalos[particle_types == part_type] = current_parent_subhalos

    return parent_halos, parent_subhalos


def find_parent(
    particle_ids: NDArray, offsets: NDArray, lengths: NDArray
) -> NDArray:
    """
    Given a set of particle IDs, offsets and lengths, find parents.

    Function takes an array of particle IDs and an array of offsets (i.e.
    the particle ID of the first particle of every possible parent group
    or subgroup) as well as an array of the number of particles in every
    group/subgroup (must be of same length as the array of offsets). It
    then assigns to every particle the index/ID of the parent group (or
    subgroup) and returns it.

    Particles that have no parent will be assigned -1 instead.

    The functions works by using searchsort to sort particles into the
    array of last particle IDs (effectively the sum of offsets and
    lengths). This gives for every particle ID the index of the
    corresponding halo. To account for unbound fuzz particles, the
    function then also finds the position into which the particles fall
    with respect to the first particle of every group. These positions
    should be shifted one position to the right. If they are however
    shifted by more than 1, the particle must be unbound.

    .. admonition:: Example

        Assume a set of 10 particles. The group starts at ID 0, and
        the fuzz starts at ID 7. The endpoint of the group is 7, so when
        sorting it into the array of endpoints ``array([5])``, it will
        be sorted into the zero-position. This is already correct, as it
        clearly belongs to the 0th (and only) group. This is confirmed
        by checking where the particle ID 5 would be sorted in the array
        of starting points ``array([0])``. Here it is sorted into
        position 1, so shifted by exactly one position. This confirms
        that it is a bound group member.

        Compare this to a particle of ID 8: in the array of endpoints it
        would be sorted into position 1, seemingly indicating that it
        belongs to group 1. However, the particle would be sorted into
        the same position in the array of starting points. This indicates
        that the particle is not part of a group, but unbound.

        This is especially important for inner fuzz: consider now 18
        particles, with two groups of length 8, the first starting at
        index 0, the second starting at index 10. The start points are
        ``array([0, 10,])`` and the endpoints are ``array([8, 18])``.
        If we look at a particle with ID 9 (clearly part of the inner
        fuzz), we get the following results:

        - Sorted into the end points ``array([8, 18])`` we get index 1,
          seemingly indicating membership in the second group. But:
        - Sorted into the starting points ``array([0, 10])``, the particle
          is sorted into position 1 as well, which is not shifte by 1
          with respec tto the previous result.

        The particle is thus correctly characterized as unbound fuzz.

    :param particle_ids: Array of particle IDs. Shape (X, )
    :param offsets: Array of offsets, i.e. the IDs of the first particle
        in every (sub)group. Must be of shape (N, ) where N is the number
        of (sub)groups.
    :param lengths: Array of number of particles per (sub)groups. Must be
        of shape (N, ).
    :return: Array of parent IDs, i.e. the indices into the array of
        offsets that the corresponding particle ID in ``particle_ids``
        belongs to. If there is no parent, the parent is set to -1.
    """
    # find indices where the particle would be sorted into:
    index_wrt_start = np.searchsorted(offsets, particle_ids, side="right")
    index_wrt_end = np.searchsorted(
        offsets + lengths, particle_ids, side="right"
    )
    # create array of indices
    parent_ids = index_wrt_end.copy().astype(np.int64)
    parent_ids[index_wrt_start - index_wrt_end != 1] = -1
    return parent_ids


AllResults: TypeAlias = tuple[NDArray, NDArray, NDArray, NDArray]
OnlyFoF: TypeAlias = tuple[NDArray, NDArray, None, None]
OnlySubhalo: TypeAlias = tuple[None, None, NDArray, NDArray]


def load_offsets_and_lens(
    base_path: Path | str,
    snap_num: int,
    group_only: bool = False,
    subhalo_only: bool = False,
) -> AllResults | OnlyFoF | OnlySubhalo:
    """
    Helper function; return offsets and lens for FoF groups and subhalos.

    Function is factored out so it may be easily patched in unit tests.

    :param base_path: Simulation base path.
    :param snap_num: Snapshot number.
    :param group_only: Set to True to only load offsets and lens for
        FoF groups.
    :param subhalo_only: Set to True to only load offsets and lens for
        subhalos.
    :return: Tuple of group offsets by type, group lengths by type,
        subhalo offsets by type, subhalo lengths by type, in that order.
        If either group or subhalo offsets/lens were not loaded due to
        setting ``group_only`` or ``subhalo_only``, then the offsets and
        lengths that were not loaded are None instead.
    """
    if not isinstance(base_path, Path):
        base_path = Path(base_path)

    group_offsets, group_lens = None, None
    subhalo_offsets, subhalo_lens = None, None

    offset_directory = base_path.parent / "postprocessing/offsets"
    offset_filepath = f"offsets_{snap_num:03d}.hdf5"

    # load offsets and group lengths
    if not subhalo_only:
        with h5py.File(offset_directory / offset_filepath, "r") as offset_file:
            group_offsets = offset_file["Group/SnapByType"][()]
        group_lens = il.groupcat.loadHalos(
            str(base_path), snap_num, ["GroupLenType"]
        )

    # load offsets and subhalo lengths
    if not group_only:
        with h5py.File(offset_directory / offset_filepath, "r") as offset_file:
            subhalo_offsets = offset_file["Subhalo/SnapByType"][()]
        subhalo_lens = il.groupcat.loadSubhalos(
            str(base_path), snap_num, ["SubhaloLenType"]
        )

    # return tuple of results
    return group_offsets, group_lens, subhalo_offsets, subhalo_lens
