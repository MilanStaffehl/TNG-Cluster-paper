"""
Utilities for the cool gas history archive.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import h5py

if TYPE_CHECKING:
    from numpy.typing import NDArray


def index_to_global(
    zoom_in: int,
    indices: NDArray,
    type_flags: NDArray,
    total_part_num: NDArray | tuple[float, float, float],
    file_offsets: NDArray | None = None,
    base_path: str | Path | None = None,
    snap_num: int | None = None,
) -> NDArray:
    """
    Convert indices from the archive file into global particle indices.

    Indices in the cool gas history archive file are indices into the
    contiguous array of particles of type 0, 4, and 5 of only that
    zoom-in, i.e. they point to particles in an artificial array that
    one constructs by loading all gas, star and BH particles of only
    the cluster FoF and the fuzz of that zoom-in and concatenates them
    in that order. This is useful for quick access to particle data, but
    not useful when working with the full simulation box. Here, it is
    customary to point to particles by their global index, sorted by
    type, i.e. global indices point to the position of the particle in
    the array of _all_ particles of that type in the full simulation
    volume.

    This function converts the indices in the archive file to global
    indices, so that they can be used for example for sorting them
    into parent FoFs or subhalos using the
    :mod:`library.processing.membership` module.

    .. note:: In order to determine indices for a particle, it is
        necessary to know the index of the first particle of every
        FoF file and fuzz file. These are available through the header
        of every simulation snapshot file and can thus either be
        loaded by this function, provided the simulation base path, or
        they can be given directly via ``file_offsets``. In the latter
        case it must be a shape ``(N, 6)``, where ``N`` is the number of
        files.

    .. attention:: Either the base path plus snap num or the array of
        file offsets _must_ be specified, otherwise the function cannot
        work and will raise an exception.

    :param zoom_in: The ID of the zoom-in from which the indices stem.
    :param indices: The array of indices from the archive file to convert
        to global indices.
    :param type_flags: The corresponding particle type flags, from the
        archive file, too.
    :param total_part_num: The total number of particles of type 0, 4,
        and 5 respectively in the current zoom-in region at this snapshot.
        Required to turn indices for type 4 and 5 back to zero-based
        indices. This is given in the archive file as the field named
        ``total_particle_num``.
    :param file_offsets: The array of file offsets for the current
        snapshot. Must be an array of shape ``(N, 6)``, giving the index
        of the first particle of every type in each of the N files of
        the snapshot. Optional, but must be supplied if ``base_path``
        is not provided.
    :param base_path: The base path of the simulation. When given, the
        file offsets are loaded from file, unless they are explicitly
        specified using ``file_offsets``. Optional, but must be supplied
        if ``file_offsets`` is not provided.
    :param snap_num: The snapshot from which the indices come. Only
        required if file offsets must be loaded from file, i.e. only
        required when ``base_path`` is provided.
    :raises RuntimeError: If neither the base path nor the file offsets
        are specified.
    :return: The array of indices ``indices``, but as global indices,
        pointing to the array of particles of their respective type
        containing _all_ particles of the simulation volume at the
        current snapshot.
    """
    # copy indices to avoid altering in place:
    indices = indices.copy()
    # load missing offsets or raise exception for missing info
    if file_offsets is None:
        if base_path is None or snap_num is None:
            logging.error(
                "`index_to_global`: Neither the base path with snap num nor "
                "the file offsets were fully provided. Cannot determine "
                "global indices."
            )
            raise RuntimeError("Missing file offsets or base path")
        # load offsets from file
        offset_directory = Path(base_path).parent / "postprocessing/offsets"
        offset_file = offset_directory / f"offsets_{snap_num:03d}.hdf5"
        with h5py.File(offset_file, "r") as file:
            file_offsets = file["FileOffsets/SnapByType"][()]

    # convert type 4 and 5 indices to zero-base
    indices[type_flags == 4] -= total_part_num[0]
    indices[type_flags == 5] -= (total_part_num[0] + total_part_num[1])

    # find the global index of all FoF particles
    for part_type in [0, 4, 5]:
        # find how many particles belong to the fuzz file for this type
        offset_this_file = file_offsets[zoom_in][part_type]
        offset_next_file = file_offsets[zoom_in + 1][part_type]
        n_part_in_fof = offset_next_file - offset_this_file
        # distinguish particles in FoF file and in fuzz file...
        is_in_fof_file = (type_flags == part_type) & (indices < n_part_in_fof)
        is_in_fuzz_file = (type_flags == part_type) & (
            indices >= n_part_in_fof
        )
        # ...and add their respective offset to the particles:
        indices[is_in_fof_file] += file_offsets[zoom_in][part_type]
        indices[is_in_fuzz_file] += file_offsets[zoom_in + 352][part_type]

    return indices
