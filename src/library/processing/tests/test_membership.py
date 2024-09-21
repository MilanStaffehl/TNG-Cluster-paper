"""
Tests for the membership module.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Iterator

import numpy as np
import pytest

from library.processing import membership

if TYPE_CHECKING:
    from unittest.mock import Mock

    from numpy.typing import NDArray


@pytest.fixture
def mock_data() -> Iterator[tuple[NDArray, NDArray, NDArray, NDArray]]:
    """
    Return a tuple of mock data.

    :return: Tuple of mock data arrays: mock group offsets, mock group
        lengths, mock subhalo offsets, mock subhalo lengths, in that
        order.
    """
    mock_group_offsets = np.array([0, 80, 155, 205, 220, 228])
    mock_group_lens = np.array([80, 75, 50, 15, 8, 6])
    mock_subhalo_offsets = np.array(
        [0, 20, 40, 55, 80, 98, 110, 155, 173, 191, 205, 215, 220, 228]
    )
    mock_subhalo_lens = np.array(
        [20, 20, 15, 10, 18, 12, 9, 18, 18, 2, 10, 4, 6, 6]
    )
    yield (
        mock_group_offsets,
        mock_group_lens,
        mock_subhalo_offsets,
        mock_subhalo_lens,
    )


@pytest.fixture
def mock_pids() -> Iterator[tuple[NDArray, NDArray, NDArray]]:
    """Return an array of mock particle IDs and expected parents"""
    # yapf: disable
    pids = np.array(
        [0, 15, 20, 39, 40, 65, 68, 80, 98, 100, 119, 120, 155, 192, 200,
         205, 215, 219, 220, 226, 227, 228, 230, 234, 249]
    )
    expected_g_parents = np.array(
        [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2,
         3, 3, 3, 4, 4, 4, 5, 5, -1, -1]
    )
    expected_s_parents = np.array(
        [0, 0, 1, 1, 2, -1, -1, 4, 5, 5, -1, -1, 7, 9, -1,
         10, 11, -1, 12, -1, -1, 13, 13, -1, -1]
    )
    # yapf: enable
    yield pids, expected_g_parents, expected_s_parents


@pytest.fixture
def patch_offset_loader(mocker) -> Iterator[Mock]:
    """Patch loading of offsets and group lengths from file"""
    # we use smaller example here, to keep it simple: 3 groups, 5 subgroups
    # for 50 gas particles, 30 stars, 8 BHs:
    group_offset_by_type = np.zeros((3, 6))
    group_len_by_type = np.zeros((3, 6))
    subhalo_offset_by_type = np.zeros((5, 6))
    subhalo_len_by_type = np.zeros((5, 6))

    # GAS PARTICLES
    group_offset_by_type[:, 0] = np.array([0, 20, 35])
    group_len_by_type[:, 0] = np.array([20, 15, 10])
    subhalo_offset_by_type[:, 0] = np.array([0, 10, 20, 28, 35])
    subhalo_len_by_type[:, 0] = np.array([10, 8, 10, 2, 7])

    # STAR PARTICLES
    group_offset_by_type[:, 4] = np.array([0, 10, 18])
    group_len_by_type[:, 4] = np.array([10, 8, 7])
    subhalo_offset_by_type[:, 4] = np.array([0, 5, 10, 12, 18])
    subhalo_len_by_type[:, 4] = np.array([5, 2, 2, 3, 4])

    # BH PARTICLES
    group_offset_by_type[:, 5] = np.array([0, 3, 5])
    group_len_by_type[:, 5] = np.array([3, 2, 2])
    subhalo_offset_by_type[:, 5] = np.array([0, 1, 3, 4, 5])
    subhalo_len_by_type[:, 5] = np.array([1, 1, 1, 0, 0])

    # patch loading function to return the mock data
    m = mocker.patch("library.processing.membership.load_offsets_and_lens")
    m.return_value = (
        group_offset_by_type,
        group_len_by_type,
        subhalo_offset_by_type,
        subhalo_len_by_type,
    )
    yield m


@pytest.fixture
def mock_pids_by_type() -> Iterator[tuple[NDArray, NDArray, NDArray, NDArray]]:
    """
    Return mock data and expected returns.

    Fixture returns an array of particle IDs and corresponding particle
    type flags for use with the ``patch_offset_loader`` fixture, as well
    as the expected parent IDs.
    """
    # yapf: disable
    pids = np.array([
        0, 5, 12, 18, 20, 30, 32, 35, 40, 43, 48,  # gas
        0, 4, 6, 8, 11, 12, 16, 20, 23, 28,  # stars
        0, 1, 2, 3, 4, 5, 6, 7, 8,  # BHs
    ])
    part_types = np.array([
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  # gas
        4, 4, 4, 4, 4, 4, 4, 4, 4, 4,  # stars
        5, 5, 5, 5, 5, 5, 5, 5, 5,  # BHs
    ])
    expected_group_parents = np.array([
        0, 0, 0, 0, 1, 1, 1, 2, 2, 2, -1,  # gas
        0, 0, 0, 0, 1, 1, 1, 2, 2, -1,  # stars
        0, 0, 0, 1, 1, 2, 2, -1, -1,  # BHs
    ])
    expected_subhalo_parents = np.array([
        0, 0, 1, -1, 2, -1, -1, 4, 4, -1, -1,  # gas
        0, 0, 1, -1, 2, 3, -1, 4, -1, -1,  # stars
        0, 1, -1, 2, -1, -1, -1, -1, -1  # BHs
    ])
    # yapf: enable
    yield pids, part_types, expected_group_parents, expected_subhalo_parents


def test_find_parent(mock_data, mock_pids) -> None:
    """Test the _find_parent function."""
    pids, expected_parent_halos, expected_parent_subhalos = mock_pids
    g_offset, g_lens, s_offset, s_lens = mock_data

    # test for group
    output = membership.find_parent(pids, g_offset, g_lens)
    np.testing.assert_array_equal(expected_parent_halos, output)

    # test for subhalos
    output = membership.find_parent(pids, s_offset, s_lens)
    np.testing.assert_array_equal(expected_parent_subhalos, output)


def test_find_parents_empty_groups() -> None:
    """Test edge case: empty group in the middle of other groups."""
    # scenario 1: sandwiched group with no members (second group)
    offset = np.array([0, 4, 4])
    length = np.array([4, 0, 2])
    pids = np.array([3, 4, 5])
    expected = np.array([0, 2, 2])
    output = membership.find_parent(pids, offset, length)
    np.testing.assert_array_equal(expected, output)

    # scenario 2: empty group followed by fuzz
    offset = np.array([0, 4])
    length = np.array([4, 0])
    pids = np.array([3, 4, 5])
    expected = np.array([0, -1, -1])
    output = membership.find_parent(pids, offset, length)
    np.testing.assert_array_equal(expected, output)


def test_particle_parents(
    patch_offset_loader: Mock,
    mock_pids_by_type: tuple[NDArray, NDArray, NDArray, NDArray],
) -> None:
    """Test the function to determine particle membership."""
    pids, ptype, expected_g_parents, expected_s_parents = mock_pids_by_type
    output = membership.particle_parents(pids, ptype, 99, "base/path/string")

    # assert output
    np.testing.assert_array_equal(output[0], expected_g_parents)
    np.testing.assert_array_equal(output[1], expected_s_parents)

    # assert output dtype
    assert np.issubdtype(output[0].dtype, np.int64)
    assert np.issubdtype(output[1].dtype, np.int64)

    # assert method call
    patch_offset_loader.assert_called_with(Path("base/path/string"), 99)
