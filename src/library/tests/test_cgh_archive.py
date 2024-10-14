"""
Tests for the cgh_archive module.
"""
from __future__ import annotations

import numpy as np

from library import cgh_archive as cgh


def test_index_to_global() -> None:
    """Test the function to turn indices global"""
    # create mock file offsets, but only those needed
    mock_offsets = np.zeros((354, 6), dtype=np.int64)
    # lengths: 240, 0, 0, 0, 90, 53
    mock_offsets[1] = np.array([220, 0, 0, 0, 120, 55])
    mock_offsets[2] = np.array([460, 0, 0, 0, 210, 108])
    mock_offsets[353] = np.array([120_240, 0, 0, 0, 8200, 2300])
    mock_total_part_num = np.array([2000, 1000, 100])
    mock_indices = np.array([
        0, 110, 239, 240, 320,
        2000, 2045, 2089, 2090, 2110,
        3000, 3023, 3052, 3053, 3059,
    ], dtype=np.int64)  # yapf: disable
    mock_types = np.array([
        0, 0, 0, 0, 0,
        4, 4, 4, 4, 4,
        5, 5, 5, 5, 5,
    ], dtype=np.int64)  # yapf: disable

    # create expected data
    expected_indices = np.array([
        220, 330, 459, 120_240, 120_320,
        120, 165, 209, 8200, 8220,
        55, 78, 107, 2300, 2306,
    ], dtype=np.int64)  # yapf: disable

    # test function
    output = cgh.index_to_global(
        1, mock_indices, mock_types, mock_total_part_num, mock_offsets
    )
    np.testing.assert_array_equal(expected_indices, output)
