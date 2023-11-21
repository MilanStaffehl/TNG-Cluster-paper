"""
Unit tests for the statistics module.
"""
import numpy as np
import numpy.ma as ma

from library.processing import statistics


def test_stack_2d_histograms_per_mass_bin():
    """
    Test the function to average 2D histograms.
    """
    # Create mock test data
    mock_mask = np.array([1, 1, 2, 2])
    mock_hists = np.array(
        [
            [[1, 2, 0], [0, 3, 1], [1, 1, 1]],
            [[1, 0, 2], [2, 1, 2], [0, 0, 0]],
            [[2, 0, 2], [3, 0, 0], [1, 2, 3]],
            [[1, 1, 0], [1, 4, 0], [1, 0, 0]],
        ]
    )
    expected_mean = np.array(
        [
            [[1, 1, 1], [1, 2, 1.5], [0.5, 0.5, 0.5]],
            [[1.5, 0.5, 1], [2, 2, 0], [1, 1, 1.5]],
        ]
    )

    test_data = statistics.stack_2d_histograms_per_mass_bin(
        mock_hists, 2, mock_mask
    )
    np.testing.assert_array_almost_equal_nulp(test_data, expected_mean)


def test_get_2d_histogram_running_average():
    """
    Test the function to retieve the running average of a histogram.
    """
    mock_hist = np.array([[1, 2, 3, 4], [2, 3, 1, 0], [0, 0, 1, 2]])
    yrange = (0, 3)
    expected = np.array([7 / 6, 1.1, 1.1, 7 / 6])
    test_data = statistics.get_2d_histogram_running_average(mock_hist, yrange)
    np.testing.assert_array_almost_equal_nulp(test_data, expected)


def test_mask_quantity_1D():
    """
    Test the masking function for a simple 1D array.
    """
    input_array = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    mask = np.array([0, 0, 1, 1, 0, 0, 1, 1, 0, 0])
    expected_output = np.array([2, 3, 6, 7])
    # test function
    output = statistics.mask_quantity(
        input_array, mask, index=1, compress=True
    )
    np.testing.assert_equal(expected_output, output)
    assert not isinstance(output, ma.MaskedArray)


def test_mask_quantity_2D():
    """
    Test the masking function for a 2D matrix.
    """
    input_array = np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]])
    mask = np.array([1, 0, 1, 0, 1])
    expected_output = np.array([[0, 1], [4, 5], [8, 9]])
    # test function
    output = statistics.mask_quantity(
        input_array, mask, index=1, compress=True
    )
    np.testing.assert_equal(expected_output, output)
    assert not isinstance(output, ma.MaskedArray)


def test_mask_quantity_3D():
    """
    Test the masking function for an 3D matrix.
    """
    input_array = np.array(
        [
            [[1, 2], [2, 3], [3, 4]],
            [[3, 4], [2, 3], [1, 2]],
            [[0, 9], [1, 8], [2, 7]],
            [[7, 2], [8, 1], [9, 0]],
            [[9, 8], [7, 6], [5, 4]],
            [[4, 6], [8, 7], [9, 5]],
        ]
    )
    mask = np.array([1, 1, 0, 0, 1, 1])
    expected_output = np.array(
        [
            [[1, 2], [2, 3], [3, 4]],
            [[3, 4], [2, 3], [1, 2]],
            [[9, 8], [7, 6], [5, 4]],
            [[4, 6], [8, 7], [9, 5]],
        ]
    )
    # test function
    output = statistics.mask_quantity(
        input_array, mask, index=1, compress=True
    )
    np.testing.assert_equal(expected_output, output)
    assert not isinstance(output, ma.MaskedArray)


def test_mask_quantity_uncompressed():
    """
    Test the masking function when setting ``compress=False``.
    """
    input_array = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    mask = np.array([0, 0, 1, 1, 0, 0, 1, 1, 0, 0])
    expected_output = np.array([2, 3, 6, 7])
    # test function
    output = statistics.mask_quantity(
        input_array, mask, index=1, compress=False
    )
    np.testing.assert_equal(expected_output, output.data)
    assert ma.isMaskedArray(output)


def test_mask_quantity_uncompressed_dimensions():
    """
    Test that uncompressed masked arrays preserve dimensionality
    """
    input_array = np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]])
    mask = np.array([1, 0, 1, 0, 1])
    output = statistics.mask_quantity(
        input_array, mask, index=1, compress=False
    )
    # assert dimensions
    assert ma.isMaskedArray(output)
    assert output.shape == (3, 2)
