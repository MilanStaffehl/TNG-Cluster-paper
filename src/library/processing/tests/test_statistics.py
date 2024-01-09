"""
Unit tests for the statistics module.
"""
import numpy as np
import numpy.ma as ma
import pytest

from library.processing import selection, statistics


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


def test_mask_quantity_1d():
    """
    Test the masking function for a simple 1D array.
    """
    input_array = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    mask = np.array([0, 0, 1, 1, 0, 0, 1, 1, 0, 0])
    expected_output = np.array([2, 3, 6, 7])
    # test function
    output = selection.mask_quantity(input_array, mask, index=1, compress=True)
    np.testing.assert_equal(expected_output, output)
    assert not isinstance(output, ma.MaskedArray)


def test_mask_quantity_2d():
    """
    Test the masking function for a 2D matrix.
    """
    input_array = np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]])
    mask = np.array([1, 0, 1, 0, 1])
    expected_output = np.array([[0, 1], [4, 5], [8, 9]])
    # test function
    output = selection.mask_quantity(input_array, mask, index=1, compress=True)
    np.testing.assert_equal(expected_output, output)
    assert not isinstance(output, ma.MaskedArray)


def test_mask_quantity_3d():
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
    output = selection.mask_quantity(input_array, mask, index=1, compress=True)
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
    output = selection.mask_quantity(
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
    output = selection.mask_quantity(
        input_array, mask, index=1, compress=False
    )
    # assert dimensions
    assert ma.isMaskedArray(output)
    assert output.shape == (3, 2)


@pytest.fixture
def hist_data():
    """Yield x- and y-data for a simple 2D histogram."""
    x_data = np.array([1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3])
    y_data = np.array([1, 2, 2, 2, 3, 3, 1, 1, 1, 1, 1, 1, 3, 3, 2, 3])
    yield x_data, y_data


def test_column_normalized_hist2d_density(hist_data):
    """
    Test that the function can return a simple count-based hist2d.
    """
    x_data, y_data = hist_data
    output = statistics.column_normalized_hist2d(x_data, y_data, (3, 3))
    assert output[0].shape == (3, 3)
    expected = np.array(
        [[1 / 6, 3 / 4, 0], [1 / 2, 0, 1 / 2], [1 / 3, 1 / 4, 1 / 2]]
    )
    np.testing.assert_almost_equal(output[0], expected, decimal=2)


def test_column_normalized_hist2d_range(hist_data):
    """
    Test the function using the range normalization.
    """
    x_data, y_data = hist_data
    output = statistics.column_normalized_hist2d(
        x_data, y_data, (3, 3), normalization="range"
    )
    assert output[0].shape == (3, 3)
    expected = np.array([[1 / 3, 1, 0], [1, 0, 1], [2 / 3, 1 / 3, 1]])
    np.testing.assert_almost_equal(output[0], expected, decimal=2)


def test_column_normalized_hist2d_existing_histogram():
    """
    Test the function using an existing histogram to normalize.
    """
    # histogram is in shape (nx, ny) - this should be respected by func!
    histogram = np.array(
        [[1, 3, 2],
         [6, 0, 2],
         [0, 1, 1]]
    )  # yapf: disable

    # density
    output = statistics.column_normalized_hist2d(histogram, None, None)
    assert output[0].shape == (3, 3)
    assert output[1] is None and output[2] is None
    expected = np.array(
        [[1 / 6, 3 / 4, 0], [1 / 2, 0, 1 / 2], [1 / 3, 1 / 4, 1 / 2]]
    )
    np.testing.assert_almost_equal(output[0], expected, decimal=2)

    # range
    output = statistics.column_normalized_hist2d(
        histogram, None, None, normalization="range"
    )
    assert output[0].shape == (3, 3)
    assert output[1] is None and output[2] is None
    expected = np.array([[1 / 3, 1, 0], [1, 0, 1], [2 / 3, 1 / 3, 1]])
    np.testing.assert_almost_equal(output[0], expected, decimal=2)
