"""
Unit tests for the statistics module.
"""
import numpy as np

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
