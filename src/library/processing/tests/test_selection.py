"""Tests for the selection module"""
import numpy as np

from library.processing import selection


def test_digitize_clusters() -> None:
    """Test that overflowing masses are assigned correct indices"""
    test_masses = 10**np.array([14.1, 14.3, 14.7, 15.0, 15.3999, 15.4001])
    expected = np.array([1, 2, 4, 6, 7, 7])
    output = selection.digitize_clusters(test_masses)
    np.testing.assert_array_equal(expected, output)


def test_digitize_clusters_custom_bins() -> None:
    """Test the function for custom bin edges"""
    test_masses = np.array([1, 4, 3, 0, 5, 600, -1])
    test_bins = np.array([0, 2, 4])
    expected = np.array([1, 2, 2, 1, 2, 2, 0])
    output = selection.digitize_clusters(test_masses, test_bins)
    np.testing.assert_array_equal(expected, output)
