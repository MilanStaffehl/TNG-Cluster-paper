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
    Test the function to retrieve the running average of a histogram.
    """
    mock_hist = np.array([[1, 2, 3, 4], [2, 3, 1, 0], [0, 0, 1, 2]])
    yrange = (0, 3)
    expected = np.array([7 / 6, 1.1, 1.1, 7 / 6])
    test_data = statistics.get_2d_histogram_running_average(mock_hist, yrange)
    assert len(test_data) == 4
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
    np.testing.assert_almost_equal(expected_output, output, decimal=6)
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
    np.testing.assert_almost_equal(expected_output, output, decimal=6)
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
    np.testing.assert_almost_equal(expected_output, output, decimal=6)
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
    np.testing.assert_almost_equal(expected_output, output.data, decimal=6)
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


def test_volume_normalized_radial_profile_basic():
    """Test the function for volume-normalized profiles"""
    rs = np.array([0, 1, 1, 1, 2, 2, 3, 4, 4, 4, 5, 5, 6, 8, 8, 8, 8, 9, 10])
    # construct expected result
    raw_hist = np.array([1, 3, 2, 1, 3, 2, 1, 0, 4, 2])
    edges = np.arange(0, 11, step=1)
    volumes = 4 / 3 * np.pi * (edges[1:]**3 - edges[:-1]**3)
    expected = raw_hist / volumes
    # test result
    output = statistics.volume_normalized_radial_profile(
        rs, np.ones_like(rs), 10
    )
    np.testing.assert_almost_equal(edges, output[1])
    np.testing.assert_almost_equal(expected, output[0])


def test_volume_normalized_radial_profile_with_given_range():
    """Test the function with a given radial range"""
    rs = np.array([0, 1, 1, 1, 2, 2, 3, 4, 4, 4, 5, 5, 6, 8, 8, 8, 8, 9, 10])
    # construct expected result
    raw_hist = np.array([3, 2, 1, 3, 2, 1, 0, 5])
    edges = np.arange(1, 10, step=1)
    volumes = 4 / 3 * np.pi * (edges[1:]**3 - edges[:-1]**3)
    expected = raw_hist / volumes
    # test result
    output = statistics.volume_normalized_radial_profile(
        rs, np.ones_like(rs), 8, radial_range=[1, 9]
    )
    np.testing.assert_almost_equal(edges, output[1])
    np.testing.assert_almost_equal(expected, output[0])


def test_volume_normalized_radial_profile_weighted():
    """Test the function with given weights"""
    rs = np.array([0, 1, 1, 1, 2, 2, 3, 4, 4, 4, 5, 5, 6, 8, 8, 8, 8, 9, 10])
    ws = np.array([1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 1, 1, 2, 2, 1, 2, 1, 1, 1])
    # construct expected result
    raw_hist = np.array([1, 3, 4, 2, 6, 2, 2, 0, 6, 2])
    edges = np.arange(0, 11, step=1)
    volumes = 4 / 3 * np.pi * (edges[1:]**3 - edges[:-1]**3)
    expected = raw_hist / volumes
    # test result
    output = statistics.volume_normalized_radial_profile(rs, ws, 10)
    np.testing.assert_almost_equal(edges, output[1])
    np.testing.assert_almost_equal(expected, output[0])


def test_volume_normalized_radial_profile_with_virial_radius():
    """Test the function with a given virial radius"""
    rs = np.array([0, 1, 1, 1, 2, 2, 3, 4, 4, 4, 5, 5, 6, 8, 8, 8, 8, 9, 10])
    # construct expected result
    raw_hist = np.array([1, 3, 2, 1, 3, 2, 1, 0, 4, 2])
    # edges are in units of physical units to get accurate volumes
    edges = np.arange(0, 11, step=1) * 100
    volumes = 4 / 3 * np.pi * (edges[1:]**3 - edges[:-1]**3)
    expected = raw_hist / volumes
    # test result
    output = statistics.volume_normalized_radial_profile(
        rs, np.ones_like(rs), 10, virial_radius=100
    )
    # edges returned are in units of virial radii again
    np.testing.assert_almost_equal(edges / 100, output[1])
    np.testing.assert_almost_equal(expected, output[0])


def test_volume_normalized_radial_profile_log_scale():
    """Test the function for profiles with log scale distances"""
    rs = np.array([0, 1, 1, 1, 2, 2, 3, 4, 4, 5])
    # construct expected result
    raw_hist = np.array([1, 3, 2, 1, 3])
    edges = 10**np.arange(0, 6, step=1)
    volumes = 4 / 3 * np.pi * (edges[1:]**3 - edges[:-1]**3)
    expected = raw_hist / volumes
    # test result
    output = statistics.volume_normalized_radial_profile(
        rs, np.ones_like(rs), 5, distances_are_log=True
    )
    np.testing.assert_almost_equal(np.log10(edges), output[1])
    np.testing.assert_almost_equal(expected, output[0])


def test_volume_normalized_radial_profile_log_scale_with_virial_radius():
    """Test the function with log scale distances and virial radius"""
    rs = np.array([0, 1, 1, 1, 2, 2, 3, 4, 4, 5])
    # construct expected result
    raw_hist = np.array([1, 3, 2, 1, 3])
    edges = 10**np.arange(0, 6, step=1) * 100
    volumes = 4 / 3 * np.pi * (edges[1:]**3 - edges[:-1]**3)
    expected = raw_hist / volumes
    # test result
    output = statistics.volume_normalized_radial_profile(
        rs, np.ones_like(rs), 5, distances_are_log=True, virial_radius=100
    )
    np.testing.assert_almost_equal(np.log10(edges / 100), output[1])
    np.testing.assert_almost_equal(expected, output[0])


def test_pearson_corrcoeff_per_bin():
    """Test the function for Pearson correlation coefficients"""
    xs = np.array([0, 1, 2, 3, 4, 0, 1, 2, 3, 4])
    ys = np.array([2, 3, 4, 5, 6, 6, 5, 4, 3, 2])
    masses = np.array([1, 1, 1, 1, 1, 3, 3, 3, 3, 3])
    expected = np.array([1, -1])
    output = statistics.pearson_corrcoeff_per_bin(xs, ys, masses, 0, 4, 2)
    np.testing.assert_almost_equal(output, expected, decimal=6)


def test_pearson_corrcoeff_per_bin_unordered():
    """Test the function for corr. coeff. works regardless of order"""
    xs = np.array([4, 0, 3, 1, 4, 2, 3, 0, 2, 1])
    ys = np.array([6, 6, 3, 3, 2, 4, 5, 2, 4, 5])
    masses = np.array([1, 3, 3, 1, 3, 1, 1, 1, 3, 3])
    expected = np.array([1., -1.])
    output = statistics.pearson_corrcoeff_per_bin(xs, ys, masses, 0, 4, 2)
    np.testing.assert_almost_equal(output, expected, decimal=6)


def test_pearson_corrcoeff_per_bin_real_values():
    """A more realistic scenario with scatter"""
    xs = np.array([7, 8, 8, 3, 6, 4, 7, 3, 5, 0, 7, 6, 1, 0, 6, 0, 8, 2, 8, 5])
    ys = np.array([8, 2, 6, 3, 0, 8, 2, 2, 5, 1, 4, 4, 8, 8, 7, 1, 5, 0, 5, 3])
    ms = np.array([1, 2, 1, 3, 4, 3, 4, 3, 4, 2, 4, 3, 3, 1, 3, 1, 3, 2, 1, 4])
    expected = np.array([0.312633, 0.720577, -0.082670, -0.259938])
    output = statistics.pearson_corrcoeff_per_bin(xs, ys, ms, 0, 5, 4)
    np.testing.assert_almost_equal(output, expected, decimal=5)


def test_two_side_difference_ratio_simple():
    """Test the function in a basic scenario"""
    # assume just one mass bin for simplicity
    masses = np.ones(10)
    # test data
    colors = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    ys = np.array([-5, -4, -3, -2, -1, 1, 2, 3, 4, 5])
    output = statistics.two_side_difference_ratio(ys, colors, masses, 0, 2, 1)
    np.testing.assert_almost_equal(output, np.array([3.5]), decimal=6)


def test_two_side_difference_binning():
    """Test the function with binning"""
    masses = np.array([1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2])
    colors = np.array([2, 4, 6, 0, 3, 0, 0, -3, 0, 4, 6, 8])
    ys = np.array([3, 4, 5, 0, 1, 2, 5, 4, 3, 2, 1, 0])
    output = statistics.two_side_difference_ratio(ys, colors, masses, 0, 3, 2)
    expected = np.array([4, -1 / 6])
    np.testing.assert_almost_equal(output, expected, decimal=6)
