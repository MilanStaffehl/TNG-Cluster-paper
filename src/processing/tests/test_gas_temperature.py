"""
Unit tests for the processing.gas_temperature module.
"""
import logging

import numpy as np
import pytest

from processing import gas_temperatures


@pytest.fixture
def mock_gas_data():
    """
    Yield mock gas data dictionary.

    The dictionary can be used to produce a simple 3x3 2D histogram of
    the following form:

    [[1 2 3]
     [0 3 1]
     [2 1 0]]
    """
    # create mock gas data
    temperatures = np.array(
        [2e5, 2e6, 2e7, 3e5, 3e6, 3e7, 4e5, 4e6, 4e7, 5e6, 6e6, 7e6, 5e7]
    )
    radii = np.array(
        [0.5, 0.5, 1.5, 0.6, 1.6, 2.6, 2.7, 1.7, 2.7, 1.8, 2.8, 2.8, 2.8]
    )
    # choose masses such that weights will all be 1
    masses = np.ones_like(temperatures)
    mock_gas_data = {
        "Temperature": temperatures,
        "Masses": masses,
        "Radius": radii,
    }
    yield mock_gas_data


def test_get_temperature_2d_histogram(mock_gas_data):
    """
    Test the function that turns two arrays into a 2D histogram.
    """
    test_data = gas_temperatures.get_temperature_2d_histogram(
        mock_gas_data,
        "Radius",
        np.array([[0, 3], [5, 8]]),
        n_bins_temperature=3,
        n_bins_x_axis=3,
        convert_units=False,
    )

    # assert that the returned array is as expected
    assert test_data.shape == (3, 3)
    norm = len(mock_gas_data["Temperature"])
    expected = np.array([[2, 0, 1], [1, 3, 2], [0, 1, 3]]) / norm
    print(f"Expected:\n{expected}\nActual:\n{test_data}\n")
    np.testing.assert_array_almost_equal_nulp(test_data, expected)


def test_get_temperature_2d_histogram_missing_field(mock_gas_data, caplog):
    """
    Test that an exception is raised when the field does not exist.
    """
    caplog.set_level(logging.INFO)
    test_data = gas_temperatures.get_temperature_2d_histogram(
        mock_gas_data,
        "IDoNotExist",
        np.array([[0, 3], [5, 8]]),
        n_bins_temperature=3,
        n_bins_x_axis=3,
    )
    expected = np.empty((3, 3))
    expected.fill(np.nan)
    np.testing.assert_array_equal(test_data, expected)
    for record in caplog.records:
        if record.levelname == "ERROR":
            assert record.msg == (
                "The chosen field IDoNotExist is not in the gas data "
                "dictionary. Returning array of NaN instead."
            )
