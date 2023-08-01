"""
Test the compute module.
"""
import numpy as np
import pytest

import compute


def test_compute_get_temperature_float():
    """
    Test the get temperature function with float input
    """
    temp = compute.get_temperature(10., 0.)
    assert temp == pytest.approx(984.94, 0.2)
    temp = compute.get_temperature(10, 0.5)
    assert temp == pytest.approx(673.04, 0.2)


def test_compute_get_temperature_array():
    """
    Test the get temperature function with array input
    """
    internal_energy = np.array([10, 20, 100, 3000], dtype=float)
    electron_abundance = np.array([0, 0, 0.2, 1], dtype=float)
    temps = compute.get_temperature(internal_energy, electron_abundance)
    expected = np.array([984.94, 1969.88, 8309.16, 153351.28], dtype=float)
    np.testing.assert_array_almost_equal(temps, expected, decimal=2)
