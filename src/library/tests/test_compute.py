"""
Test the compute module.
"""
import numpy as np
import pytest

from library import compute


def test_compute_get_temperature_float():
    """
    Test the get temperature function for numpy with float input
    """
    temp = compute.get_temperature(10., 0., 0.)
    assert temp == pytest.approx(984.94, 0.2)
    temp = compute.get_temperature(10, 0.5, 0.)
    assert temp == pytest.approx(673.04, 0.2)


def test_compute_get_temperature_array():
    """
    Test the get temperature function for numpy with array input
    """
    internal_energy = np.array([10, 20, 100, 3000], dtype=float)
    electron_abundance = np.array([0, 0, 0.2, 1], dtype=float)
    star_formation_rates = np.zeros(4)
    temps = compute.get_temperature(
        internal_energy, electron_abundance, star_formation_rates
    )
    expected = np.array([984.94, 1969.88, 8309.16, 153351.28], dtype=float)
    np.testing.assert_array_almost_equal(temps, expected, decimal=2)


def test_compute_get_temperature_non_zero_sfr():
    """
    Test that when the SFR of a gas cell is non-zero, temperature is set to
    an artificial value.
    """
    temp = compute.get_temperature(10., 0., 0.5)
    assert temp == 1e3
    # test the same for arrays
    internal_energy = np.array([10, 20, 100, 3000], dtype=float)
    electron_abundance = np.array([0, 0, 0.2, 1], dtype=float)
    sfrs = np.array([1, 0, 1, 1])
    expected = np.array([1000., 1969.88, 1000., 1000.], dtype=float)
    # verification
    temps = compute.get_temperature(internal_energy, electron_abundance, sfrs)
    np.testing.assert_array_almost_equal(temps, expected, decimal=2)


def test_compute_get_temperature_vectorized_float():
    """
    Test the get temperature function with float input
    """
    temp = compute.get_temperature_vectorized(10., 0., 0.)
    assert temp == pytest.approx(984.94, 0.2)
    temp = compute.get_temperature_vectorized(10, 0.5, 0.)
    assert temp == pytest.approx(673.04, 0.2)


def test_compute_get_temperature_vectorized_array():
    """
    Test the get temperature function with array input
    """
    internal_energy = np.array([10, 20, 100, 3000], dtype=float)
    electron_abundance = np.array([0, 0, 0.2, 1], dtype=float)
    star_formation_rates = np.zeros(4)
    temps = compute.get_temperature_vectorized(
        internal_energy, electron_abundance, star_formation_rates
    )
    expected = np.array([984.94, 1969.88, 8309.16, 153351.28], dtype=float)
    np.testing.assert_array_almost_equal(temps, expected, decimal=2)


def test_compute_get_temperature_vectorized_non_zero_sfr():
    """
    Test that when the SFR of a gas cell is non-zero, temperature is set to
    an artificial value.
    """
    temp = compute.get_temperature_vectorized(10., 0., 0.5)
    assert temp == 1e3


# TODO: tests for virial temperature function


def test_compute_get_radial_velocities():
    """
    Test the function that returns radial velocities
    """
    test_center = np.array([1, 2])
    pos = np.array([[2, 2], [2, 3], [2, 1], [1, 1], [1, 3]])
    vel = np.array([[0, 1], [1, 1], [-1, 1], [1, -1], [1, -1]])
    expected = np.array([0, np.sqrt(2), -np.sqrt(2), 1, -1])

    output = compute.get_radial_velocities(test_center, pos, vel)
    np.testing.assert_allclose(output, expected, rtol=1e-4)
