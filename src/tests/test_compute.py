"""
Test the compute module.
"""
import pytest
import compute


def test_compute_get_temperature_float():
    """
    Test the get temperature function with float input
    """
    temp = compute.get_temperature(10., 0.)
    assert temp == pytest.approx(1)
