"""
Test the DAQ module for halos.
"""
import numpy as np
import pytest

import library.data_acquisition as daq
from library import constants, units


@pytest.fixture
def mock_halo_data():
    """
    Provide a mock halo data dictionary.
    """
    mass = np.linspace(0, 1, 40)
    radius = np.linspace(10, 100, 40)
    mock_data = {"Group_M_Crit200": mass, "Group_R_Crit200": radius}
    yield mock_data


def test_get_halo_properties(mock_halo_data, mocker):
    """
    Test the basic functionality.
    """
    # patch external call to loading func
    mock_loader = mocker.patch("illustris_python.groupcat.loadHalos")
    mock_loader.return_value = mock_halo_data
    # test function
    fields = ["Group_M_Crit200", "Group_R_Crit200"]
    test_data = daq.halos.get_halo_properties("base/path", 99, fields)
    # assert mock calls
    mock_loader.assert_called_with("base/path", 99, fields=fields)
    # verify data structure
    assert isinstance(test_data, dict)
    assert "Group_M_Crit200" in test_data.keys()
    assert "Group_R_Crit200" in test_data.keys()
    assert "IDs" in test_data.keys()
    # verify data values (unit conversion): masses
    expected_mass = mock_halo_data["Group_M_Crit200"] * 1e10 / constants.HUBBLE
    np.testing.assert_equal(test_data["Group_M_Crit200"], expected_mass)
    # verify radii (units converted)
    expected_radius = mock_halo_data["Group_R_Crit200"] / constants.HUBBLE
    np.testing.assert_equal(test_data["Group_R_Crit200"], expected_radius)
    # verify IDs
    expected_ids = np.linspace(0, 39, 40)
    np.testing.assert_equal(test_data["IDs"], expected_ids)


def test_get_halo_properties_unsupported_unit():
    """
    Test that an error is raised for unsupported units.
    """
    # provoke an UnsupportedUnitError
    with pytest.raises(units.UnsupportedUnitError):
        fields = ["Group_M_Crit200", "Group_R_Crit200", "Unsupported"]
        daq.halos.get_halo_properties("base/path", 99, fields)
