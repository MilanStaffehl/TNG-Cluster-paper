"""
Test the DAQ module for gas cells.
"""
import numpy as np

import library.data_acquisition as daq

fields = ["InternalEnergy", "ElectronAbundance", "Masses", "StarFormationRate"]


def mock_data(additional_field=False):
    """
    Return a dataset to mock loaded data from a simlation.
    """
    mock_data_ = {
        "count": 10,
        "InternalEnergy": np.array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19]),
        "ElectronAbundance": np.array([0, .1, .2, .3, .4, .5, .6, .7, .8, .9]),
        "StarFormationRate": np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
        "Masses":
            np.array([100, 200, 300, 400, 500, 600, 700, 800, 900, 1000])
    }
    if additional_field:
        mock_data_.update(
            {
                "AdditionalField":
                    np.array(
                        [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5]
                    )
            }
        )
    return mock_data_


def mock_temperatures():
    """
    Return an array of mock temperatures for patching of the compute module.
    """
    return np.array([2e6, 4e4, 8e6, 3e5, 9e4, 5e5, 1e7, 2e3, 9e4, 1e9])


def test_get_halo_temperatures(mocker):
    """
    Test the function for benevolent data and minimal args.
    """
    mock_loader = mocker.patch("illustris_python.snapshot.loadHalo")
    mock_loader.return_value = mock_data()
    mock_temps = mocker.patch("library.compute.get_temperature")
    mock_temps.return_value = mock_temperatures()
    # call the function
    test_data = daq.gas.get_halo_temperatures(174, "base/path", 99)
    # assert results
    expected_data = mock_data()
    expected_data.update({"Temperature": mock_temperatures()})
    for key in test_data.keys():
        np.testing.assert_array_equal(test_data[key], expected_data[key])
    # assert mock calls
    mock_loader.assert_called_with(
        "base/path", 99, 174, partType=0, fields=fields
    )
    np.testing.assert_array_equal(
        mock_temps.call_args[0][0],
        np.array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19])
    )
    np.testing.assert_array_equal(
        mock_temps.call_args[0][1],
        np.array([0, .1, .2, .3, .4, .5, .6, .7, .8, .9])
    )
    np.testing.assert_array_equal(
        mock_temps.call_args[0][2], np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    )


def test_get_halo_temperatures_additional_fields(mocker):
    """
    Test the function when instructed to load additional fields.
    """
    mock_loader = mocker.patch("illustris_python.snapshot.loadHalo")
    mock_loader.return_value = mock_data(True)
    mock_temps = mocker.patch("library.compute.get_temperature")
    mock_temps.return_value = mock_temperatures()
    # call the function
    test_data = daq.gas.get_halo_temperatures(
        174, "base/path", 99, additional_fields=["AdditionalField"]
    )
    # assert result
    assert "AdditionalField" in test_data.keys()
    np.testing.assert_array_equal(
        test_data["AdditionalField"],
        np.array([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5])
    )
    # assert mock calls
    mock_loader.assert_called_with(
        "base/path", 99, 174, partType=0, fields=fields + ["AdditionalField"]
    )
    # assert results
    expected_data = mock_data()
    expected_data.update({"Temperature": mock_temperatures()})
