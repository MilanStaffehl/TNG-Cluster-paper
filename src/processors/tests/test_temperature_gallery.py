import logging

import numpy as np
import pytest

import compute
import constants
from processors import temperature_gallery


@pytest.fixture
def minimal_setup():
    logger = logging.getLogger("root")
    processor = temperature_gallery.TemperatureDistributionGalleryProcessor(
        "TNG300-1", logger, data_length=10, mass_bins=[1e8, 1e9, 1e10, 1e11]
    )
    yield processor


@pytest.fixture
def full_setup():
    logger = logging.getLogger("root")
    processor = temperature_gallery.TemperatureDistributionGalleryProcessor(
        "TNG300-1",
        logger,
        data_length=50,
        mass_bins=[1e8, 1e9, 1e10, 1e11, 1e12, 1e13, 1e14, 1e15],
    )
    yield processor


@pytest.fixture
def mock_return():
    mock_masses = np.power(10, 8 + np.random.rand(200) * 3)
    mock_masses = mock_masses / 1e10 * constants.HUBBLE
    mock_radii = np.random.rand(200) * 2500 * constants.HUBBLE
    mock_return = {
        "Group_M_Crit200": mock_masses, "Group_R_Crit200": mock_radii
    }
    yield mock_return


def test_temperature_gallery_get_halo_data(minimal_setup, mock_return, mocker):
    """Test the method that loads and selects halos by mass"""
    proc = minimal_setup
    proc.plots_per_bin = 2
    mock_load = mocker.patch(
        "illustris_python.groupcat.loadHalos", return_value=mock_return
    )

    # run the method
    proc._get_halo_data()

    # verify the load function was called as expected
    mock_load.assert_called_with(
        "/virgotng/universe/IllustrisTNG/TNG300-1/output",
        99,
        fields=["Group_M_Crit200", "Group_R_Crit200"]
    )

    # verify results
    assert len(proc.indices) == 12  # 3 mass bins x 4 halos chosen
    assert len(proc.masses) == 12
    assert len(proc.radii) == 12

    mock_masses = mock_return["Group_M_Crit200"]
    mock_radii = mock_return["Group_R_Crit200"]
    for i, index in enumerate(proc.indices):
        assert mock_masses[index] * 1e10 / constants.HUBBLE == proc.masses[i]
        assert mock_radii[index] / constants.HUBBLE == proc.radii[i]

    # assert the masses actually fall into the expected mass range
    for i in range(proc.n_mass_bins):
        assert (
            proc.mass_bins[i] <= proc.masses[i * 4 + 0] < proc.mass_bins[i + 1]
        )
        assert (
            proc.mass_bins[i] <= proc.masses[i * 4 + 1] < proc.mass_bins[i + 1]
        )
        assert (
            proc.mass_bins[i] <= proc.masses[i * 4 + 2] < proc.mass_bins[i + 1]
        )
        assert (
            proc.mass_bins[i] <= proc.masses[i * 4 + 3] < proc.mass_bins[i + 1]
        )


def test_temperature_gallery_get_auxilary_data(
    minimal_setup, mock_return, mocker
):
    """Test the method that calculates the virial temperatures"""
    proc = minimal_setup
    proc.plots_per_bin = 2
    mocker.patch(
        "illustris_python.groupcat.loadHalos", return_value=mock_return
    )
    proc._get_halo_data()
    proc._get_auxilary_data(0, False)

    # verify results
    assert len(proc.virial_temperatures) == 12  # equal to lenght of masses
    mock_masses = mock_return["Group_M_Crit200"] * 1e10 / constants.HUBBLE
    mock_radii = mock_return["Group_R_Crit200"] / constants.HUBBLE
    vt = compute.get_virial_temperature(mock_masses, mock_radii)
    for i, index in enumerate(proc.indices):
        assert proc.virial_temperatures[i] == vt[index]


def test_temperature_gallery_post_process_data(
    minimal_setup, mock_return, mocker
):
    """Test the reshaping of data in the post processing method"""
    proc = minimal_setup
    proc.plots_per_bin = 2
    mocker.patch(
        "illustris_python.groupcat.loadHalos", return_value=mock_return
    )
    proc._get_halo_data()
    proc._get_auxilary_data(0, False)
    # mock the data array
    proc.data = np.zeros((12, 10))
    # NEVER REMOVE THE to_file=False OR IT WILL OVERWRITE REAL DATA!
    proc._post_process_data(0, False, to_file=False)

    # assert shapes
    assert proc.indices.shape == (3, 4)
    assert proc.masses.shape == (3, 4)
    assert proc.radii.shape == (3, 4)
    assert proc.virial_temperatures.shape == (3, 4)
    assert proc.data.shape == (3, 4, 10)
