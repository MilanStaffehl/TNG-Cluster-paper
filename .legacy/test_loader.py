import warnings

import config
import constants
import illustris_python as il
import loader
import numpy as np
import pytest

warnings.warn(
    "This module is deprecated and will likely not work.",
    DeprecationWarning,
    stacklevel=2,
)


@pytest.fixture
def test_config():
    """Prepare a testing config instance"""
    test_cfg = config.get_default_config("TNG50-3")
    yield test_cfg


@pytest.fixture
def test_data(test_config):
    """Set up test data"""
    # set up
    bins = [1e9, 1e10, 1e11, 1e12, 1e13, 1e14]
    test_data = loader.get_halos_binned_by_mass(bins, test_config)
    # yield
    yield test_data


def test_loader_get_halo_binned_by_mass_binning(test_config, mocker):
    """Test that the function will bin values correctly"""
    # patch the load function to instead return mock test data
    mock_data = np.array(
        [10, 110, 1010, 20, 0, 49, 3500, 240, 999, 5], dtype=float
    ) * 1e-10 * constants.HUBBLE  # revert scaling inside method
    mocker.patch("illustris_python.groupcat.loadHalos", return_value=mock_data)
    # call the loading function
    bins = [1, 10, 100, 1000, 10000]
    test_data = loader.get_halos_binned_by_mass(bins, test_config)
    print(test_data)

    # assert the function returns correctly ordered indices and masses
    # bin 1: [1, 10)
    expected_idx = np.array([9])
    expected_mass = np.array([5.])
    np.testing.assert_allclose(test_data[0][0], expected_idx)
    np.testing.assert_allclose(test_data[1][0], expected_mass)
    # bin 2: [10, 100)
    expected_idx = np.array([0, 3, 5])
    expected_mass = np.array([10, 20, 49])
    np.testing.assert_allclose(test_data[0][1], expected_idx)
    np.testing.assert_allclose(test_data[1][1], expected_mass)
    # bin 3: [100, 1000)
    expected_idx = np.array([1, 7, 8])
    expected_mass = np.array([110, 240, 999])
    np.testing.assert_allclose(test_data[0][2], expected_idx)
    np.testing.assert_allclose(test_data[1][2], expected_mass)
    # bin 4: [1000, 10000)
    expected_idx = np.array([2, 6])
    expected_mass = np.array([1010, 3500])
    np.testing.assert_allclose(test_data[0][3], expected_idx)
    np.testing.assert_allclose(test_data[1][3], expected_mass)


def test_loader_get_halo_binned_by_mass_types(test_data):
    """Test that the return types are correct"""
    # test top level types
    assert isinstance(test_data, tuple)
    assert isinstance(test_data[0], list)
    assert isinstance(test_data[1], list)

    # test the entries of the two lists
    for entry in test_data[0]:  # indices
        assert isinstance(entry, np.ndarray)
        assert entry.dtype is np.dtype("int64")
    for entry in test_data[1]:  # masses
        assert isinstance(entry, np.ndarray)
        assert entry.dtype is np.dtype("float64")


def test_loader_get_halo_binned_by_mass_lengths(test_config):
    """Test that the number of elements in list is valid"""
    # test adds up the lengths of the return arrays and ensures it will
    # no exceed the number of halos in the dataset
    halos = il.groupcat.loadHalos(
        test_config.base_path,
        test_config.snap_num,
        fields=test_config.mass_field,
    )
    total_num_halos = len(halos)

    # load test data such that all halos should be included
    bins = [0, 1e10, 1e11, 1e12, 1e13, 1e20]
    test_data = loader.get_halos_binned_by_mass(bins, test_config)

    # check that all halos are included in lists
    index_num = sum([len(index_list) for index_list in test_data[0]])
    assert index_num == total_num_halos
    mass_num = sum([len(mass_list) for mass_list in test_data[1]])
    assert mass_num == total_num_halos


def test_loader_get_halo_binned_by_mass_arrays_match(test_data):
    """Test that the index and mass arrays match in length"""
    assert len(test_data[0]) == len(test_data[1])
    for bin_num in range(len(test_data[0])):
        assert len(test_data[0][bin_num]) == len(test_data[1][bin_num])


def test_loader_get_halo_binned_by_mass_bins(test_config):
    """Test that returned masses fall into bins"""
    # test requires access to bins, therefore fixture test_data is not used
    bins = [1e9, 1e10, 1e11, 1e12, 1e13, 1e14]
    test_data = loader.get_halos_binned_by_mass(bins, test_config)
    # check masses
    for i, bin_ in enumerate(test_data[1]):
        for mass in bin_:
            assert bins[i] <= mass <= bins[i + 1]
