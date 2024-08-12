"""
Test the config module.
"""
import os.path
import platform
from pathlib import Path

import pytest

from library.config import config

if platform.system() == "Windows":
    mock_sim_home = Path(os.path.expandvars("%LOCALAPPDATA%"))
else:
    mock_sim_home = Path().home() / ".local"


def test_default_config_generic(mocker):
    """
    Test the default config returned by get_default_config.
    """
    # set global vars to some path
    mock_config = {
        "paths":
            {
                "data_home": str(Path(__file__).parents[4].resolve() / "data"),
                "figures_home": str(Path(__file__).parents[4].resolve() / "figures"),
                "base_paths": {"TNG300-1": str(mock_sim_home)},
                "cool_gas_history_archive": {"TNG300-1": "default"},
            }
    }  # yapf: disable
    mock_load = mocker.patch("yaml.full_load")
    mock_load.return_value = mock_config
    test_cfg = config.get_default_config("TNG300-1")
    sim_path = mock_sim_home.resolve()
    assert test_cfg.base_path == str(sim_path)
    assert test_cfg.snap_num == 99
    assert test_cfg.mass_field == "Group_M_Crit200"
    assert test_cfg.radius_field == "Group_R_Crit200"
    root_dir = Path(__file__).parents[4].resolve()
    assert Path(test_cfg.data_home).resolve() == root_dir / "data"
    assert Path(test_cfg.figures_home).resolve() == root_dir / "figures"
    expected_file = (
        test_cfg.data_home / "tracer_history/TNG300_1/cool_gas_history.hdf5"
    )
    assert test_cfg.cool_gas_history == expected_file


def test_default_config_default(mocker):
    """
    Test that when the base paths are set to "default", a valid config is generated.
    """
    # set global vars to some path
    mock_config = {
        "paths":
            {
                "data_home": "default",
                "figures_home": "default",
                "base_paths": {"TNG300-1": str(mock_sim_home)},
                "cool_gas_history_archive": {"TNG300-1": "default"}
            }
    }  # yapf: disable
    mock_load = mocker.patch("yaml.full_load")
    mock_load.return_value = mock_config
    test_cfg = config.get_default_config("TNG300-1")
    sim_path = mock_sim_home.resolve()
    assert test_cfg.base_path == str(sim_path)
    assert test_cfg.snap_num == 99
    assert test_cfg.mass_field == "Group_M_Crit200"
    assert test_cfg.radius_field == "Group_R_Crit200"
    root_dir = Path(__file__).parents[4].resolve()
    assert Path(test_cfg.data_home).resolve() == root_dir / "data"
    assert Path(test_cfg.figures_home).resolve() == root_dir / "figures"
    expected_file = (
        test_cfg.data_home / "tracer_history/TNG300_1/cool_gas_history.hdf5"
    )
    assert test_cfg.cool_gas_history == expected_file


def test_custom_config(mocker):
    """
    Test the config received when specifying parameters.
    """
    # set global vars to some path
    mock_config = {
        "paths":
            {
                "data_home": str(Path(__file__).parents[4].resolve() / "data"),
                "figures_home": str(Path(__file__).parents[4].resolve() / "figures"),
                "base_paths": {"TNG50-2": str(mock_sim_home)},
                "cool_gas_history_archive": {"TNG50-2": "./my_dir/archive.hdf5"}
            }
    }  # yapf: disable
    mock_load = mocker.patch("yaml.full_load")
    mock_load.return_value = mock_config
    test_cfg = config.get_default_config(
        "TNG50-2", 50, "Group_M_Crit500", "Radius"
    )
    sim_path = mock_sim_home.resolve()
    assert test_cfg.base_path == str(sim_path)
    assert test_cfg.snap_num == 50
    assert test_cfg.mass_field == "Group_M_Crit500"
    assert test_cfg.radius_field == "Radius"
    root_dir = Path(__file__).parents[4].resolve()
    assert Path(test_cfg.data_home).resolve() == root_dir / "data"
    assert Path(test_cfg.figures_home).resolve() == root_dir / "figures"
    expected_file = test_cfg.data_home / "my_dir/archive.hdf5"
    assert test_cfg.cool_gas_history == expected_file


@pytest.mark.skipif(
    platform.system() != "Linux", reason="Test only runs on Linux."
)
def test_custom_paths_linux(mocker):
    """
    Test the config received when altering the global variables.
    """
    # set global vars to some path
    mock_config = {
        "paths": {
            "data_home": str(Path().home() / ".local"),
            "figures_home": str(Path().home() / ".local"),
            "base_paths": {
                "TNG300-1": str(Path().home() / ".local"),
            },
            "cool_gas_history_archive":
                {"TNG300-1": str(Path().home() / "archive.hdf5")}
        }
    }  # yapf: disable
    mock_load = mocker.patch("yaml.full_load")
    mock_load.return_value = mock_config
    # create and test config
    test_cfg = config.get_default_config("TNG300-1")
    sim_path = (Path.home() / ".local").resolve()
    assert test_cfg.base_path == str(sim_path)
    assert test_cfg.snap_num == 99
    assert test_cfg.mass_field == "Group_M_Crit200"
    assert test_cfg.radius_field == "Group_R_Crit200"
    home_dir = Path().home().resolve()
    assert Path(test_cfg.data_home) == home_dir / ".local"
    assert Path(test_cfg.figures_home) == home_dir / ".local"
    assert Path(test_cfg.cool_gas_history) == home_dir / "archive.hdf5"


@pytest.mark.skipif(
    platform.system() != "Windows", reason="Test only runs on Windows."
)
def test_custom_paths_windows(mocker):
    """
    Test the config received when altering the global variables.
    """
    # set global vars to some path
    app_data = Path(os.path.expandvars("%LOCALAPPDATA%"))
    mock_config = {
        "paths": {
            "data_home": app_data,
            "figures_home": app_data,
            "base_paths": {"TNG300-1": app_data},
            "cool_gas_history_archive": {"TNG300-1": app_data / "archive.hdf5"}
        }
    }  # yapf: disable
    mock_load = mocker.patch("yaml.full_load")
    mock_load.return_value = mock_config
    # create and test config
    test_cfg = config.get_default_config("TNG300-1")
    sim_path = str(app_data)
    assert test_cfg.base_path == sim_path
    assert test_cfg.snap_num == 99
    assert test_cfg.mass_field == "Group_M_Crit200"
    assert test_cfg.radius_field == "Group_R_Crit200"
    assert Path(test_cfg.data_home) == app_data
    assert Path(test_cfg.figures_home) == app_data
    assert Path(test_cfg.cool_gas_history) == app_data / "archive.hdf5"


def test_invalid_paths(mocker):
    """
    Test that the config raises a custom exception for invalid paths.
    """
    # set paths to something non-existent and/or invalid
    mock_config = {
        "paths":
            {
                "data_home": "/this/path/does/not/exits",
                "figures_home": "/neither/does/this/path",
                "base_paths": {
                    "TNG300-1": str(Path().home() / ".local"),
                }
            }
    }
    mock_load = mocker.patch("yaml.full_load")
    mock_load.return_value = mock_config
    # create and test config
    with pytest.raises(config.InvalidConfigPathError) as e:
        config.get_default_config("TNG300-1")
    # figures home is tested first, so it should raise the exception
    wrong_path = Path("/neither/does/this/path")
    expected_msg = f"The config path {str(wrong_path)} does not exist"
    assert str(e.value) == expected_msg


def test_invalid_simulation_name(mocker):
    """
    Test that an exception is raised when an unknown simulation name is given.
    """
    # set paths to something non-existent and/or invalid
    mock_config = {
        "paths": {
            "data_home": str(Path().home() / ".local"),
            "figures_home": str(Path().home() / ".local"),
            "base_paths": {
                "TNG300-1": str(Path().home() / ".local"),
                "TNG100-1": str(Path().home() / ".local"),
                "TNG50-1": str(Path().home() / ".local"),
                "TNG-Cluster": str(Path().home() / ".local"),
            }
        }
    }  # yapf: disable
    mock_load = mocker.patch("yaml.full_load")
    mock_load.return_value = mock_config
    # create and test config
    with pytest.raises(config.InvalidSimulationNameError) as e:
        config.get_default_config("TNG50-4")
    # TNG50-4 is not in the config file, so it should raise the exception
    expected_msg = (
        "'There is no entry for a simulation named TNG50-4 in the config.yaml "
        "configuration file for base paths.'"
    )
    assert str(e.value) == expected_msg


def test_missing_simulation_in_archive_config(mocker):
    """
    Test that an exception is raised when an unknown simulation name is given.
    """
    # set paths to something non-existent and/or invalid
    mock_config = {
        "paths": {
            "data_home": str(Path().home() / ".local"),
            "figures_home": str(Path().home() / ".local"),
            "base_paths": {
                "TNG300-1": str(Path().home() / ".local"),
                "TNG-Cluster": str(Path().home() / ".local"),
            },
            "cool_gas_data_archive": {
                "TNG-Cluster": str(Path().home() / "archive.hdf5")
            }
        }
    }  # yapf: disable
    mock_load = mocker.patch("yaml.full_load")
    mock_load.return_value = mock_config
    # create and test config
    cfg = config.get_default_config("TNG300-1")
    # TNG300-1 is not in the config file, so the path is set to None
    assert cfg.cool_gas_history is None
