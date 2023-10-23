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


@pytest.mark.skipif(
    not Path("/virgotng/").exists(),
    reason="Cannot be executed outside of the Vera cluster."
)
def test_default_config_vera():
    """
    Test the default config returned by get_default_config.
    """
    test_cfg = config.get_default_config("TNG300-1")
    default_sim_path = "/virgotng/universe/IllustrisTNG/TNG300-1/output"
    assert test_cfg.base_path == default_sim_path
    assert test_cfg.snap_num == 99
    assert test_cfg.mass_field == "Group_M_Crit200"
    assert test_cfg.radius_field == "Group_R_Crit200"
    root_dir = Path(__file__).parents[3].resolve()
    assert Path(test_cfg.data_home).resolve() == root_dir / "data"
    assert Path(test_cfg.figures_home).resolve() == root_dir / "figures"


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
                "simulation_home": str(mock_sim_home),
            }
    }  # yapf: disable
    mock_load = mocker.patch("yaml.full_load")
    mock_load.return_value = mock_config
    test_cfg = config.get_default_config("TNG300-1")
    sim_path = str(mock_sim_home / "TNG300-1" / "output")
    assert test_cfg.base_path == sim_path
    assert test_cfg.snap_num == 99
    assert test_cfg.mass_field == "Group_M_Crit200"
    assert test_cfg.radius_field == "Group_R_Crit200"
    root_dir = Path(__file__).parents[4].resolve()
    assert Path(test_cfg.data_home).resolve() == root_dir / "data"
    assert Path(test_cfg.figures_home).resolve() == root_dir / "figures"


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
                "simulation_home": str(mock_sim_home),
            }
    }  # yapf: disable
    mock_load = mocker.patch("yaml.full_load")
    mock_load.return_value = mock_config
    test_cfg = config.get_default_config(
        "TNG50-2", 50, "Group_M_Crit500", "Radius"
    )
    sim_path = str(mock_sim_home / "TNG50-2" / "output")
    assert test_cfg.base_path == sim_path
    assert test_cfg.snap_num == 50
    assert test_cfg.mass_field == "Group_M_Crit500"
    assert test_cfg.radius_field == "Radius"
    root_dir = Path(__file__).parents[4].resolve()
    assert Path(test_cfg.data_home).resolve() == root_dir / "data"
    assert Path(test_cfg.figures_home).resolve() == root_dir / "figures"


@pytest.mark.skipif(
    platform.system() != "Linux", reason="Test only runs on Linux."
)
def test_custom_paths_linux(mocker):
    """
    Test the config received when altering the global variables.
    """
    # set global vars to some path
    mock_config = {
        "paths":
            {
                "data_home": str(Path().home() / ".local"),
                "figures_home": str(Path().home() / ".local"),
                "simulation_home": str(Path().home() / ".local"),
            }
    }
    mock_load = mocker.patch("yaml.full_load")
    mock_load.return_value = mock_config
    # create and test config
    test_cfg = config.get_default_config("TNG300-1")
    sim_path = str(Path.home() / ".local" / "TNG300-1" / "output")
    assert test_cfg.base_path == sim_path
    assert test_cfg.snap_num == 99
    assert test_cfg.mass_field == "Group_M_Crit200"
    assert test_cfg.radius_field == "Group_R_Crit200"
    home_dir = Path().home()
    assert Path(test_cfg.data_home) == home_dir / ".local"
    assert Path(test_cfg.figures_home) == home_dir / ".local"


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
        "paths":
            {
                "data_home": app_data,
                "figures_home": app_data,
                "simulation_home": app_data,
            }
    }
    mock_load = mocker.patch("yaml.full_load")
    mock_load.return_value = mock_config
    # create and test config
    test_cfg = config.get_default_config("TNG300-1")
    sim_path = str(app_data / "TNG300-1" / "output")
    assert test_cfg.base_path == sim_path
    assert test_cfg.snap_num == 99
    assert test_cfg.mass_field == "Group_M_Crit200"
    assert test_cfg.radius_field == "Group_R_Crit200"
    assert Path(test_cfg.data_home) == app_data
    assert Path(test_cfg.figures_home) == app_data


def test_invalid_paths(mocker):
    """
    Test that the config raises a custom exception for invalid paths.
    """
    # set paths to something non-existent and/or invalid
    mock_config = {
        "paths":
            {
                "data_home": "this/path/does/not/exits",
                "figures_home": "neither/does/this/path",
                "simulation_home": str(Path().home() / ".local"),
            }
    }
    mock_load = mocker.patch("yaml.full_load")
    mock_load.return_value = mock_config
    # create and test config
    with pytest.raises(config.InvalidConfigPathError) as e:
        config.get_default_config("TNG300-1")
    # figures home is tested first, so it should raise the exception
    wrong_path = Path("neither/does/this/path")
    expected_msg = f"The config path {str(wrong_path)} does not exist"
    assert str(e.value) == expected_msg
