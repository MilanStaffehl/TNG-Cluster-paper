"""
Test the config module.
"""
from pathlib import Path

from config import config


def test_default_config():
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


def test_custom_config():
    """
    Test the config received when specifying parameters.
    """
    test_cfg = config.get_default_config(
        "TNG50-2", 50, "Group_M_Crit500", "Radius"
    )
    sim_path = "/virgotng/universe/IllustrisTNG/TNG50-2/output"
    assert test_cfg.base_path == sim_path
    assert test_cfg.snap_num == 50
    assert test_cfg.mass_field == "Group_M_Crit500"
    assert test_cfg.radius_field == "Radius"
    root_dir = Path(__file__).parents[3].resolve()
    assert Path(test_cfg.data_home).resolve() == root_dir / "data"
    assert Path(test_cfg.figures_home).resolve() == root_dir / "figures"


def test_custom_paths(mocker):
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
