from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class Config:
    """
    Hold a runtime configuration with information about the used simulation.

    Instances of this class can be used to easily run the same code
    with different configurations, i.e. with different parameters. For
    example, one can use two different Config instances with different
    base paths to test code with a lower resolution simulation and to
    run actual analysis with higher resolution data. In such a scenario,
    depending on which config is desired, the client can pass different
    Config instances.

    :param sim_name: name of the simulation to use, must match the simulation
        under the given ``base_path``
    :param base_path: the base path of the simulation to use
    :param snap_num: number of the snapshot to use
    :param mass_field: the field name to use as mass indicator for
        groups/halos
    :param radius_field: the field name to use as radius indicator for
        groups/halos
    """
    sim_name: str
    base_path: str
    snap_num: int
    mass_field: str
    radius_field: str
    data_home: str | Path
    figures_home: str | Path

    def __post_init__(self):
        """
        Set up aux fields from existing fields.
        """
        self.sim_path: str = self.sim_name.replace("-", "_")


class InvalidConfigPathError(Exception):
    """Raise when a loaded config contains invalid paths"""

    def __init__(self, path: Path, *args: object) -> None:
        super().__init__(*args)
        if not isinstance(path, Path):
            path = Path(path)
        self.path = path

    def __str__(self) -> str:
        if not self.path.exists():
            return f"The config path {self.path} does not exist"
        elif not self.path.is_dir():
            return f"The config path {self.path} does not point to a directory"
        else:
            return f"The config path {self.path} is not a valid config path"


class InvalidSimulationNameError(KeyError):
    """Raised when an unknown simulation name is used"""

    def __init__(self, name: str, *args: object) -> None:
        msg = (
            f"There is no simulation named {name} in the config.yaml "
            f"configuration file."
        )
        super().__init__(msg, *args)


def get_default_config(
    sim: str,
    snap: int = 99,
    mass_field: str = "Group_M_Crit200",
    radius_field: str = "Group_R_Crit200"
) -> Config:
    """
    Return a configuration for the specified simulation.

    :param sim: name of the simulation as used in the simulation file
        directory, e.g. TNG50-3
    :param snap: snapshot number to use, defaults to 99 (z = 0)
    :param mass_field: name of the simulation field that signifies the
        halo mass, defaults to M_crit200
    :param radius_field: name of the simulation field that signifies the
        halo radius, defaults to R_crit200
    :return: configuration for this specific
    """
    # find directories for data and figures
    cur_dir = Path(__file__).parent.resolve()
    root_dir = cur_dir.parents[2]
    with open(root_dir / "config.yaml", "r") as config_file:
        stream = config_file.read()
    config = yaml.full_load(stream)

    # set paths
    figures_home = config["paths"]["figures_home"]
    if figures_home == "default":
        figures_home = root_dir / "figures"
    else:
        figures_home = Path(figures_home).resolve()
    data_home = config["paths"]["data_home"]
    if data_home == "default":
        data_home = root_dir / "data"
    else:
        data_home = Path(data_home).resolve()

    try:
        base_path = Path(config["paths"]["base_paths"][sim]).expanduser()
    except KeyError:
        raise InvalidSimulationNameError(sim)

    # verify paths
    for path in [figures_home, data_home, base_path]:
        if not path.exists() or not path.is_dir():
            raise InvalidConfigPathError(path)

    # return config
    # base_path = simulation_home / sim / "output"
    final_config = Config(
        sim,
        str(base_path),  # illustris_python does not support Path-likes
        snap_num=snap,
        mass_field=mass_field,
        radius_field=radius_field,
        data_home=data_home,
        figures_home=figures_home,
    )
    return final_config


def get_supported_simulations() -> list[str]:
    """Return a list of the names of supported simulations."""
    # find directories for data and figures
    cur_dir = Path(__file__).parent.resolve()
    root_dir = cur_dir.parents[2]
    with open(root_dir / "config.yaml", "r") as config_file:
        stream = config_file.read()
    config = yaml.full_load(stream)
    return list(config["paths"]["base_paths"].keys())
