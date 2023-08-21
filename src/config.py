from dataclasses import dataclass
from pathlib import Path

# overrides for default data and figures directories
DATA_HOME = None
FIGURES_HOME = None


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

    :param sim: name of the simulation to use, must match the simulation
        under the given ``base_path``
    :param base_path: the base path of the simulation to use
    :param snap_num: number of the snapshot to use
    :param mass_field: the field name to use as mass indicator for
        groups/halos
    :param radius_field: the field name to use as radius indicator for
        groups/halos
    """
    sim: str
    base_path: str | Path
    snap_num: int
    mass_field: str
    radius_field: str
    data_home: str | Path
    figures_home: str | Path


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
    cur_dir = Path(__file__).parent.resolve()
    root_dir = cur_dir.parent
    data_home = root_dir / "data" if DATA_HOME is None else DATA_HOME
    fig_home = root_dir / "figures" if FIGURES_HOME is None else FIGURES_HOME
    final_config = Config(
        sim,
        f"/virgotng/universe/IllustrisTNG/{sim}/output",
        snap_num=snap,
        mass_field=mass_field,
        radius_field=radius_field,
        data_home=data_home,
        figures_home=fig_home,
    )
    return final_config
