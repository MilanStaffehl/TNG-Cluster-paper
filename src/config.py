from dataclasses import dataclass
from pathlib import Path


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

    :param base_path: the base path of the simulation to use
    :param snap_num: number of the snapshot to use
    :param mass_field: the field name to use as mass indicator for
        groups/halos
    """

    base_path: str | Path
    snap_num: int
    mass_field: str


def get_config(
    sim: str, snap: int = 99, mass_field: str = "Group_M_Crit200"
) -> Config:
    """
    Return a configuration for the specified simulation.

    :param sim: name of the simulation as used in the simulation file
        directory, e.g. TNG50-3
    :param snap: snapshot number to use, defaults to 99 (z = 0)
    :param mass_field: name of the simulation field that signifies the
        halo mass, defaults to M_crit200
    :return: configuration for this specific
    """
    final_config = Config(
        f"/virgotng/universe/IllustrisTNG/{sim}/output", snap, mass_field
    )
    return final_config
