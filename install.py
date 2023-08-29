#!usr/bin/env python3
from pathlib import Path

# vars
SIMUATIONS = ["TNG300-1", "TNG50-3", "TNG50-4"]
MILESTONES = 3


def install():
    """
    Installs the project by creating the required directory structure.
    """
    # cur dir (equals to root dir)
    root = Path(__file__).resolve()
    # top level dirs
    data_home = root / "data"
    external = root / "external"
    figures_home = root / "figures"

    # create directories
    for directory in [data_home, external, figures_home]:
        directory.mkdir()

    # create subdirs
    for i in range(1, MILESTONES + 1, 1):
        milestone_dir = f"{i:03d}"
        (figures_home / milestone_dir).mkdir()
        (data_home / milestone_dir).mkdir()
        for simulation in SIMUATIONS:
            sim_dir = simulation.replace("-", "_")
            (figures_home / milestone_dir / sim_dir).mkdir()
            (data_home / milestone_dir / sim_dir).mkdir()


if __name__ == "__main__":
    install()
