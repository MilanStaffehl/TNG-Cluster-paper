#!usr/bin/env python3
import sys
from pathlib import Path

root_dir = Path(__file__).parent.resolve()


def install():
    """
    Install the project by creating a config file and other directories.
    """
    config_file = root_dir / "config.yaml"
    if not config_file.exists():
        create_config_file()
    create_dirs()


def create_config_file():
    """
    Creates a config.yaml file with the default content.
    """
    # yapf: disable
    default_config = (
        "paths:\n"
        "  base_paths:\n"
        "    TNG300-1: /virgotng/universe/IllustrisTNG/TNG300-1/output\n"
        "    TNG50-4: /virgotng/universe/IllustrisTNG/TNG50-4/output\n"
        "    TNG50-3: /virgotng/universe/IllustrisTNG/TNG50-3/output\n"
        "    TNG-Cluster: /virgotng/mpia/TNG-Cluster/TNG-Cluster/output\n"
        "  data_home: default\n"
        "  figures_home: default\n"
        "  cool_gas_history_archive:\n"
        "    TNG300-1: default\n"
        "    TNG-Cluster: default\n"
    )
    # yapf: enable
    with open(root_dir / "config.yaml", "w") as config_file:
        config_file.write(default_config)
    print("Created a default config file.")


def create_dirs():
    """
    Create the required directory structure.
    """
    with open(root_dir / "config.yaml", "r") as config_file:
        lines = config_file.readlines()
    data_home = None
    figures_home = None
    for line in lines:
        line = line.lstrip()
        line = line.rstrip("\n")
        if line.startswith("data_home"):
            data_home = line.removeprefix("data_home: ")
        elif line.startswith("figures_home"):
            figures_home = line.removeprefix("figures_home: ")

    if not all([data_home, figures_home]):
        print("Could not parse config file, not all paths were found!")
        sys.exit(1)

    external = root_dir / "external"
    if data_home == "default":
        data_home = root_dir / "data"
    else:
        data_home = Path(data_home).resolve()
    if figures_home == "default":
        figures_home = root_dir / "figures"
    else:
        figures_home = Path(figures_home).resolve()

    # create directories
    for directory in [data_home, external, figures_home]:
        if not directory.exists():
            print(f"Creating missing directory:{str(directory)}")
            directory.mkdir(parents=True)


if __name__ == "__main__":
    install()
