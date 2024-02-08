#!usr/bin/env python3
import sys
from pathlib import Path


def install():
    """
    Installs the project by creating the required directory structure.
    """
    root_dir = Path(__file__).parent.resolve()
    with open(root_dir / "config.yaml", "r") as config_file:
        lines = config_file.readlines()
    data_home = None
    figures_home = None
    for line in lines:
        line = line.lstrip()
        line = line.rstrip("\n")
        if line.startswith("data_home"):
            data_home = Path(line.removeprefix("data_home: ")).expanduser()
        elif line.startswith("figures_home"):
            figures_home = Path(line.removeprefix("figures_home: ")
                                ).expanduser()

    external = root_dir / "external"

    if not all([data_home, figures_home]):
        print("Could not parse config file, not all paths were found!")
        sys.exit(1)

    # create directories
    for directory in [data_home, external, figures_home]:
        if not directory.exists():
            print(f"Creating missing directory:{str(directory)}")
            directory.mkdir(parents=True)


if __name__ == "__main__":
    install()
