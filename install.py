#!usr/bin/env python3
from pathlib import Path


def install():
    """
    Installs the project by creating the required directory structure.
    """
    # cur dir (equals to root dir)
    root = Path(__file__).parent.resolve()
    # top level dirs
    data_home = root / "data"
    external = root / "external" / "illustris_python"
    figures_home = root / "figures"

    # create directories
    print("Setting up top-level directories.")
    for directory in [data_home, external, figures_home]:
        directory.mkdir()


if __name__ == "__main__":
    install()
