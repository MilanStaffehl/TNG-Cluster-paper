#!usr/bin/env python3
from pathlib import Path


def install():
    """
    Installs the project by creating the required directory structure.
    """
    root_dir = Path(__file__).parent.resolve()
    with open(root_dir / "config.yaml", "r") as config_file:
        lines = config_file.readlines()
    for line in lines:
        line.lstrip()
        if line.startswith("data_home"):
            data_home = Path(line.removeprefix("data_home: ")).expanduser()
        elif line.startswith("figures_home"):
            figures_home = Path(line.removeprefix("figures_home: ")
                                ).expanduser()

    external = root_dir / "external"

    # create directories
    print("Setting up top-level directories.")
    for directory in [data_home, external, figures_home]:
        if not directory.exists():
            print(f"Creating missing directory:{str(directory)}")
            directory.mkdir(parents=True)


if __name__ == "__main__":
    install()
