import argparse
import logging
import logging.config
import sys
from pathlib import Path

import yaml

root_dir = Path(__file__).parents[2].resolve()
sys.path.insert(0, str(root_dir / "src"))

from library import scriptparse
from library.config import config, logging_config
from pipelines.mass_trends.cool_gas_fracs_clusters import (
    ClusterCoolGasMassTrendPipeline,
)


def main(create_plots: bool) -> None:
    """Create plots of gas mass trends for individual halos"""
    # set up logging
    log_config = logging_config.get_logging_config(logging.INFO)
    logging.config.dictConfig(log_config)
    logging.addLevelName(18, "MEMORY")  # custom level

    # get list of available fields
    config_file = root_dir / "src/pipelines/mass_trends/plot_config.yaml"
    with open(config_file, "r") as f:
        stream = f.read()
    configuration = yaml.full_load(stream)
    available_fields = list(configuration.keys())

    # build pipeline arguments (these are always the same
    pipeline_config = {
        "config": config.get_default_config("TNG-Cluster"),
        "processes": 0,
        "to_file": True,  # we want files to be saved
        "no_plots": create_plots,
        "fig_ext": "pdf",
        "color_scale": None,  # use default
        "deviation_scale": None,  # use default
        "forbid_recalculation": True,
        "force_recalculation": False,
    }

    for field in available_fields:
        for domain in ["halo", "central"]:
            for use_mass in [False, True]:
                # find type flag depending on field name
                type_flag = f"clusters_{field.replace('-', '_')}"
                if domain == "central":
                    type_flag += "_core"

                # subdirectory for figures
                subdir = field.replace("-", "_")
                if domain == "central":
                    subdir += "/core"

                # assemble paths for current setup
                paths = scriptparse._assemble_path_dict(
                    "mass_trends",
                    pipeline_config["config"],
                    type_flag,
                    figures_subdirectory=f"./../clusters/{subdir}",
                    suppress_sim_path_in_names=True,
                )

                # add custom parameters
                pipeline_config.update(
                    {
                        "paths": paths,
                        "field": field,
                        "gas_domain": domain,
                        "use_absolute_mass": use_mass,
                    }
                )

                print(f"Running: {field} on domain {domain}")
                pipeline = ClusterCoolGasMassTrendPipeline(**pipeline_config)
                try:
                    pipeline.run()
                except Exception as exc:
                    logging.error(f"Encountered exception:\n{exc}")
                print("\n")


DESCRIPTION = """Plot mass trends for all selected properties.

Script runs the plotting pipeline for the mass trend figures of cool gas
mass fraction and cool gas mass vs. cluster mass for all properties under
consideration, and for both cool gas mass and cool gas fraction on the
y-axis, and for both the entire two virial radii sphere and only the
central region of the cluster, effectively going through all different
possible combinations of arguments in the `plot_cool_gas_mass_trends.py`
script.

It uses the default configuration for the plots, specified under the
`src/pipelines/mass_trends/plot_config.yaml` file. Data for the cool gas
mass and cool gas mass fraction as well as cluster mass (i.e. values for
the y- and x-axis) must either already exist from previous runs of the
`plot_cool_gas_mass_trends.py` script, or the radial density profiles of
individual clusters and their data must have already been saved to file
using the `radial_profiles/plot_individual_radial_profiles.py` script.
This is necessary as this script explicitly forbids recalculating the
cool gas fraction and mass directly from the simulation.
"""

if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser(
        prog=f"python {Path(__file__).name}",
        description=DESCRIPTION,
    )
    parser.add_argument(
        "-x",
        "--no-plots",
        help=(
            "Suppresses creation of plots. Useful to prevent overwriting "
            "of existing figure files."
        ),
        dest="no_plots",
        action="store_true",
    )
    args_ = parser.parse_args()
    try:
        main(args_.no_plots)
    except KeyboardInterrupt:
        print("Execution forcefully stopped.")
        sys.exit(1)
