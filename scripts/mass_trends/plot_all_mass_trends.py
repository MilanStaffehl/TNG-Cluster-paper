import logging
import logging.config
import sys
from pathlib import Path

import yaml

root_dir = Path(__file__).parents[2].resolve()
sys.path.insert(0, str(root_dir / "src"))

from library import scriptparse
from library.config import config, logging_config
from pipelines.mass_trends.cool_gas_fracs_clusters_ import (
    ClusterCoolGasMassTrendPipeline,
)


def main() -> None:
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
        "no_plots": False,
        "fig_ext": "pdf",
        "color_scale": None,  # use default
        "deviation_scale": None,  # use default
        "forbid_recalculation": True,
        "force_recalculation": False,
    }

    for field in available_fields:
        for domain in ["halo", "central"]:
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
                }
            )

            print(f"Running: {field} on domain {domain}")
            pipeline = ClusterCoolGasMassTrendPipeline(**pipeline_config)
            try:
                pipeline.run()
            except Exception as exc:
                logging.error(f"Encountered exception:\n{exc}")
            print("\n")


if __name__ == "__main__":
    # parse arguments
    try:
        main()
    except KeyboardInterrupt:
        print("Execution forcefully stopped.")
        sys.exit(1)
