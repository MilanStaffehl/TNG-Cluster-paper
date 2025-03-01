import argparse
import sys
from pathlib import Path

import yaml

root_dir = Path(__file__).parents[2].resolve()
sys.path.insert(0, str(root_dir / "src"))

from library import scriptparse
from pipelines.mass_trends.cool_fracs_with_satellites import (
    CoolGasMassTrendSatellitesPipeline,
)


def main(args: argparse.Namespace) -> None:
    """Create plot of gas mass trends for individual halos"""
    args.sim = "TNG-Cluster"
    # find type flag depending on field name
    if args.field is None:
        type_flag = "clusters_raw_extended"
    else:
        type_flag = f"clusters_{args.field.lower().replace('-', '_')}_extended"

    # base pipeline config dict
    pipeline_config = scriptparse.startup(
        args,
        "mass_trends",
        type_flag,
        figures_subdirectory="./../clusters/extended",
        suppress_sim_name_in_files=True,
    )

    # add custom parameters
    pipeline_config.update(
        {
            "field": args.field.lower(),
            "color_scale": args.color_scale,
            "deviation_scale": None,
            "gas_domain": "halo",
            "forbid_recalculation": args.forbid_recalculation,
            "force_recalculation": args.force_recalculation,
            "use_absolute_mass": True,
        }
    )

    if args.from_file:
        raise NotImplementedError("Not implemented yet.")
    else:
        pipeline = CoolGasMassTrendSatellitesPipeline(**pipeline_config)
    sys.exit(pipeline.run())


if __name__ == "__main__":
    # get list of available fields
    config_file = root_dir / "src/pipelines/mass_trends/plot_config.yaml"
    with open(config_file, "r") as f:
        stream = f.read()
    configuration = yaml.full_load(stream)
    available_fields = list(configuration.keys())

    # construct parser
    parser = scriptparse.BaseScriptParser(
        prog=f"python {Path(__file__).name}",
        description=(
            "Plot mass trends of gas of halos in TNG. This script plots the "
            "same plots as `plot_cool_gas_mass_trends.py`, with the exception "
            "of adding additionally a running mean and another running mean "
            "only for cool gas in satellite galaxies. Additionally, it always "
            "plots both plots for gas mass and gas fraction."
        ),
    )
    parser.remove_argument("sim")
    parser.remove_argument("processes")
    parser.add_argument(
        "--field",
        help=(
            "The field to use for the color data. Must be one of the "
            "supported fields."
        ),
        dest="field",
        choices=available_fields,
        required=True,
    )
    parser.add_argument(
        "--color-scale",
        help=(
            "The normalisation for the color data. If not explicitly set, "
            "the default set in the config is used."
        ),
        dest="color_scale",
        choices=["log", "linear"],
        default=None,
    )
    exclusive_group = parser.add_mutually_exclusive_group(required=False)
    exclusive_group.add_argument(
        "-xr",
        "--forbid-recalculation",
        help=(
            "Forbid the recalculation of gas fractions and the color data. If "
            "color data is not available on file, the pipeline will fail. If "
            "the base data (cool gas fractions/masses) are not available on "
            "file, they will be loaded from the radial profile data files."
        ),
        dest="forbid_recalculation",
        action="store_true",
    )
    exclusive_group.add_argument(
        "-fr",
        "--force-recalculation",
        help=(
            "Force the recalculation of the color data. Gas fraction will "
            "be read from file if available, but if it is not found will be "
            "recalculated as well. If the gas domain is set to the virial "
            "radius, missing gas fraction data or radial density profile data "
            "will lead to an exception, as recalculating gas fraction and "
            "mass from simulation data directly is not currently implemented."
        ),
        dest="force_recalculation",
        action="store_true",
    )

    # parse arguments
    try:
        args = parser.parse_args()
        main(args)
    except KeyboardInterrupt:
        print("Execution forcefully stopped.")
        sys.exit(1)
