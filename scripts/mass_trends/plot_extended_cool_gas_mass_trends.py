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
        raise NotImplementedError(
            "Not implemented yet. Please drop the `-l` option and use the "
            "`-xr` option, which will similarly lead to fast execution."
        )
    else:
        pipeline = CoolGasMassTrendSatellitesPipeline(**pipeline_config)
    sys.exit(pipeline.run())


DESCRIPTION = """Plot mass trends of gas of halos in TNG.

This script plots the same plots as `plot_cool_gas_mass_trends.py`, with
the exception of adding additionally a running mean over the data points,
plus a second running mean line for only cool gas in satellite galaxies.
Additionally, it always plots both plots for gas mass and gas fraction.
This script skips the creation of the deviation plots, but in all other
regards, it is identical to `plot_cool_gas_mass_trends.py`; refer to its
description for details.
"""

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
        description=DESCRIPTION,
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
            "the default set in the config under "
            "`src/pipelines/mass_trends/plot_config.yaml` is used."
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
            "Forbid the recalculation of cool gas fractions. Color data is"
            "not affected by this option (i.e. whether color data is loaded "
            "from file or recalculated from simulation data is not controlled "
            "by this option). If the base data (cool gas fractions/masses) "
            "are not available on file, they will be loaded from the radial "
            "profile data files instead. If they do not exist either, script "
            "execution fails."
        ),
        dest="forbid_recalculation",
        action="store_true",
    )
    exclusive_group.add_argument(
        "-fr",
        "--force-recalculation",
        help=(
            "Force the recalculation of cool gas fraction data. Cannot be "
            "used together with `--gas-domain vr`, as recalculating gas "
            "fraction and mass from simulation data directly is not currently "
            "implemented. Has no effect on the color data specified with the "
            "`--field` option, which is independently loaded or recalculated."
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
