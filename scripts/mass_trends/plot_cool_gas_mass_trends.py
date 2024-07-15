import argparse
import sys
from pathlib import Path

import yaml

root_dir = Path(__file__).parents[2].resolve()
sys.path.insert(0, str(root_dir / "src"))

from library import scriptparse
from pipelines.mass_trends.cool_gas_fracs_clusters_ import (
    ClusterCoolGasFromFilePipeline,
    ClusterCoolGasMassTrendPipeline,
)


def main(args: argparse.Namespace) -> None:
    """Create plot of gas mass trends for individual halos"""
    args.sim = "TNG-Cluster"
    # find type flag depending on field name
    if args.field is None:
        type_flag = "clusters_raw"
    else:
        type_flag = f"clusters_{args.field.lower().replace('-', '_')}"
    if args.gas_domain == "central":
        type_flag += "_core"

    # subdirectory for figures
    subdir = args.field.replace("-", "_")
    if args.gas_domain == "central":
        subdir += "/core"

    # base pipeline config dict
    pipeline_config = scriptparse.startup(
        args,
        "mass_trends",
        type_flag,
        figures_subdirectory=f"./../clusters/{subdir}",
        suppress_sim_name_in_files=True,
    )

    # add custom parameters
    pipeline_config.update(
        {
            "field": args.field.lower(),
            "color_scale": args.color_scale,
            "deviation_scale": args.deviation_scale,
            "gas_domain": args.gas_domain,
            "forbid_recalculation": args.forbid_recalculation,
            "force_recalculation": args.force_recalculation,
        }
    )

    if args.from_file:
        pipeline = ClusterCoolGasFromFilePipeline(**pipeline_config)
    else:
        pipeline = ClusterCoolGasMassTrendPipeline(**pipeline_config)
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
        description="Plot mass trends of gas of halos in TNG",
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
    parser.add_argument(
        "--deviation-scale",
        help=(
            "The scale of the deviation plot, which can either be log or "
            "linear. The colorbar will extent to both sides of unity (or zero "
            "for logarithmic scaling) in the corresponding scale. If not set, "
            "the default set in the config is used."
        ),
        dest="deviation_scale",
        choices=["log", "linear"],
        default=None,
    )
    parser.add_argument(
        "--gas-domain",
        help=(
            "The domain for the gas fraction. The y-axis can either show the "
            "gas fraction of the entire cluster out to 2 virial radii (halo) "
            "or only the in the core region, meaning within 5%% of the virial "
            "radius (central)."
        ),
        dest="gas_domain",
        choices=["halo", "central"],
        default="halo",
    )
    exclusive_group = parser.add_mutually_exclusive_group(required=False)
    exclusive_group.add_argument(
        "-xr",
        "--forbid-recalculation",
        help=(
            "Forbid the recalculation of gas fractions and the color data. If "
            "either is not available on file, the pipeline will fail."
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
            "recalculated as well."
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
