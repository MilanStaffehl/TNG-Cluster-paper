import argparse
import sys
from pathlib import Path

root_dir = Path(__file__).parents[2].resolve()
sys.path.insert(0, str(root_dir / "src"))

from library import scriptparse
from pipelines.mass_trends.cool_gas_fracs_clusters import (
    ClusterCoolGasFromFilePipeline,
    ClusterCoolGasMassTrendPipeline,
)


def main(args: argparse.Namespace) -> None:
    """Create plot of gas mass trends for individual halos"""
    args.sim = "TNG-Cluster"
    # find type flag depending on field name
    if args.color_field is None:
        type_flag = "clusters"
    else:
        type_flag = f"clusters_{args.color_field.lower()}"
    if args.core_only:
        type_flag += "_core"

    if args.median_deviation:
        subdir = "median_deviation"
    else:
        subdir = "standard"
    if args.core_only:
        subdir += "/core"
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
            "log": args.log,
            "color_log": args.color_log,
            "color_field": args.color_field,
            "forbid_recalculation": args.forbid_recalculation,
            "core_only": args.core_only,
            "median_deviation": args.median_deviation,
        }
    )

    if args.from_file:
        pipeline = ClusterCoolGasFromFilePipeline(**pipeline_config)
    else:
        pipeline = ClusterCoolGasMassTrendPipeline(**pipeline_config)
    pipeline.run()


if __name__ == "__main__":
    parser = scriptparse.BaseScriptParser(
        prog=f"python {Path(__file__).name}",
        description="Plot mass trends of gas of halos in TNG",
    )
    parser.remove_argument("sim")
    parser.remove_argument("processes")
    parser.add_argument(
        "--log",
        help=(
            "When used, the y-axis (cool gas fraction) will be plotted in "
            "log scale."
        ),
        dest="log",
        action="store_true",
    )
    parser.add_argument(
        "--color-log",
        help=(
            "When used, the given field will be plotted in log scale in "
            "color space. Has no effect when --field is not set or when using "
            "--load-data."
        ),
        dest="color_log",
        action="store_true",
    )
    parser.add_argument(
        "--field",
        help=(
            "The name of the field to color the points by. If not set, "
            "defaults to None which means the points will not be colored."
        ),
        dest="color_field",
        metavar="FIELD",
        default=None,
    )
    parser.add_argument(
        "-r",
        "--forbid-recalculation",
        help=(
            "Forbid recalculation of cool gas fraction and instead load it "
            "from the radial density profile histograms."
        ),
        dest="forbid_recalculation",
        action="store_true",
    )
    parser.add_argument(
        "-cc",
        "--cluster-core",
        help=(
            "Limit the cool gas fraction to only consider gas in the cluster "
            "core (that is within 5%% of the virial radius)."
        ),
        dest="core_only",
        action="store_true",
    )
    parser.add_argument(
        "-d",
        "--median-deviation",
        help=(
            "Plot not the color quantity itself but the deviation of it from "
            "the median in 0.2 dex mass bins. Also adds lines as visual marks "
            "for the mass bins to the plot."
        ),
        dest="median_deviation",
        action="store_true",
    )

    # parse arguments
    try:
        args = parser.parse_args()
        main(args)
    except KeyboardInterrupt:
        print("Execution forcefully stopped.")
        sys.exit(1)
