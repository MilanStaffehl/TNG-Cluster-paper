import argparse
import sys
from pathlib import Path

root_dir = Path(__file__).parents[2].resolve()
sys.path.insert(0, str(root_dir / "src"))

from library import scriptparse
from pipelines.mass_trends.temperatures_clusters import (
    ClusterCoolGasFromFilePipeline,
    ClusterCoolGasMassTrendPipeline,
)


def main(args: argparse.Namespace) -> None:
    """Create plot of gas mass trends for individual halos"""
    # find type flag depending on field name
    if args.color_field is None:
        type_flag = ""
    else:
        type_flag = args.color_field.lower()

    pipeline_config = scriptparse.startup(
        args,
        "mass_trends",
        type_flag,
    )

    # add custom parameters
    pipeline_config.update(
        {
            "log": args.log,
            "color_field": args.color_field,
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
            "When used, the given field will be plotted in log scale in "
            "color space. Has no effect when --field is not set."
        ),
        dest="log",
        action="store_true",
    )
    parser.add_argument(
        "--field",
        help=(
            "The name of the field to color the points by. If not set, "
            "defaults to None which means the points will not be colored."
            "If set, it must be a valid TNG group catalogue field name."
        ),
        dest="color_field",
        metavar="GROUPCAT_FIELD",
        default=None,
    )

    # parse arguments
    try:
        args = parser.parse_args()
        main(args)
    except KeyboardInterrupt:
        print(
            "Execution forcefully stopped. Some subprocesses might still be "
            "running and need to be killed manually if multiprocessing was "
            "used."
        )
        sys.exit(1)
