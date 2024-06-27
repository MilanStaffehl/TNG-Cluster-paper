import argparse
import sys
from pathlib import Path

root_dir = Path(__file__).parents[2].resolve()
sys.path.insert(0, str(root_dir / "src"))

from library import scriptparse
from pipelines.history.test_plot import (
    FollowParticlesFromFilePipeline,
    FollowParticlesPipeline,
)


def main(args: argparse.Namespace) -> None:
    """Create plot of gas velocity distribution"""
    args.sim = "TNG-Cluster"

    type_flag = "test_plot"

    pipeline_config = scriptparse.startup(
        args,
        "history",
        type_flag,
    )

    pipeline_config.update(
        {
            "max_tracers": args.max_tracers, "plot_lines": args.plot_lines
        }
    )

    if args.from_file:
        pipeline = FollowParticlesFromFilePipeline(**pipeline_config)
    else:
        pipeline = FollowParticlesPipeline(**pipeline_config)
    pipeline.run()


if __name__ == "__main__":
    parser = scriptparse.BaseScriptParser(
        prog=f"python {Path(__file__).name}",
        description="Plot the positions of some particles with time.",
    )
    parser.remove_argument("sim")
    parser.remove_argument("processes")
    parser.add_argument(
        "-m",
        "--max-tracers",
        help="Maximum number of tracers to follow. Defaults to 5.",
        dest="max_tracers",
        type=int,
        default=5,
    )
    parser.add_argument(
        "-pl",
        "--plot-lines",
        help="Plot positions connected by lines instead of as points.",
        action="store_true",
        dest="plot_lines",
    )

    # parse arguments
    try:
        args = parser.parse_args()
        main(args)
    except KeyboardInterrupt:
        print("\nExecution forcefully stopped.\n")
        sys.exit(1)
