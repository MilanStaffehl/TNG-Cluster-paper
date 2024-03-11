import argparse
import sys
from pathlib import Path

root_dir = Path(__file__).parents[2].resolve()
sys.path.insert(0, str(root_dir / "src"))

from library import scriptparse
from pipelines.gas_flow.binned_by_mass import (
    MassBinnedVelocityDistributionFromFilePipeline,
    MassBinnedVelocityDistributionPipeline,
)


def main(args: argparse.Namespace) -> None:
    """Create plot of gas velocity distribution"""
    pipeline_config = scriptparse.startup(
        args,
        "gas_flow",
        args.regime,
    )

    pipeline_config.update({"regime": args.regime})
    if args.from_file:
        pipeline = MassBinnedVelocityDistributionFromFilePipeline(
            **pipeline_config
        )
    else:
        pipeline = MassBinnedVelocityDistributionPipeline(**pipeline_config)
    pipeline.run()


if __name__ == "__main__":
    parser = scriptparse.BaseScriptParser(
        prog=f"python {Path(__file__).name}",
        description="Plot velocity distribution of gas in clusters.",
    )
    parser.remove_argument("sim")
    parser.remove_argument("processes")
    parser.add_argument(
        "--regime",
        choices=["cool", "warm", "hot"],
        help=(
            "The temperature regime for which to plot the velocity "
            "distribution."
        ),
        dest="regime",
        default="cool",
    )

    # parse arguments
    try:
        args = parser.parse_args()
        main(args)
    except KeyboardInterrupt:
        print("\nExecution forcefully stopped.\n")
        sys.exit(1)
