import argparse
import sys
from pathlib import Path

root_dir = Path(__file__).parents[2].resolve()
sys.path.insert(0, str(root_dir / "src"))

from library import scriptparse
from pipelines.tracer_history.star_particle_fraction import (
    PlotStarParticleFractionPipeline,
)


def main(args: argparse.Namespace) -> None:
    """Create plot of gas velocity distribution"""

    type_flag = "star_fraction"
    pipeline_config = scriptparse.startup(
        args,
        "tracer_history",
        type_flag,
    )

    pipeline = PlotStarParticleFractionPipeline(**pipeline_config)
    pipeline.run()


if __name__ == "__main__":
    parser = scriptparse.BaseScriptParser(
        prog=f"python {Path(__file__).name}",
        description=(
            "Plot the fraction of tracers that end up in cool gas at redshift "
            "zero in stars with time."
        ),
    )
    parser.remove_argument("processes")

    # parse arguments
    try:
        args = parser.parse_args()
        main(args)
    except KeyboardInterrupt:
        print("\nExecution forcefully stopped.\n")
        sys.exit(1)
