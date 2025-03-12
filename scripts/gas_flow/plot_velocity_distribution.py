import argparse
import sys
from pathlib import Path

root_dir = Path(__file__).parents[2].resolve()
sys.path.insert(0, str(root_dir / "src"))

from library import scriptparse
from pipelines.gas_flow.binned_by_mass import (
    MassBinnedVelocityDistributionCombined,
    MassBinnedVelocityDistributionFromFilePipeline,
    MassBinnedVelocityDistributionPipeline,
)


def main(args: argparse.Namespace) -> None:
    """Create plot of gas velocity distribution"""
    type_flag = args.regime
    if args.core_only:
        type_flag += "_core"

    pipeline_config = scriptparse.startup(
        args,
        "gas_flow",
        type_flag,
        suppress_sim_name_in_files=True,
    )

    pipeline_config.update(
        {
            "regime": args.regime,
            "log": args.log,
            "core_only": args.core_only
        }
    )

    if args.from_file and not args.combine:
        pipeline = MassBinnedVelocityDistributionFromFilePipeline(
            **pipeline_config
        )
    elif args.combine:
        pipeline = MassBinnedVelocityDistributionCombined(**pipeline_config)
    else:
        pipeline = MassBinnedVelocityDistributionPipeline(**pipeline_config)
    sys.exit(pipeline.run())


DESCRIPTION = """Plot velocity distribution of gas in clusters.

Script creates eight panels, each showing the distribution histogram of
the radial velocity of all gas particles in cluster of TNG300-1 and
TNG-Cluster belonging to the temperature regime chosen by the `--regime`
argument. Each panel shows only distributions of clusters in a specific
mass range, with the first seven panels each spanning 0.2 dex in mass,
while the eighth panel contains all clusters together.

The panels additionally include a line for the mean virial temperature
of the mass bin and a label displaying the fraction of gas that has
positive and negative velocity respectively. The distributions of the
individual clusters are colored by their redshift zero mass, and an
overall mean and median distribution over all clusters of a mass bin is
added as well.
"""

if __name__ == "__main__":
    parser = scriptparse.BaseScriptParser(
        prog=f"python {Path(__file__).name}",
        description=DESCRIPTION,
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
    parser.add_argument(
        "-cc",
        "--core-only",
        help=(
            "Plot the velocity distribution only for the cluster core "
            "(within 5% of R_200c)."
        ),
        dest="core_only",
        action="store_true",
    )
    parser.add_argument(
        "--log",
        help="Plot gas mass (y-axis) in log-scale.",
        dest="log",
        action="store_true",
    )
    parser.add_argument(
        "-c",
        "--combine",
        help="Combine the eight mass bin panels into one plot.",
        dest="combine",
        action="store_true",
    )

    # parse arguments
    try:
        args = parser.parse_args()
        main(args)
    except KeyboardInterrupt:
        print("\nExecution forcefully stopped.\n")
        sys.exit(1)
