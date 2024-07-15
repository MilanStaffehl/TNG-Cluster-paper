"""
Script to run the plotting pipeline for earliest snapshot of clusters.
"""
import argparse
import logging
import sys
from pathlib import Path

root_dir = Path(__file__).parents[2].resolve()
sys.path.insert(0, str(root_dir / "src"))

from library import scriptparse
from pipelines.tracer_history.find_mainleaf_progenitor import (
    CombinedEarliestSnapshotDistributionPipeline,
    PlotEarliestSnapNumDistributionPipeline300,
    PlotEarliestSnapNumDistributionPipelineCluster,
)


def main(args: argparse.Namespace) -> None:
    """Create plot of snapshot distribution"""

    type_flag = "earliest_snapshot"
    if args.combine:
        data_subdir = None
        fig_subdir = "./.."
    else:
        data_subdir = args.sim.replace("-", "_")
        fig_subdir = None
    pipeline_config = scriptparse.startup(
        args,
        "tracer_history",
        type_flag,
        figures_subdirectory=fig_subdir,
        data_subdirectory=data_subdir,
        suppress_sim_name_in_files=args.combine,  # suppress when combined
    )

    if args.combine:
        pipeline = CombinedEarliestSnapshotDistributionPipeline(
            **pipeline_config
        )
    elif args.sim == "TNG-Cluster":
        pipeline = PlotEarliestSnapNumDistributionPipelineCluster(
            **pipeline_config
        )
    elif args.sim.startswith("TNG300"):
        pipeline = PlotEarliestSnapNumDistributionPipeline300(
            **pipeline_config
        )
    else:
        logging.error(f"Unsupported simulation: {args.sim}.")
        sys.exit(1)
    sys.exit(pipeline.run())


if __name__ == "__main__":
    parser = scriptparse.BaseScriptParser(
        prog=f"python {Path(__file__).name}",
        description=(
            "Plot the distribution of snapshots until which the main "
            "progenitor of a clusters primary subhalo can be traced back to."
        ),
        allowed_sims=["TNG-Cluster", "TNG300-1"],
    )
    parser.remove_argument("processes")
    parser.remove_argument("from_file")

    parser.add_argument(
        "-c",
        "--combine",
        help=(
            "When set, ignores the -s flag and plots combined distribution "
            "for both TNG-Cluser and TNG300-1."
        ),
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
