"""
Script to generate tracer data required for other scripts.
"""
import argparse
import logging
import sys
from pathlib import Path

root_dir = Path(__file__).parents[2].resolve()
sys.path.insert(0, str(root_dir / "src"))

from library import scriptparse
from pipelines.tracer_history.generate.generate_data import (
    FindTracedParticleIDsInSnapshot,
    GenerateTNGClusterTracerIDsAtRedshiftZero,
)


def main(args: argparse.Namespace) -> None:
    """Generate tracer data"""
    # base pipeline config dict
    pipeline_config = scriptparse.startup(
        args,
        "tracer_history",
        "tracer_data",
    )
    # update config snap num
    pipeline_config["config"].snap_num = args.start_snapshot

    match args.runtype:
        case "identify":
            pipeline = GenerateTNGClusterTracerIDsAtRedshiftZero
        case "trace-back":
            if args.snap_num is None:
                logging.fatal(
                    "Must specify snap num for run-type `trace-back`."
                )
                sys.exit(2)
            pipeline = FindTracedParticleIDsInSnapshot
            pipeline_config.update({"snap_num": args.snap_num})
        case _:
            logging.fatal(f"Unknown run type: {args.runtype}")
            sys.exit(1)

    pipe = pipeline(**pipeline_config)
    pipe.run()


if __name__ == "__main__":
    parser = scriptparse.BaseScriptParser(
        prog=f"python {Path(__file__).name}",
        description="Generate tracer data for scripts of cool gas origin.",
    )
    parser.remove_argument("to_file")
    parser.remove_argument("from_file")
    parser.remove_argument("no_plots")
    parser.remove_argument("fig_ext")
    parser.add_argument(
        "runtype",
        help=(
            "The type of pipeline to run. Options are `identify` to "
            "find the tracer IDs of tracers in cool gas at redshift zero,"
            "or `trace-back` to save to file the indices of particles in a"
            "specified snapshot that end up in cool gas at redshift zero."
        ),
        choices=["identify", "trace-back"],
        default="identify",
        metavar="RUNTYPE",
    )
    parser.add_argument(
        "-n",
        "--snap-num",
        help=(
            "The snapshot to process. Has no effect when using run type "
            "`identify`."
        ),
        dest="snap_num",
        type=int,
    )
    parser.add_argument(
        "-sn",
        "--start-snapshot",
        help=(
            "The snapshot from which to start the analysis. This is the "
            "snapshot in which the `identify` run type will look for cool "
            "gas and identify its tracers, and save the IDs of these tracers. "
            "In all other modes, analysis will start from this snapshot and "
            "go back in time. Defaults to snapshot 99 (redshift zero)."
        ),
        dest="start_snapshot",
        type=int,
        default=99,
    )

    # parse arguments
    try:
        args = parser.parse_args()
        main(args)
    except KeyboardInterrupt:
        print("\nExecution forcefully stopped.\n")
        sys.exit(1)
