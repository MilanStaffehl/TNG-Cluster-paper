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
from pipelines.history.generate.generate_data import (
    FindTracedParticleIDsInSnapshot,
    GenerateTNGClusterTracerIDsAtRedshiftZero,
)


def main(args: argparse.Namespace) -> None:
    """Generate tracer data"""
    # base pipeline config dict
    pipeline_config = scriptparse.startup(
        args,
        "history",
        "tracer_data",
    )

    match args.runtype:
        case "redshift-zero":
            pipeline = GenerateTNGClusterTracerIDsAtRedshiftZero
        case "get-indices":
            if args.snap_num is None:
                logging.fatal(
                    "Must specify snap num for run-type `get-indices`."
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
            "The type of pipeline to run. Options are `redshift-zero` to "
            "find the tracer IDs of tracers in cool gas at redshift zero."
        ),
        choices=["redshift-zero", "get-indices"],
        default="redshift-zero",
        metavar="RUNTYPE",
    )
    parser.add_argument(
        "-n",
        "--snap-num",
        help=(
            "The snapshot to process. Has no effect when using run type "
            "`redshift-zero`"
        ),
        dest="snap_num",
        type=int,
    )

    # parse arguments
    try:
        args = parser.parse_args()
        main(args)
    except KeyboardInterrupt:
        print("\nExecution forcefully stopped.\n")
        sys.exit(1)
