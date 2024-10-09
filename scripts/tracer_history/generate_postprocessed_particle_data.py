import argparse
import logging
import sys
from pathlib import Path

root_dir = Path(__file__).parents[2].resolve()
sys.path.insert(0, str(root_dir / "src"))

from library import scriptparse
from pipelines.tracer_history.generate.postprocess_particle_data import (
    ParentCategoryPipeline,
    TimeOfCrossingPipeline,
)


def main(args: argparse.Namespace) -> None:
    """Postprocess existing archived cool gas data"""
    args.sim = "TNG-Cluster"

    # type flag, subdirectories, and other config changes
    type_flag = f"time_development_{args.what}"

    pipeline_config = scriptparse.startup(
        args,
        "tracer_history",
        type_flag,
        figures_subdirectory=".",
        data_subdirectory=".",
        suppress_sim_name_in_files=False,
    )

    # update pipeline config here
    pipeline_config.update({"zoom_in": args.zoom})

    # select and build pipeline
    if args.what == "crossing-times":
        pipeline = TimeOfCrossingPipeline(**pipeline_config)
    elif args.what == "parent-category":
        pipeline = ParentCategoryPipeline(**pipeline_config)
    else:
        logging.fatal(f"Unrecognized data type {args.what}.")
        sys.exit(1)
    sys.exit(pipeline.run())


if __name__ == "__main__":
    parser = scriptparse.BaseScriptParser(
        prog=f"python {Path(__file__).name}",
        description=(
            "Process existing particle data for the traced cool gas into "
            "other, derived quantites."
        ),
        allowed_sims=["TNG-Cluster"],
    )
    # remove unnecessary args
    parser.remove_argument("sim")
    parser.remove_argument("to_file")
    parser.remove_argument("processes")
    parser.remove_argument("fig_ext")
    parser.remove_argument("no_plots")
    parser.remove_argument("from_file")

    # add new args
    parser.add_argument(
        "what",
        help=(
            "The quantity to generate. This is the gas quantity which will be "
            "traced back in time for those cells that end up in cool "
            "gas at redshift zero. Can only choose from the valid options."
        ),
        choices=["crossing-times", "parent-category"],
    )
    parser.add_argument(
        "-z",
        "--zoom-in",
        help=(
            "When given, must be a number between 0 and 351. This is then the "
            "ID of the only zoom-in region for which the data will be "
            "generated. If left unset, data are created for all zoom-in "
            "regions."
        ),
        dest="zoom",
        type=int,
        metavar="ZOOM-IN ID",
    )

    # parse arguments
    try:
        args_ = parser.parse_args()
        main(args_)
    except KeyboardInterrupt:
        print("\nExecution forcefully stopped.\n")
        sys.exit(1)
