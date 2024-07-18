import argparse
import logging
import sys
from pathlib import Path

root_dir = Path(__file__).parents[2].resolve()
sys.path.insert(0, str(root_dir / "src"))

from library import scriptparse
from pipelines.tracer_history.simple_quantities import (
    TraceDistancePipeline,
    TraceTemperaturePipeline,
)


def main(args: argparse.Namespace) -> None:
    """Plot how different gas quantities evolve with time"""
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

    # select pipelline and config
    match args.what:
        case "temperature":
            pipeline_class = TraceTemperaturePipeline
            quantity_label = "Temperature [K]"
        case "distance":
            pipeline_class = TraceDistancePipeline
            quantity_label = "Distance from cluster center [ckpc]"
        case _:
            logging.fatal(f"Unsupported quantity {args.what}.")
            sys.exit(1)

    pipeline_config.update(
        {
            "quantity_label": quantity_label,
            "color": args.color,
            "make_ridgeline": False
        }
    )

    # select and build pipeline
    if args.from_file:
        raise NotImplementedError("Loading from file is not implemented.")
    else:
        pipeline = pipeline_class(**pipeline_config)
    sys.exit(pipeline.run())


if __name__ == "__main__":
    parser = scriptparse.BaseScriptParser(
        prog=f"python {Path(__file__).name}",
        description=(
            "Plot the development of a gas quantity for all those gas "
            "cells that end up in cool gas at redshift zero."
        ),
        allowed_sims=["TNG-Cluster"],
    )
    # remove unnecessary args
    parser.remove_argument("processes")
    parser.remove_argument("sim")

    # add new args
    parser.add_argument(
        "what",
        help=(
            "The quantity to plot. This is the gas quantity which will be "
            "traced back in time for thise gas cells that end up in cool "
            "gas at redshift zero. Can onlz choose from the valid options."
        ),
        choices=["temperature", "distance"],
    )
    parser.add_argument(
        "-c",
        "--color",
        help="Color for the faint lines of individual clusters",
        dest="color",
        type=str,
        default="dodgerblue",
    )

    # parse arguments
    try:
        args_ = parser.parse_args()
        main(args_)
    except KeyboardInterrupt:
        print("\nExecution forcefully stopped.\n")
        sys.exit(1)
