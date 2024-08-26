import argparse
import logging
import sys
from pathlib import Path

root_dir = Path(__file__).parents[2].resolve()
sys.path.insert(0, str(root_dir / "src"))

from library import scriptparse
from pipelines.tracer_history.generate.particle_data import (
    TraceDensityPipeline,
    TraceDistancePipeline,
    TraceMassPipeline,
    TraceTemperaturePipeline,
)
from pipelines.tracer_history.simple_quantities import (
    PlotSimpleQuantityWithTimePipeline,
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
        figures_subdirectory=f"./{type_flag}",
        data_subdirectory="TNG_Cluster",
        suppress_sim_name_in_files=False,
    )

    # select pipeline and config
    match args.what:
        case "temperature":
            pipeline_class = TraceTemperaturePipeline
            quantity_label = "Gas temperature [K]"
        case "distance":
            pipeline_class = TraceDistancePipeline
            quantity_label = "Distance from cluster center [ckpc]"
        case "density":
            pipeline_class = TraceDensityPipeline
            quantity_label = r"Gas density [$M_\odot / ckpc^3$]"
        case "mass":
            pipeline_class = TraceMassPipeline
            quantity_label = r"Particle mass [$M_\odot$]"
        case _:
            logging.fatal(f"Unsupported quantity {args.what}.")
            sys.exit(1)

    # select and build pipeline
    if args.from_file:
        pipeline_config.update(
            {
                "quantity": pipeline_class.quantity,
                "quantity_label": quantity_label,
                "color": "dodgerblue",
            }
        )
        pipeline = PlotSimpleQuantityWithTimePipeline(**pipeline_config)
        sys.exit(pipeline.run())
    else:
        # data generation pipeline
        data_pipeline = pipeline_class(
            **pipeline_config,
            unlink=args.unlink,
            force_overwrite=args.force_overwrite,
            zoom_id=args.zoom,
            archive_single=args.archive_single,
        )
        exit_code = data_pipeline.run()
        if args.no_plots or exit_code != 0:
            sys.exit(exit_code)
        # plotting pipeline
        pipeline_config.update(
            {
                "quantity": pipeline_class.quantity,
                "quantity_label": quantity_label,
                "color": "black",
            }
        )
        plot_pipeline = PlotSimpleQuantityWithTimePipeline(**pipeline_config)
        exit_code += plot_pipeline.run()
        sys.exit(exit_code)


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
    parser.remove_argument("to_file")

    # add new args
    parser.add_argument(
        "what",
        help=(
            "The quantity to plot. This is the gas quantity which will be "
            "traced back in time for those cells that end up in cool "
            "gas at redshift zero. Can only choose from the valid options."
        ),
        choices=["temperature", "distance", "density", "mass"],
    )
    parser.add_argument(
        "-u",
        "--unlink",
        help=(
            "Whether to clean up intermediate files after generating data. "
            "Has no effect when --from-file is used."
        ),
        dest="unlink",
        action="store_true",
    )
    parser.add_argument(
        "-fo",
        "--force-overwrite",
        help=(
            "Force overwriting intermediate files. If not set, existing "
            "intermediate files are re-used."
        ),
        dest="force_overwrite",
        action="store_true",
    )
    parser.add_argument(
        "-z",
        "--zoom-in",
        help=(
            "When given, must be a number between 0 and 351. This is then the "
            "ID of the only zoom-in region for which the data will be "
            "generated and plotted. If left unset, data and plots are created "
            "for all zoom-in regions."
        ),
        dest="zoom",
        type=int,
    )
    parser.add_argument(
        "--archive-single",
        help=(
            "Only has an effect when `--zoom-in` is set: when set, the data "
            "created for the single zoom-in will be added to the hdf5 archive "
            "and the intermediate files created will be unlinked, if "
            "`--unlink` is set. If this is not set, and a zoom-id is "
            "specified, the pipeline will only create the intermediate file "
            "and will not attempt to add it to the archive. It must then be "
            "added later, either manually or by running the script again "
            "without the `--zoom-id` and `--force-overwrite` arguments."
        ),
        action="store_true",
        dest="archive_single",
    )

    # parse arguments
    try:
        args_ = parser.parse_args()
        main(args_)
    except KeyboardInterrupt:
        print("\nExecution forcefully stopped.\n")
        sys.exit(1)
