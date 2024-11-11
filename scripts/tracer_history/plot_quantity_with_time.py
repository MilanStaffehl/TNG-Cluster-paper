import argparse
import logging
import sys
from pathlib import Path

root_dir = Path(__file__).parents[2].resolve()
sys.path.insert(0, str(root_dir / "src"))

from library import scriptparse
from pipelines.tracer_history.generate.complex_particle_data import (
    TraceDistancePipeline,
    TraceParentHaloPipeline,
    TraceParentSubhaloPipeline,
)
from pipelines.tracer_history.generate.simple_particle_data import (
    TraceDensityPipeline,
    TraceMassPipeline,
    TraceTemperaturePipeline,
)
from pipelines.tracer_history.traced_quantities import (
    PlotSimpleQuantitiesForSingleClusters,
    PlotSimpleQuantityWithTimePipeline,
    PlotType,
)


def main(args: argparse.Namespace) -> None:
    """Plot how different gas quantities evolve with time"""
    args.sim = "TNG-Cluster"

    # type flag, subdirectories, and other config changes
    type_flag = f"time_development_{args.what}"

    # figures subdir
    figures_subdir = f"./{type_flag}"
    if args.zoom is not None:
        figures_subdir += "/individuals"

    pipeline_config = scriptparse.startup(
        args,
        "tracer_history",
        type_flag,
        figures_subdirectory=figures_subdir,
        data_subdirectory="TNG_Cluster",
        suppress_sim_name_in_files=False,
    )

    # select pipeline and config
    match args.what:
        case "temperature":
            pipeline_class = TraceTemperaturePipeline
        case "distance":
            pipeline_class = TraceDistancePipeline
        case "density":
            pipeline_class = TraceDensityPipeline
        case "mass":
            pipeline_class = TraceMassPipeline
        case "parent-halo":
            pipeline_class = TraceParentHaloPipeline
        case "parent-subhalo":
            pipeline_class = TraceParentSubhaloPipeline
        case _:
            logging.fatal(f"Unsupported quantity {args.what}.")
            sys.exit(1)

    # unpack plot types
    try:
        plot_types_str = args.plot_types.split(",")
        plot_types = [int(e) for e in plot_types_str]
    except AttributeError:
        plot_types = None

    # select plotting pipeline
    if args.zoom is not None:
        plotting_pipeline = PlotSimpleQuantitiesForSingleClusters
        additional_configs = {
            "quantity": pipeline_class.quantity,
            "zoom_in": args.zoom,
            "part_limit": args.particle_limit,
            "volume_normalize": args.volume_normalize,
            "plot_types": plot_types,
            "split_by": args.split_by,
        }
    else:
        plotting_pipeline = PlotSimpleQuantityWithTimePipeline
        additional_configs = {
            "quantity": pipeline_class.quantity,
            "color": "dodgerblue",
            "normalize": args.normalize,
            "volume_normalize": args.volume_normalize,
            "plot_types": plot_types,
            "split_by": args.split_by,
        }

    # select and build pipeline
    if args.from_file:
        pipeline_config.update(additional_configs)
        pipeline = plotting_pipeline(**pipeline_config)
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
        pipeline_config.update(additional_configs)
        plot_pipeline = plotting_pipeline(**pipeline_config)
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
    parser.remove_argument("sim")
    parser.remove_argument("to_file")

    # add new args
    parser.add_argument(
        "what",
        help=(
            "The quantity to generate, archive and/or plot. This is the gas "
            "quantity which will be traced back in time for those cells that "
            "end up in cool gas at redshift zero. Can only choose from the "
            "valid options."
        ),
        choices=[
            "temperature",
            "distance",
            "density",
            "mass",
            "parent-halo",
            "parent-subhalo",
        ],
    )
    parser.add_argument(
        "-z",
        "--zoom-in",
        help=(
            "When given, must be a number between 0 and 351. This is then the "
            "ID of the only zoom-in region for which the data will be "
            "generated and plotted. If left unset, data and plots are created "
            "for all zoom-in regions. Plots for individual clusters are "
            "different from those for all clusters."
        ),
        dest="zoom",
        type=int,
        metavar="ZOOM-IN ID",
    )
    parser.add_argument(
        "-u",
        "--unlink",
        help=(
            "Whether to clean up intermediate files after generating data. "
            "Has no effect when --from-file is used. Has no effect when using "
            "`--load-data`."
        ),
        dest="unlink",
        action="store_true",
    )
    parser.add_argument(
        "-fo",
        "--force-overwrite",
        help=(
            "Force overwriting intermediate files. If not set, existing "
            "intermediate files are re-used. Has no effect when using "
            "`--load-data`."
        ),
        dest="force_overwrite",
        action="store_true",
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
            "without the `--zoom-id` and `--force-overwrite` arguments. Has "
            "no effect when using `--load-data`."
        ),
        action="store_true",
        dest="archive_single",
    )
    parser.add_argument(
        "-pt",
        "--plot-types",
        help=(
            f"Comma-separated list of plot types to create. When not set, all "
            f"available plot types are plotted. Must be one of the following "
            f"valid plot types: "
            f"{', '.join([f'{p.value}: {p.name.lower()}' for p in PlotType])}."
        ),
        dest="plot_types",
        metavar="LIST",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--limit-particles",
        help=(
            "Only has an effect when `--zoom-in` is set: Limit the number of "
            "particles to plot to the given number."
        ),
        type=int,
        dest="particle_limit",
        metavar="N",
    )
    parser.add_argument(
        "-n",
        "--normalize",
        help=(
            "Normalize the quantity to a related characteristic cluster "
            "property before plotting, for example normalize distances to the "
            "cluster virial radius or temperatures to its virial temperature. "
            "Has no effect when using `--zoom-in`. Some quantities may not "
            "support this option, in which case this flag has no effect. Only "
            "works for plot types `global2dhist` and `globalridgeline`."
        ),
        action="store_true",
    )
    parser.add_argument(
        "-vn",
        "--volume-normalize",
        help=(
            "Normalize distance 2D histogram by shell volume instead of "
            "normalizing to sum to one. Only works for distances, and is "
            "automatically disabled for other quantities. Has no effect when "
            "using `--zoom-in`."
        ),
        dest="volume_normalize",
        action="store_true",
    )
    parser.add_argument(
        "--split-by",
        help=(
            "Split the plots of quantity vs. distance by the specified "
            "category. When using `--zoom-in`, this also colors the lines of "
            "the individual tracer track plot by that category."
        ),
        dest="split_by",
        choices=[
            "parent-category",
            "parent-category-at-zero",
            "bound-state",
            "bound-state-at-zero",
            "distance-at-zero",
        ],
    )

    # parse arguments
    try:
        args_ = parser.parse_args()
        main(args_)
    except KeyboardInterrupt:
        print("\nExecution forcefully stopped.\n")
        sys.exit(1)
