import argparse
import logging
import sys
from pathlib import Path

root_dir = Path(__file__).parents[2].resolve()
sys.path.insert(0, str(root_dir / "src"))

import glob_util
from library import scriptparse
from library.config import config
from pipelines.temperature_distribution.histograms_temperatures import (
    CombinedPlotsFromFilePipeline,
    CombinedPlotsPipeline,
    FromFilePipeline,
    PlotGridPipeline,
    TemperatureHistogramsPipeline,
)


def main(args: argparse.Namespace) -> None:
    """Create histograms of temperature distribution"""
    # config
    try:
        cfg = config.get_default_config(args.sim)
    except config.InvalidSimulationNameError:
        logging.fatal(f"Unsupported simulation: {args.sim}")

    # warn of impossible combinations
    if args.grid and args.combine:
        logging.fatal("Combining the -g and -c flags is not possible.")
        sys.exit(1)

    # histogram weights
    if args.use_mass:
        weight_type = "mass"
    else:
        weight_type = "frac"

    # temperature divisions
    if args.divisions:
        if args.normalize:
            temperature_division = (-2, -1)
        else:
            temperature_division = (4.5, 5.5)
    else:
        temperature_division = None

    # subdirectory
    if args.combine:
        subdirectory = "combined"
    elif args.normalize:
        subdirectory = "normalized"
    else:
        subdirectory = "histograms"

    # type flag
    type_flag = weight_type
    if args.combine:
        type_flag = f"combined_{type_flag}"
    if args.normalize:
        type_flag = f"norm_{type_flag}"

    # file paths
    file_data = glob_util.assemble_path_dict(
        "temperature_distribution",
        cfg,
        type_flag,
        True,
        args.figurespath,
        args.datapath,
        subdirectory,
    )

    pipeline_config = {
        "config": cfg,
        "paths": file_data,
        "processes": args.processes,
        "fig_ext": args.extension,
        "mass_bin_edges": [1e8, 1e9, 1e10, 1e11, 1e12, 1e13, 1e14, 1e15],
        "n_temperature_bins": args.bins,
        "temperature_range": (-4.0, +4.0) if args.normalize else (3., 8.),
        "weights": weight_type,
        "normalize": args.normalize,
        "with_virial_temperatures": not args.normalize and args.overplot,
        "temperature_divisions": temperature_division,
        "quiet": args.quiet,
        "to_file": args.to_file,
        "no_plots": args.no_plots,
    }
    if args.from_file and not args.combine and not args.grid:
        hist_plotter = FromFilePipeline(**pipeline_config)
    elif args.from_file and args.combine:
        hist_plotter = CombinedPlotsFromFilePipeline(**pipeline_config)
    elif not args.from_file and args.combine:
        hist_plotter = CombinedPlotsPipeline(**pipeline_config)
    elif args.grid:
        hist_plotter = PlotGridPipeline(**pipeline_config)
    else:
        hist_plotter = TemperatureHistogramsPipeline(**pipeline_config)
    hist_plotter.run()


if __name__ == "__main__":
    parser = scriptparse.BaseScriptParser(
        prog=f"python {Path(__file__).name}",
        description="Plot temperature distribution of halos in TNG",
    )
    parser.add_argument(
        "-o",
        "--no-overplot",
        help="Suppress overplotting of virial temperature regions",
        dest="overplot",
        action="store_false",
    )
    parser.add_argument(
        "-m",
        "--use-mass",
        help="Use gas mass as hist weights instead of gas mass fraction",
        dest="use_mass",
        action="store_true",
    )
    parser.add_argument(
        "-n",
        "--normalize-temperatures",
        help="Normalize temperatures to virial temperature",
        dest="normalize",
        action="store_true",
    )
    exclusive_group = parser.add_mutually_exclusive_group(required=False)
    exclusive_group.add_argument(
        "-g",
        "--grid",
        help=(
            "Plot all plots into one figure in a grid. This is only possible "
            "if the histogram data already exists as it must be loaded. Not "
            "compatible with -c."
        ),
        dest="grid",
        action="store_true",
    )
    exclusive_group.add_argument(
        "-c",
        "--combine",
        help="Combine all mass bins into one plot. Not compatible with -g.",
        dest="combine",
        action="store_true",
    )
    parser.add_argument(
        "-d",
        "--show-divisions",
        help="Add vertical lines to plots to show temperature regimes",
        dest="divisions",
        action="store_true",
    )
    parser.add_argument(
        "-b",
        "--bins",
        help="The number of temperature bins, defaults to 50",
        dest="bins",
        type=int,
        default=50,
        metavar="NUMBER",
    )

    # parse arguments
    try:
        args = parser.parse_args()
        main(args)
    except KeyboardInterrupt:
        print(
            "Execution forcefully stopped. Some subprocesses might still be "
            "running and need to be killed manually if multiprocessing was "
            "used."
        )
        sys.exit(1)
