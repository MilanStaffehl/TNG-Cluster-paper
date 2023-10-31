import argparse
import sys
from pathlib import Path

root_dir = Path(__file__).parents[2].resolve()
sys.path.insert(0, str(root_dir / "src"))

import glob_util
from library.config import config
from pipelines.temperature_distribution import histograms_temperatures as ht


def main(args: argparse.Namespace) -> None:
    """Create histograms of temperature distribution"""
    # sim data
    sim = glob_util.translate_sim_name(args.sim)

    # config
    cfg = config.get_default_config(sim)

    # histogram weights
    if args.use_mass:
        weight_type = "mass"
    else:
        weight_type = "frac"

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
        "temperature_hist",
        cfg,
        type_flag,
        not args.normalize and args.overplot,
        args.figurespath,
        args.datapath,
        subdirectory,
    )

    pipeline_config = {
        "config": cfg,
        "paths": file_data,
        "processes": args.processes,
        "mass_bin_edges": [1e8, 1e9, 1e10, 1e11, 1e12, 1e13, 1e14, 1e15],
        "n_temperature_bins": args.bins,
        "temperature_range": (-4.0, +4.0) if args.normalize else (3., 8.),
        "weights": weight_type,
        "normalize": args.normalize,
        "with_virial_temperatures": not args.normalize and args.overplot,
        "temperature_divisions": (4.5, 5.5) if args.divisions else None,
        "quiet": args.quiet,
        "to_file": args.to_file,
        "no_plots": args.no_plots,
    }
    if args.from_file and not args.combine:
        hist_plotter = ht.FromFilePipeline(**pipeline_config)
    elif args.from_file and args.combine:
        hist_plotter = ht.CombinedPlotsFromFilePipeline(**pipeline_config)
    elif not args.from_file and args.combine:
        hist_plotter = ht.CombinedPlotsPipeline(**pipeline_config)
    else:
        hist_plotter = ht.TemperatureHistogramsPipeline(**pipeline_config)
    hist_plotter.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog=f"python {Path(__file__).name}",
        description="Plot temperature distribution of halos in TNG",
    )
    parser.add_argument(
        "-s",
        "--sim",
        help=(
            "Type of the simulation to use; main sim is TNG300-1, dev sim "
            "is TNG50-3 and test sim is TNG50-4"
        ),
        dest="sim",
        type=str,
        default="MAIN_SIM",
        choices=["MAIN_SIM", "DEV_SIM", "TEST_SIM"],
    )
    parser.add_argument(
        "-p",
        "--processes",
        help=("Use multiprocessing, with the specified number of processes."),
        type=int,
        default=0,
        dest="processes",
        metavar="NUMBER",
    )
    parser.add_argument(
        "-f",
        "--to-file",
        help=(
            "Whether to write the histogram and virial temperature data "
            "calclated to file"
        ),
        dest="to_file",
        action="store_true",
    )
    parser.add_argument(
        "-l",
        "--load-data",
        help=(
            "When given, data is loaded from data files rather than newly "
            "acquired. This only works if data files of the expected name are "
            "present. When used, the flags -p, -f, -q have no effect."
        ),
        dest="from_file",
        action="store_true",
    )
    parser.add_argument(
        "-x",
        "--no-plots",
        help=(
            "Suppresses creation of plots, use to prevent overwriting "
            "existing files."
        ),
        dest="no_plots",
        action="store_true",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        help=(
            "Prevent progress information to be emitted. Has no effect when "
            "multiprocessing is used."
        ),
        dest="quiet",
        action="store_true",
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
    parser.add_argument(
        "-c",
        "--combine",
        help="Combine all mass bins into one plot",
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
    parser.add_argument(
        "--figures-dir",
        help=(
            "The directory path under which to save the figures, if created. "
            "Directories that do not exist will be recursively created. "
            "It is recommended to leave this at the default value unless "
            "the expected directories do not exist."
        ),
        dest="figurespath",
        default=None,
        metavar="DIR PATH",
    )
    parser.add_argument(
        "--data-dir",
        help=(
            "The directory path under which to save the plots, if created. "
            "Directories that do not exist will be recursively created. "
            "When using --load-data, this directory is queried for data. "
            "It is recommended to leave this at the default value unless "
            "the expected directories do not exist and/or data has been saved "
            "somewhere else."
        ),
        dest="datapath",
        default=None,
        metavar="DIR PATH",
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
