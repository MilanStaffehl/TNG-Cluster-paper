import argparse
import sys
from pathlib import Path

import _paths  # noqa: F401  # inserts the src directory into path
from library.config import config
from pipelines.temperature_distribution import histograms_temperatures as ht


def main(args: argparse.Namespace) -> None:
    """Create histograms of temperature distribution"""
    # sim data
    if args.sim == "TEST_SIM":
        sim = "TNG50-4"
    elif args.sim == "DEV_SIM":
        sim = "TNG50-3"
    elif args.sim == "MAIN_SIM":
        sim = "TNG300-1"
    else:
        raise ValueError(f"Unknown simulation type {args.sim}.")

    # config
    cfg = config.get_default_config(sim)

    # histogram weights
    if args.use_mass:
        weight_type = "mass"
    else:
        weight_type = "frac"

    # paths
    base_dir = cfg.figures_home / f"temperature_distribution/{cfg.sim_path}"
    if args.normalize:
        figure_path = base_dir / "normalized"
        figure_stem = f"temperature_hist_norm_{weight_type}_{cfg.sim_path}"
        data_stem = f"temperature_hists_norm_{weight_type}_{cfg.sim_path}"
    else:
        figure_path = base_dir / "histograms"
        figure_stem = f"temperature_hist_{weight_type}_{cfg.sim_path}"
        data_stem = f"temperature_hist_{weight_type}_{cfg.sim_path}"

    if args.figurespath:
        new_path = Path(args.figurespath)
        if new_path.exists() and new_path.is_dir():
            figure_path = new_path
        else:
            print(
                f"WARNING: Given figures path is invalid: {str(new_path)}."
                f"Using fallback path {str(figure_path)} instead."
            )

    data_path = cfg.data_home / "temperature_distribution"
    if args.datapath:
        new_path = Path(args.datapath)
        if new_path.exists() and new_path.is_dir():
            data_path = new_path
        else:
            print(
                f"WARNING: Given data path is invalid: {str(new_path)}."
                f"Using fallback path {str(data_path)} instead."
            )

    file_data = {
        "figures_dir": figure_path,
        "data_dir": data_path,
        "figures_file_stem": figure_stem,
        "data_file_stem": data_stem,
        "virial_temp_file_stem": f"virial_temperatures_{cfg.sim_path}"
    }

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
