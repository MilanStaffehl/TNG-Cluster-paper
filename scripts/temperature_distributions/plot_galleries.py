import argparse
import sys
from pathlib import Path

# import the helper scripts
cur_dir = Path(__file__).parent.resolve()
sys.path.append(str(cur_dir.parent.parent / "pipelines"))

from config import config
from temperature_distribution import histogram_galleries


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

    # paths
    base_dir = cfg.figures_home / f"temperature_distribution/{cfg.sim_path}"
    if args.normalize:
        figure_path = base_dir / "galleries"
        figure_stem = f"temperature_gallery_norm_{cfg.sim_path}"
        data_stem = f"temperature_gallery_norm_{cfg.sim_path}"
    else:
        figure_path = base_dir / "galleries"
        figure_stem = f"temperature_gallery_{cfg.sim_path}"
        data_stem = f"temperature_gallery_{cfg.sim_path}"

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
        "virial_temp_file_stem": "",
    }

    pipeline_config = {
        "config": cfg,
        "paths": file_data,
        "plots_per_bin": args.plots_per_bin,
        "mass_bin_edges": [1e8, 1e9, 1e10, 1e11, 1e12, 1e13, 1e14, 1e15],
        "n_temperature_bins": args.bins,
        "temperature_range": (3., 8.),
        "normalize": args.normalize,
        "quiet": args.quiet,
        "no_plots": args.no_plots,
        "to_file": args.to_file,
    }
    if args.load_data:
        gallery_plotter = histogram_galleries.FromFilePipeline(**pipeline_config)  # yapf: disable
    else:
        gallery_plotter = histogram_galleries.Pipeline(**pipeline_config)
    gallery_plotter.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog=f"python {Path(__file__).name}",
        description="Plot temperature distribution galleries of halos in TNG",
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
        "--plots-per-bin",
        help=(
            "The number of halos to select and plot per mass bin. Default "
            "is 10"
        ),
        dest="plots_per_bin",
        type=int,
        default=10,
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
        "-n",
        "--normalize-temperatures",
        help="Normalize temperatures to virial temperature",
        dest="normalize",
        action="store_true",
    )
    parser.add_argument(
        "-l",
        "--load-data",
        help=("Load data from file instead of selecting new halos."),
        dest="load_data",
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
        "--plot-dir",
        help=(
            "The directory path under which to save the plots, if created. "
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
        print("Execution forcefully stopped.")
        sys.exit(1)
