import argparse
import sys
from pathlib import Path

# import the helper scripts
cur_dir = Path(__file__).parent.resolve()
sys.path.append(str(cur_dir.parent.parent / "pipelines"))

from temperature_distribution import histograms_by_fraction


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

    # histogram weights
    # if args.total_mass:
    #     weight_type = "mass"
    # else:
    #     weight_type = "frac"

    pipeline_config = {
        "simulation": sim,
        "processes": args.processes,
        "mass_bin_edges": [1e8, 1e9, 1e10, 1e11, 1e12, 1e13, 1e14, 1e15],
        "n_temperature_bins": args.bins,
        "temperature_range": (3., 8.),
        "with_virial_temperatures": args.overplot,
        "quiet": args.quiet,
        "to_file": args.to_file,
        "no_plots": args.no_plots,
        "figures_dir": args.plotpath,
        "data_dir": args.datapath,
    }
    hist_plotter = histograms_by_fraction.Pipeline(**pipeline_config)
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
        "-o",
        "--no-overplot",
        help="Suppress overplotting of virial temperature regions",
        dest="overplot",
        action="store_false",
    )
    parser.add_argument(
        "-t",
        "--total-mass",
        help="Use gas mass as hist weights instead of gas mass fraction",
        dest="total_mass",
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
        "-b",
        "--bins",
        help="The number of temperature bins, defaults to 50",
        dest="bins",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--plot-output-dir",
        help=(
            "The directory path under which to save the plots, if created. "
            "It is recommended to leave this at the default value unless "
            "the expected directories do not exist."
        ),
        dest="plotpath",
        default=None,
    )
    parser.add_argument(
        "--data-output-dir",
        help=(
            "The directory path under which to save the plots, if created. "
            "It is recommended to leave this at the default value unless "
            "the expected directories do not exist."
        ),
        dest="datapath",
        default=None,
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
