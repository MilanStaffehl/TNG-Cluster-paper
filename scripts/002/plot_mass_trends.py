import argparse
import logging
import logging.config
import sys
import time
from pathlib import Path

# import the helper scripts
cur_dir = Path(__file__).parent.resolve()
sys.path.append(str(cur_dir.parent.parent / "src"))
import logging_config
from plotters import mass_dependence


def main(args: argparse.Namespace) -> None:
    """Create plots of mass dependence of gas mass (fraction)"""
    logging_cfg = logging_config.get_logging_config("INFO")
    logging.config.dictConfig(logging_cfg)
    logger = logging.getLogger("root")

    # sim data
    if args.sim == "TEST_SIM":
        SIMULATION = "TNG50-4"
    elif args.sim == "DEV_SIM":
        SIMULATION = "TNG50-3"
    elif args.sim == "MAIN_SIM":
        SIMULATION = "TNG300-1"
    else:
        raise ValueError(f"Unknown simulation type {args.sim}.")

    sim = SIMULATION.replace("-", "_")
    data = "mass" if args.total_mass else "frac"
    file_suffix = f"_{sim}_{args.regime}_gas_{data}"

    mass_plotter = mass_dependence.MassDependenceOfGasMassPlotter(
        SIMULATION, logger, args.total_mass
    )

    begin = time.time()
    mass_plotter.get_halo_data()
    mass_plotter.get_gas_data()
    if args.multiproc:
        mass_plotter.get_plot_data(
            args.regime, args.processes, args.to_file, file_suffix
        )
    else:
        mass_plotter.get_plot_data_lin(
            args.regime, args.quiet, args.to_file, file_suffix
        )
    end = time.time()

    # get time spent on computation
    time_diff = end - begin
    time_fmt = time.strftime('%H:%M:%S', time.gmtime(time_diff))
    logger.info(f"Spent {time_fmt} hours on execution.")

    if args.no_plots:
        sys.exit(0)

    mass_plotter.plot_mass_dependence(args.regime, file_suffix)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog=f"python {Path(__file__).name}",
        description="Plot temperature distribution of halos in TNG",
    )
    parser.add_argument(
        "regime",
        help=(
            "The temperature regime for which to plot the gas mass "
            "(fraction) vs halo mass"
        ),
        choices=["cold", "warm", "hot"]
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
        "-m",
        "--multiproc",
        help="Use multiprocessing",
        dest="multiproc",
        action="store_true",
    )
    parser.add_argument(
        "-p",
        "--processes",
        help=(
            "Number of processes to use for multiprocessing. Ignored if -m "
            "is not used."
        ),
        dest="processes",
        type=int,
        default=16,
    )
    parser.add_argument(
        "-n",
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
        "-t",
        "--total-mass",
        help="Use gas mass as hist weights instead of gas mass fraction",
        dest="total_mass",
        action="store_true",
    )
    parser.add_argument(
        "-f",
        "--to-file",
        help=(
            "Whether to write the plot data (gas mass fraction and halo "
            "masses) calclated to file"
        ),
        dest="to_file",
        action="store_true",
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
