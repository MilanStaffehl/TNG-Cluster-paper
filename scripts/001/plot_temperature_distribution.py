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
from plotters import temperature_hists


def main(
    sim: str,
    multiproc: bool = True,
    processes: int = 16,
    to_file: bool = False,
    suppress_plots: bool = False,
    quiet: bool = False,
) -> None:
    """Create histograms of temperature distribution"""
    logging_cfg = logging_config.get_logging_config("INFO")
    logging.config.dictConfig(logging_cfg)
    logger = logging.getLogger("root")

    # sim data
    if sim == "TEST_SIM":
        SIMULATION = "TNG50-4"
    elif sim == "DEV_SIM":
        SIMULATION = "TNG50-3"
    elif sim == "MAIN_SIM":
        SIMULATION = "TNG300-1"
    else:
        raise ValueError(f"Unknown simulation type {sim}.")

    # plot hist data
    MASS_BINS = [1e8, 1e9, 1e10, 1e11, 1e12, 1e13, 1e14, 1e15]
    hist_plotter = temperature_hists.TemperatureDistributionPlotter(
        SIMULATION, MASS_BINS, logger
    )

    # time the full calculation process
    begin = time.time()
    if multiproc:
        hist_plotter.get_hists(processes=processes)
    else:
        hist_plotter.get_hists_lin(quiet=quiet)
    # calculate average hist data
    FILE_SUFFIX = f"_{SIMULATION.replace('-', '_')}"
    hist_plotter.stack_bins(to_file=to_file, suffix=FILE_SUFFIX)
    end = time.time()

    # get time spent on computation
    time_diff = end - begin
    time_fmt = time.strftime('%H:%M:%S', time.gmtime(time_diff))
    logger.info(f"Spent {time_fmt} hours on execution.")

    if suppress_plots:
        sys.exit(0)
    # plot histograms
    for i in range(len(MASS_BINS) - 1):
        hist_plotter.plot_stacked_hist(i, suffix=FILE_SUFFIX)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="python 001_plot_temperature_distribution.py",
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
        "-f",
        "--to-file",
        help="Whether to write the histogram data to .npy file",
        dest="to_file",
        action="store_true",
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

    # parse arguments
    try:
        args = parser.parse_args()
        main(
            args.sim,
            args.multiproc,
            args.processes,
            args.to_file,
            args.no_plots,
            args.quiet,
        )
    except KeyboardInterrupt:
        print(
            "Execution forcefully stopped. Some subprocesses might still be "
            "running and need to be killed manually if multiprocessing was "
            "used."
        )
        sys.exit(1)
