import argparse
import logging
import logging.config
import sys
from pathlib import Path

# import the helper scripts
cur_dir = Path(__file__).parent.resolve()
sys.path.append(str(cur_dir.parent / "src"))
import logging_config
from plotters import temperature_hists


def main(sim: str) -> None:
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

    # load hist data from file
    hist_plotter.load_stacked_hist("~/thesisProject/data/")


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

    # parse arguments
    try:
        args = parser.parse_args()
        main(args.sim)
    except KeyboardInterrupt:
        print(
            "Execution forcefully stopped. Some subprocesses might still be "
            "running and need to be killed manually if multiprocessing was "
            "used."
        )
        sys.exit(1)
