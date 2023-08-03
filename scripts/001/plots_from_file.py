import argparse
import logging
import logging.config
import sys
from pathlib import Path

# import the helper scripts
cur_dir = Path(__file__).parent.resolve()
sys.path.append(str(cur_dir.parent.parent / "src"))
import logging_config
from plotters import temperature_hists


def main(sim: str, vir_temp: bool = True) -> None:
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
    FILE_SUFFIX = f"_{SIMULATION.replace('-', '_')}"
    filename = f"temperature_hists{FILE_SUFFIX}.npz"
    filepath = Path().home() / "thesisProject" / "data" / "001"
    hist_plotter.load_stacked_hist(filepath / filename)
    if vir_temp:
        filename = f"virial_temperatures{FILE_SUFFIX}.npy"
        hist_plotter.load_virial_temperatures(filepath / filename)

    # plot the historgram with the file data
    for i in range(len(MASS_BINS) - 1):
        hist_plotter.plot_stacked_hist(
            i, suffix=FILE_SUFFIX, plot_vir_temp=vir_temp
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog=f"python {Path(__file__).name}",
        description=(
            "Plot temperature distribution of halos in TNG, using "
            "data from file"
        ),
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
        "-n",
        "--no-virial-temperatures",
        help="Suppress plotting virial temperature region overlay",
        dest="virial_temperatures",
        action="store_false",
    )

    # parse arguments
    try:
        args = parser.parse_args()
        main(args.sim, args.virial_temperatures)
    except KeyboardInterrupt:
        print("Execution forcefully stopped by user.")
        sys.exit(1)
