import argparse
import logging
import logging.config
import sys
from pathlib import Path

# import the helper scripts
cur_dir = Path(__file__).parent.resolve()
sys.path.append(str(cur_dir.parent.parent / "src"))
import logging_config
from processors import temperature_hists


def main(args: argparse.Namespace) -> None:
    """Plot temperature distribution from data in files"""
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

    # histogram weights
    if args.total_mass:
        weight_type = "mass"
    else:
        weight_type = "frac"

    # plot hist data
    MASS_BINS = [1e8, 1e9, 1e10, 1e11, 1e12, 1e13, 1e14, 1e15]
    hist_plotter = temperature_hists.TemperatureDistributionProcessor(
        sim=SIMULATION,
        logger=logger,
        n_temperature_bins=args.bins,
        mass_bins=MASS_BINS,
        weight=weight_type,
    )

    # load hist data from file
    FILE_SUFFIX = f"_{SIMULATION.replace('-', '_')}_{weight_type}"
    filename = f"temperature_hists{FILE_SUFFIX}.npz"
    filepath = Path().home() / "thesisProject" / "data" / "001"
    hist_plotter.load_data(filepath / filename)
    if args.virial_temperatures:
        filename = f"virial_temperatures_{SIMULATION.replace('-', '_')}.npy"
        hist_plotter.load_virial_temperatures(filepath / filename)

    # plot the historgram with the file data
    for i in range(len(MASS_BINS) - 1):
        hist_plotter.plot_data(
            i, suffix=FILE_SUFFIX, plot_vir_temp=args.virial_temperatures
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
    parser.add_argument(
        "-t",
        "--total-mass",
        help="Use the histograms weighted by gas mass instead of gas fraction",
        dest="total_mass",
        action="store_true",
    )
    parser.add_argument(
        "-b",
        "--bins",
        help=(
            "The number of temperature bins which must match the number of "
            "bins in the saved data, defaults to 50"
        ),
        dest="bins",
        type=int,
        default=50,
    )

    # parse arguments
    try:
        args = parser.parse_args()
        main(args)
    except KeyboardInterrupt:
        print("Execution forcefully stopped by user.")
        sys.exit(1)
