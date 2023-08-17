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
from processors import temperature_hists


def main(args: argparse.Namespace) -> None:
    """Create histograms of temperature distribution"""
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

    # plotter for hist data
    MASS_BINS = [1e8, 1e9, 1e10, 1e11, 1e12, 1e13, 1e14, 1e15]
    hist_plotter = temperature_hists.TemperatureDistributionProcessor(
        sim=SIMULATION,
        logger=logger,
        n_temperature_bins=args.bins,
        mass_bins=MASS_BINS,
        weight=weight_type,
    )

    # assemble pre- and post processing kwargs
    aux_kwargs = {
        "virial_temperatures": args.overplot,
        "to_file": args.to_file,
        "suffix": f"_{SIMULATION.replace('-', '_')}",
    }
    post_kwargs = {
        "to_file": args.to_file,
        "suffix": f"_{SIMULATION.replace('-', '_')}_{weight_type}"
    }

    # time the full calculation process
    begin = time.time()
    hist_plotter.get_data(
        args.processes,
        args.quiet,
        aux_kwargs=aux_kwargs,
        post_kwargs=post_kwargs
    )
    end = time.time()

    # get time spent on computation
    time_diff = end - begin
    time_fmt = time.strftime('%H:%M:%S', time.gmtime(time_diff))
    logger.info(f"Spent {time_fmt} hours on execution.")

    if args.no_plots:
        sys.exit(0)

    # plot histograms
    for i in range(len(MASS_BINS) - 1):
        hist_plotter.plot_data(
            i,
            suffix=f"_{SIMULATION.replace('-', '_')}_{weight_type}",
            plot_vir_temp=args.overplot,
        )


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
        help="Use multiprocessing, optionally define a number of processes",
        type=int,
        nargs="?",
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
        "-b",
        "--bins",
        help="The number of temperature bins, defaults to 50",
        dest="bins",
        type=int,
        default=50,
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
