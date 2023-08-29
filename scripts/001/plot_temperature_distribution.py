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

    # which processor to use:
    if args.normalize:
        processor = temperature_hists.NormalizedProcessor
    else:
        processor = temperature_hists.TemperatureDistributionProcessor

    # plotter for hist data
    MASS_BINS = [1e8, 1e9, 1e10, 1e11, 1e12, 1e13, 1e14, 1e15]
    hist_plotter = processor(
        sim=SIMULATION,
        logger=logger,
        n_temperature_bins=args.bins,
        mass_bins=MASS_BINS,
        weight=weight_type,
    )

    # path setup
    sim_dir_name = SIMULATION.replace("-", "_")
    # dir for plots
    if args.plotpath is not None:
        figures_path = Path(args.plotpath)
    else:
        figures_path = (
            hist_plotter.config.figures_home / "001" / sim_dir_name / "galleries"
        )  # yapf: disable
    # dir for data
    if args.datapath is not None:
        data_path = Path(args.datapath)
    else:
        data_path = (hist_plotter.config.data_home / "001" / sim_dir_name)

    # assemble pre- and post processing kwargs
    aux_kwargs = {
        "virial_temperatures": args.overplot,
        "to_file": args.to_file,
        "output": data_path / f"virial_temperatures_{sim_dir_name}.npy",
    }
    post_kwargs = {
        "to_file":
            args.to_file,
        "output":
            data_path / f"temperature_hists_{sim_dir_name}_{weight_type}.npz"
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
        file_name = f"temperature_hists_{i}_{sim_dir_name}_{weight_type}.pdf"
        hist_plotter.plot_data(
            i,
            output=figures_path / file_name,
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
        "--noralize-temperatures",
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
