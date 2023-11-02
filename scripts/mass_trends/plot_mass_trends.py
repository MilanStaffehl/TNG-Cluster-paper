import argparse
import sys
from pathlib import Path

root_dir = Path(__file__).parents[2].resolve()
sys.path.insert(0, str(root_dir / "src"))

import glob_util
from library.config import config
from pipelines.mass_trends.temperatures import (
    FromFilePipeline,
    IndividualsMassTrendPipeline,
)


def main(args: argparse.Namespace) -> None:
    """Create plot of gas mass trends for individual halos"""
    # sim data
    sim = glob_util.translate_sim_name(args.sim)

    # config
    cfg = config.get_default_config(sim)

    # temperature divisions
    if args.normalize:
        temperature_divs = [-100.0, -3.5, -1.0, 100.0]
    else:
        temperature_divs = [0.0, 4.5, 5.5, 10.0]

    # whether to use median or mean
    if args.average:
        statistics = "mean"
    else:
        statistics = "median"

    # file name type flag
    if args.normalize:
        type_flag = "normalized"
    else:
        type_flag = "standard"
    type_flag = f"{type_flag}_{statistics}"
    if args.running_median:
        type_flag = f"{type_flag}_rm"

    # paths
    file_data = glob_util.assemble_path_dict(
        "mass_trends",
        cfg,
        type_flag,
        args.normalize,
        args.figurespath,
        args.datapath,
    )

    pipeline_config = {
        "config": cfg,
        "paths": file_data,
        "processes": args.processes,
        "mass_bin_edges": [1e8, 1e9, 1e10, 1e11, 1e12, 1e13, 1e14, 1e15],
        "temperature_divisions": temperature_divs,
        "normalize": args.normalize,
        "statistic_method": statistics,
        "running_median": args.running_median,
        "quiet": args.quiet,
        "to_file": args.to_file,
        "no_plots": args.no_plots,
    }
    if args.from_file:
        pipeline = FromFilePipeline(**pipeline_config)
    else:
        pipeline = IndividualsMassTrendPipeline(**pipeline_config)
    pipeline.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog=f"python {Path(__file__).name}",
        description="Plot mass trends of gas of halos in TNG",
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
        metavar="PROCESSES",
    )
    parser.add_argument(
        "-f",
        "--to-file",
        help=(
            "Whether to write the plot data and virial temperature data "
            "calculated to file"
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
        "-n",
        "--normalize-temperatures",
        help="Normalize temperatures to virial temperature",
        dest="normalize",
        action="store_true",
    )
    parser.add_argument(
        "-a",
        "--use-average",
        help="Plot averages instead of medians in the plot",
        dest="average",
        action="store_true",
    )
    parser.add_argument(
        "-r",
        "--running-median",
        help=(
            "Plot a continuous median with confidence region instead of "
            "binned data points. Also works for averages."
        ),
        dest="running_median",
        action="store_true",
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
