import argparse
import sys
from pathlib import Path

root_dir = Path(__file__).parents[2].resolve()
sys.path.insert(0, str(root_dir / "src"))

import glob_util
from library.config import config
from pipelines.radial_profiles.density_individual import (
    IDProfilesFromFilePipeline,
    IndividualDensityProfilePipeline,
)
from pipelines.radial_profiles.temperature_individual import (
    IndividualTemperatureProfilePipeline,
    ITProfilesFromFilePipeline,
)


def main(args: argparse.Namespace) -> None:
    """Create histograms of temperature distribution"""
    # sim data
    sim = glob_util.translate_sim_name(args.sim)

    # config
    cfg = config.get_default_config(sim)

    # paths
    file_data = glob_util.assemble_path_dict(
        "radial_profiles",
        cfg,
        args.what,
        False,
        args.figurespath,
        args.datapath,
        figures_subdirectory="./individuals/",
        data_subdirectory=f"./individuals/{cfg.sim_path}/",
    )

    pipeline_config = {
        "config": cfg,
        "paths": file_data,
        "processes": args.processes,
        "quiet": args.quiet,
        "to_file": args.to_file,
        "no_plots": args.no_plots,
        "radial_bins": args.rbins,
        "temperature_bins": args.tbins,
        "log": args.log,
        "forbid_tree": args.forbid_tree,
    }
    if args.what == "temperature":
        if args.from_file:
            pipeline = ITProfilesFromFilePipeline(**pipeline_config)
        else:
            pipeline = IndividualTemperatureProfilePipeline(**pipeline_config)
    else:
        # remove temperature bins argument
        pipeline_config.pop("temperature_bins")
        if args.from_file:
            pipeline = IDProfilesFromFilePipeline(**pipeline_config)
        else:
            pipeline = IndividualDensityProfilePipeline(**pipeline_config)
    pipeline.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog=f"python {Path(__file__).name}",
        description=(
            "Plot individual radial profiles of all halos in TNG with mass "
            "above 10^14 solar masses."
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
        "-w",
        "--what",
        help=(
            "What type of radial profile to plot: temperature or density. Defaults "
            "to temperature."
        ),
        dest="what",
        type=str,
        default="temperature",
        choices=["temperature", "density"],
    )
    parser.add_argument(
        "-p",
        "--processes",
        help=(
            "Use multiple workers when querying KDTree, with the number of "
            "workers specified after this flag. Has no effect when associated "
            "particle ID data files exist."
        ),
        type=int,
        default=0,
        dest="processes",
        metavar="NUMBER",
    )
    parser.add_argument(
        "-t",
        "--forbid-tree",
        help=(
            "Forbid the construction of a KDTree. Will cause the job to fail "
            "if construction of a KDTree is required in order to find the "
            "neighboring particles of all halos above the mass threshold."
        ),
        dest="forbid_tree",
        action="store_true",
    )
    parser.add_argument(
        "--log",
        help="Plot the figures in log scale instead of linear scale.",
        action="store_true",
        dest="log",
    )
    parser.add_argument(
        "-f",
        "--to-file",
        help="Whether to write the histogram data calculated to file.",
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
        help="Prevent progress and memory usage information to be emitted.",
        dest="quiet",
        action="store_true",
    )
    parser.add_argument(
        "-tb",
        "--tbins",
        help="The number of temperature bins, defaults to 50",
        dest="tbins",
        type=int,
        default=50,
        metavar="NUMBER",
    )
    parser.add_argument(
        "-rb",
        "--rbins",
        help="The number of radial bins, defaults to 50",
        dest="rbins",
        type=int,
        default=50,
        metavar="NUMBER",
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
