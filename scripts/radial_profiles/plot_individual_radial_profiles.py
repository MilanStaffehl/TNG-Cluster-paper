import argparse
import sys
from pathlib import Path

import numpy as np

root_dir = Path(__file__).parents[2].resolve()
sys.path.insert(0, str(root_dir / "src"))

import glob_util
from library.config import config
from pipelines.radial_profiles.individuals import (
    IndividualProfilesFromFilePipeline,
    IndividualRadialProfilePipeline,
    IndivTemperatureTNGClusterPipeline,
)


def main(args: argparse.Namespace) -> None:
    """Create histograms of temperature distribution"""
    # sim data
    sim = glob_util.translate_sim_name(args.sim)

    # config
    cfg = config.get_default_config(sim)

    # paths
    if args.core_only:
        type_flag = f"{args.what}_core"  # prevent overwriting
    else:
        type_flag = args.what
    file_data = glob_util.assemble_path_dict(
        "radial_profiles",
        cfg,
        type_flag,
        False,
        args.figurespath,
        args.datapath,
        figures_subdirectory="./individuals/",
        data_subdirectory=f"./individuals/{cfg.sim_path}/",
    )

    # temperature bins is either the number of bins or the three regimes
    if args.what == "temperature":
        tbins = args.tbins
    else:
        tbins = np.array([0, 4.5, 5.5, np.inf])

    # if only the core is to be shown, restrict radial range
    if args.core_only:
        ranges = np.array([[0, 0.05], [3, 8.5]])  # units: R_vir, log K
    else:
        ranges = np.array([[0, 2], [3, 8.5]])  # units: R_vir, log K

    pipeline_config = {
        "config": cfg,
        "paths": file_data,
        "processes": args.processes,
        "quiet": args.quiet,
        "to_file": args.to_file,
        "no_plots": args.no_plots,
        "what": args.what,
        "radial_bins": args.rbins,
        "temperature_bins": tbins,
        "log": args.log,
        "forbid_tree": args.forbid_tree,
        "ranges": ranges,
        "core_only": args.core_only,
    }
    if args.from_file:
        pipeline = IndividualProfilesFromFilePipeline(**pipeline_config)
    elif sim == "TNG-Cluster":
        pipeline = IndivTemperatureTNGClusterPipeline(**pipeline_config)
    else:
        pipeline = IndividualRadialProfilePipeline(**pipeline_config)
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
            "is TNG50-3, test sim is TNG50-4, and CLUSTER is TNG-Cluster. "
            "Defaults to TNG300-1."
        ),
        dest="sim",
        type=str,
        default="MAIN_SIM",
        choices=["MAIN_SIM", "DEV_SIM", "TEST_SIM", "CLUSTER"],
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
            "present. When used, the flags -p, -f, -t, -q have no effect."
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
        "-c",
        "--cluster-core",
        help=(
            "Plot the core region of the cluster only. This will restrict the "
            "radial range of the plot to around 50 kpc physical size."
        ),
        dest="core_only",
        action="store_true",
    )
    parser.add_argument(
        "-tb",
        "--tbins",
        help=(
            "The number of temperature bins, defaults to 50. Has no effect "
            "when plotting density profiles."
        ),
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
