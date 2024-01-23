import argparse
import sys
from pathlib import Path

root_dir = Path(__file__).parents[2].resolve()
sys.path.insert(0, str(root_dir / "src"))

import glob_util
from library.config import config
from pipelines.radial_profiles.stacks import StackProfilesPipeline


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
        data_subdirectory=f"./individuals/{cfg.sim_path}/{args.what}_profiles",
    )

    pipeline_config = {
        "config": cfg,
        "paths": file_data,
        "processes": -1,
        "quiet": False,
        "to_file": True,
        "no_plots": False,
        "log": args.log,
        "what": args.what,
        "method": args.method,
    }
    pipeline = StackProfilesPipeline(**pipeline_config)
    pipeline.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog=f"python {Path(__file__).name}",
        description=(
            "Stack individual radial profiles of all halos in TNG with mass "
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
        "-m",
        "--method",
        help=(
            "How to stack the histograms, i.e. what statistical method to use "
            "for the stacking. Options are mean (with standard deviation) or "
            "median (with 16th and 84th percentile as error). Defaults to "
            "mean."
        ),
        dest="method",
        type=str,
        default="mean",
        choices=["mean", "median"],
    )
    parser.add_argument(
        "--log",
        help="Plot the figures in log scale instead of linear scale.",
        action="store_true",
        dest="log",
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
            "The directory path from which to load the data. "
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
