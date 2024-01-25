import argparse
import sys
from pathlib import Path

root_dir = Path(__file__).parents[2].resolve()
sys.path.insert(0, str(root_dir / "src"))

from library.config import config
from pipelines.radial_profiles.stacks_binned import StackProfilesBinnedPipeline


def main(args: argparse.Namespace) -> None:
    """Create stacks of TNG300 and TNG Cluster clusters"""
    # sim data
    sim = "TNG-Cluster"  # set arbitrarily, not used

    # config
    cfg = config.get_default_config(sim)

    # paths
    if args.figurespath:
        figure_path = Path(args.figurepath)
    else:
        figure_path = cfg.figures_home / "radial_profiles"
    if args.datapath:
        data_path = Path(args.datapath)
    else:
        data_path = cfg.data_home / "radial_profiles" / "individuals"
    file_data = {
        "figures_dir": figure_path.resolve(),
        "data_dir": data_path.resolve(),
        "figures_file_stem": f"radial_profiles_{args.what}_clusters",
        "data_file_stem": f"radial_profiles_{args.what}_clusters",
    }

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
    pipeline = StackProfilesBinnedPipeline(**pipeline_config)
    pipeline.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog=f"python {Path(__file__).name}",
        description=(
            "Stack individual radial profiles of all clusters in TNG300-1 and "
            "TNG Cluster, binned by mass."
        ),
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
        print("Execution forcefully stopped.")
        sys.exit(1)
