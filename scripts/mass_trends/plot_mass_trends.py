import argparse
import sys
from pathlib import Path

# import the helper scripts
cur_dir = Path(__file__).parent.resolve()
sys.path.append(str(cur_dir.parent.parent / "pipelines"))
sys.path.append(str(cur_dir.parent.parent / "src"))

from config import config
from mass_trends.binned import MassTrendPipeline


def main(args: argparse.Namespace) -> None:
    """Create a plot of the trends of mass"""
    # sim data
    if args.sim == "TEST_SIM":
        sim = "TNG50-4"
    elif args.sim == "DEV_SIM":
        sim = "TNG50-3"
    elif args.sim == "MAIN_SIM":
        sim = "TNG300-1"
    else:
        raise ValueError(f"Unknown simulation type {args.sim}.")

    # config
    cfg = config.get_default_config(sim)

    # paths
    figure_path = cfg.figures_home / f"mass_trends/{cfg.sim_path}"
    figure_stem = f"mass_trend_{cfg.sim_path}"

    if args.figurespath:
        new_path = Path(args.figurespath)
        if new_path.exists() and new_path.is_dir():
            figure_path = new_path
        else:
            print(
                f"WARNING: Given figures path is invalid: {str(new_path)}."
                f"Using fallback path {str(figure_path)} instead."
            )

    data_path = cfg.data_home / "temperature_distribution"
    data_stem = f"temperature_hist_frac_{cfg.sim_path}"
    if args.datapath:
        new_path = Path(args.datapath)
        if new_path.exists() and new_path.is_dir():
            data_path = new_path
        else:
            print(
                f"WARNING: Given data path is invalid: {str(new_path)}."
                f"Attempting fallback path {str(data_path)} instead."
            )

    # create pipeline parameter dictionary
    file_data = {
        "figures_dir": figure_path,
        "data_dir": data_path,
        "figures_file_stem": figure_stem,
        "data_file_stem": data_stem,
    }

    pipeline_config = {
        "config": cfg,
        "paths": file_data,
        "processes": 1,
        "mass_bin_edges": [1e8, 1e9, 1e10, 1e11, 1e12, 1e13, 1e14, 1e15],
        "normalize": args.normalized,
        "cool_bins": args.n_cool_bins,
        "warm_bins": args.n_warm_bins,
    }

    pipeline = MassTrendPipeline(**pipeline_config)
    pipeline.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog=f"python {Path(__file__).name}",
        description="Plot radial temperature profiles of halos in TNG",
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
        "--normalized",
        help="Whether to use the division based on virial temperature.",
        dest="normalized",
        action="store_true",
    )
    parser.add_argument(
        "--cool-bins",
        help=(
            "The number of bins that belong to the cold regime. Defaults"
            "to 15."
        ),
        dest="n_cool_bins",
        metavar="NUMBER-OF-BINS",
        type=int,
        default=15,
    )
    parser.add_argument(
        "--warm-bins",
        help=(
            "The number of bins that belong to the warm regime, counted "
            "from the last cool gas bin. Defaults to 15."
        ),
        dest="n_warm_bins",
        metavar="NUMBER-OF-BINS",
        type=int,
        default=15,
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
            "The directory where the data files are located. They are "
            "expected to have the default names."
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
