import argparse
import logging
import logging.config
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# deprecation notice
notice = (
    "This script is no longer supported. The data required by this script "
    "can not be produced by the code base anymore. This script will soon "
    "be removed from the code base. Usage is discouraged. To force execution "
    "of this script anyway, use the -f flag."
)

# import the helper scripts
cur_dir = Path(__file__).parent.resolve()
sys.path.append(str(cur_dir.parent.parent / "src"))
import logging_config


def main(args: argparse.Namespace) -> None:
    """Plot the difference when treating SF gas differently"""
    warnings.warn(message=notice, category=FutureWarning, stacklevel=2)

    logging_cfg = logging_config.get_logging_config("INFO")
    logging.config.dictConfig(logging_cfg)
    logger = logging.getLogger("root")

    # mass bins used
    mass_bins = [1e8, 1e9, 1e10, 1e11, 1e12, 1e13, 1e14, 1e15]

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

    # load the hist data
    logger.info("Loading data from file.")
    sim_file_name = SIMULATION.replace("-", "_")
    filepath = "./../../data/001/"
    base_fn = f"temperature_hists_{sim_file_name}_{weight_type}_legacy.npz"
    with np.load(f"{filepath}{base_fn}") as hist_data:
        base_means = hist_data["hist_mean"]
    treated_fn = f"temperature_hists_{sim_file_name}_{weight_type}.npz"
    with np.load(f"{filepath}{treated_fn}") as hist_data:
        treated_means = hist_data["hist_mean"]
    logger.info("Successfully loaded data from file.")

    # calculate difference and plot it
    logger.info("Calculating differences.")
    for mass_idx in range(len(base_means)):
        base_mean = base_means[mass_idx]
        treated_mean = treated_means[mass_idx]
        # differences
        difference = treated_mean - base_mean
        pos_diff = difference.copy()
        pos_diff[pos_diff < 0] = 0  # set all entries below zero to zero
        neg_diff = difference.copy()
        neg_diff[neg_diff > 0] = 0  # set all entries above zero to zero

        # plot
        logger.info(f"Plotting difference for mass bin {mass_idx}.")
        fig, axes = plt.subplots(figsize=(5, 4))
        axes.set_title(
            r"$M_{200c}$: "
            rf"${np.log10(mass_bins[mass_idx])} < \log \ M_\odot "
            rf"< {np.log10(mass_bins[mass_idx + 1])}$"
        )
        axes.set_xlabel("Gas temperature [log K]")
        if weight_type == "frac":
            axes.set_ylabel("Average gas mass fraction")
        else:
            axes.set_ylabel(r"Average gas mass per cell [$M_\odot$]")

        # calculate bin positions
        _, bins = np.histogram(np.array([0]), bins=50, range=(3.0, 8.0))
        centers = (bins[:-1] + bins[1:]) / 2

        # plot data
        facecolor = "lightblue" if weight_type == "frac" else "lightcoral"
        plot_config = {
            "histtype": "stepfilled",
            "facecolor": facecolor,
            "edgecolor": "black",
            "log": True,
            "zorder": 10,
        }
        # hack: produce exactly one entry for every bin, but weight it
        # by the histogram bar length, to achieve a "fake" bar plot
        axes.hist(
            centers,
            bins=bins,
            range=(3.0, 8.0),
            weights=(base_mean - np.abs(neg_diff)),
            **plot_config
        )
        # plot behind it the changes
        plot_config.update({"facecolor": "red", "zorder": 5})
        axes.hist(
            centers,
            bins=bins,
            range=(3.0, 8.0),
            weights=(base_mean + np.abs(neg_diff)),
            **plot_config
        )
        plot_config.update({"facecolor": "green", "zorder": 1})
        axes.hist(
            centers,
            bins=bins,
            range=(3.0, 8.0),
            weights=(base_mean + np.abs(pos_diff)),
            **plot_config,
        )

        # save figure
        suffix = f"_{SIMULATION.replace('-', '_')}"
        filename = f"temperature_hist_{mass_idx}{suffix}_sfr_diff.pdf"
        fig.savefig(f"./../../figures/001/{filename}", bbox_inches="tight")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog=f"python {Path(__file__).name}",
        description=(
            "Plot temperature distribution of halos in TNG, using "
            f"data from file.\nWARNING: {notice}"
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
        "-t",
        "--total-mass",
        help="Use the histograms weighted by gas mass instead of gas fraction",
        dest="total_mass",
        action="store_true",
    )
    parser.add_argument(
        "-f",
        "--force",
        help=(
            "Force the execution of the script. If not used, a FutureWarning will be raised."
        ),
        action="store_true",
        dest="force",
    )

    # parse arguments
    try:
        args = parser.parse_args()
        if not args.force:
            raise FutureWarning(notice)
        main(args)
    except KeyboardInterrupt:
        print("Execution forcefully stopped by user.")
        sys.exit(1)
