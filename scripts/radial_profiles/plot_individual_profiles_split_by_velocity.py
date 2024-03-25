import argparse
import logging
import sys
from pathlib import Path

root_dir = Path(__file__).parents[2].resolve()
sys.path.insert(0, str(root_dir / "src"))

from library import scriptparse
from pipelines.radial_profiles.by_velocity import (
    GenerateIndividualHistogramsPipeline,
    PlotMeanProfilesPipeline,
)


def main(args: argparse.Namespace) -> None:
    """Create histograms of temperature distribution"""
    # paths
    if args.core_only:
        type_flag = f"{args.regime}_core"  # prevent overwriting
    else:
        type_flag = args.regime
    if args.virial_velocity:
        type_flag += "_virial_vel"

    pipeline_config = scriptparse.startup(
        args,
        "density_profiles_split",
        type_flag,
        with_virial_temperatures=False,
        figures_subdirectory="./../",
    )

    pipeline_config.update(
        {
            "limiting_velocity": args.limit_velocity,
            "regime": args.regime,
            "radial_bins": args.rbins,
            "max_distance": 0.05 if args.core_only else 2.0,
            "use_virial_velocity": args.virial_velocity,
            "log": args.log,
        }
    )

    # find the plotting pipeline to use
    if args.plot_type == "overall-mean":
        plot_pipeline = PlotMeanProfilesPipeline(**pipeline_config)
    else:
        logging.fatal(f"Unknown/unsupported plot type {args.plot_type}.")
        return

    if args.force_generation:
        gen_pipeline = GenerateIndividualHistogramsPipeline(**pipeline_config)
        gen_pipeline.run()
        plot_pipeline.run()
    else:
        # attempt to plot directly, assuming data is available
        success = plot_pipeline.run()
        if success == 0:
            # worked, all done!
            return
        # did not work, try fallbacks
        logging.error(
            "Some data files did not exist. Could not start plotting pipeline."
        )
        if args.forbid_generation:
            logging.fatal(
                "Data generation was explicitly forbidden, so execution must "
                "be abandoned."
            )
            return
        # otherwise, generate data and run plotting pipeline again
        logging.info("Re-running data generation pipeline.")
        gen_pipeline = GenerateIndividualHistogramsPipeline(**pipeline_config)
        gen_pipeline.run()
        plot_pipeline.run()


if __name__ == "__main__":
    parser = scriptparse.BaseScriptParser(
        prog=f"python {Path(__file__).name}",
        description=(
            "Plot individual radial density profiles of all clusters in TNG "
            "split by the velocity of the gas (into inflowing, quasi-static "
            "and outflowing gas)."
        ),
        allowed_sims=("TNG300", "TNG-Cluster"),
    )
    parser.remove_argument("sim")
    parser.remove_argument("processes")
    parser.remove_argument("from_file")
    parser.remove_argument("to_file")
    parser.add_argument(
        "-t",
        "--plot-type",
        help="The type of plot to plot from the data",
        dest="plot_type",
        choices=["overall-mean"],
        default="overall-mean",
    )
    parser.add_argument(
        "-r",
        "--regime",
        help=(
            "The temperature regime in which to plot the radial density "
            "profile split by velocity."
        ),
        dest="regime",
        type=str,
        default="cool",
        choices=["cool", "warm", "hot", "total"],
    )
    exclusion_group = parser.add_mutually_exclusive_group(required=False)
    exclusion_group.add_argument(
        "-g",
        "--force-generation",
        help=(
            "Force the (re-)generation of data. Data generation will be "
            "skipped by default if the data already exists on file. To "
            "forcibly generate the data from file again, set this flag."
        ),
        dest="force_generation",
        action="store_true",
    )
    exclusion_group.add_argument(
        "-n",
        "--forbid-generation",
        help=(
            "Forbid the (re-)generation of data. Data generation will be "
            "skipped by default if the data already exists on file."
        ),
        dest="forbid_generation",
        action="store_true",
    )
    parser.add_argument(
        "--vmax",
        help=(
            "Limiting velocity for the quasi-static gas. All gas with an "
            "absolute radial velocity smaller than this threshold will be "
            "considered quasi-static. When using virial velocities instead "
            "of physical velocities, this must be a fraction of the virial "
            "velocity. Defaults to 100 km/s."
        ),
        dest="limit_velocity",
        type=float,
        default=100.0,
        metavar="VELOCITY",
    )
    parser.add_argument(
        "--log",
        help="Plot the figures in log scale instead of linear scale.",
        action="store_true",
        dest="log",
    )
    parser.add_argument(
        "-uv",
        "--use-virial-velocities",
        help=(
            "Use virial velocities instead of velocity in absolute units. "
            "This wil normalize gas velocities in every cluster to the "
            "cluster virial velocity before plotting."
        ),
        dest="virial_velocity",
        action="store_true",
    )
    parser.add_argument(
        "-cc",
        "--cluster-core",
        help=(
            "Plot the core region of the cluster only. This will restrict the "
            "radial range of the plot to around 50 kpc physical size."
        ),
        dest="core_only",
        action="store_true",
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

    # parse arguments
    try:
        args = parser.parse_args()
        main(args)
    except KeyboardInterrupt:
        print("Execution forcefully stopped.")
        sys.exit(1)
