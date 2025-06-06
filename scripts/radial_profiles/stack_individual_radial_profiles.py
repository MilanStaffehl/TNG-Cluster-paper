import argparse
import sys
from pathlib import Path

root_dir = Path(__file__).parents[2].resolve()
sys.path.insert(0, str(root_dir / "src"))

from library import scriptparse
from pipelines.radial_profiles.stacks import StackProfilesPipeline


def main(args: argparse.Namespace) -> None:
    """Create stacks of radial profiles for clusters"""
    sim_path = args.sim.replace("-", "_")
    pipeline_config = scriptparse.startup(
        args,
        "radial_profiles",
        args.what,
        with_virial_temperatures=False,
        data_subdirectory=f"./individuals/{sim_path}/{args.what}_profiles"
    )

    pipeline_config.update(
        {
            "log": args.log,
            "what": args.what,
            "method": args.method,
        }
    )
    pipeline = StackProfilesPipeline(**pipeline_config)
    sys.exit(pipeline.run())


if __name__ == "__main__":
    parser = scriptparse.BaseScriptParser(
        prog=f"python {Path(__file__).name}",
        description=(
            "Stack individual radial profiles of all halos in TNG with mass "
            "above 10^14 solar masses."
        ),
        allowed_sims=("TNG300", "TNG100", "TNG50", "TNG-Cluster"),
    )
    parser.remove_argument("processes")
    parser.remove_argument("to_file")
    parser.remove_argument("from_file")
    parser.remove_argument("no_plots")
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
