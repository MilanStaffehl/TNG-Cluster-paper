import argparse
import sys
from pathlib import Path

root_dir = Path(__file__).parents[2].resolve()
sys.path.insert(0, str(root_dir / "src"))

from library import scriptparse
from pipelines.tracer_history.combined_histograms import PlotStackedHistogramPipeline


def main(args: argparse.Namespace) -> None:
    """Plot a 4D histogram of distance vs. redshift for clusters."""
    args.sim = "TNG-Cluster"

    # type flag, subdirectories, and other config changes
    type_flag = "combined_hist"

    pipeline_config = scriptparse.startup(
        args,
        "tracer_history",
        type_flag,
        figures_subdirectory="./combined_histograms",
        data_subdirectory="TNG_Cluster",
        suppress_sim_name_in_files=False,
    )

    # update pipeline config here

    # select and build pipeline
    pipeline = PlotStackedHistogramPipeline(**pipeline_config)
    sys.exit(pipeline.run())


if __name__ == "__main__":
    parser = scriptparse.BaseScriptParser(
        prog=f"python {Path(__file__).name}",
        description=(
            "Plot a 2D histogram of distance vs. redshift of traced cool gas "
            "predecessors, with the color encodng both their mass "
            "distribution and mean temperature."
        ),
        allowed_sims=["TNG-Cluster"],
    )
    # remove unnecessary args
    parser.remove_argument("sim")
    parser.remove_argument("from_file")
    parser.remove_argument("to_file")
    parser.remove_argument("processes")
    parser.remove_argument("no_plots")

    # add new args

    # parse arguments
    try:
        args_ = parser.parse_args()
        main(args_)
    except KeyboardInterrupt:
        print("\nExecution forcefully stopped.\n")
        sys.exit(1)
