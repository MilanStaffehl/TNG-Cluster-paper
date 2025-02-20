import argparse
import sys
from pathlib import Path

root_dir = Path(__file__).parents[2].resolve()
sys.path.insert(0, str(root_dir / "src"))

from library import scriptparse
from pipelines.tracer_history.final_plots import (
    ParentCategoryBarPlotPipeline,
    ParentCategoryWithClusterMass,
    PlotTracerFractionInRadius,
)


def main(args: argparse.Namespace) -> None:
    """Plot finalizing points using all existing data."""
    args.sim = "TNG-Cluster"

    # type flag, subdirectories, and other config changes
    type_flag = f"final_plots_{args.what.replace('-', '_')}"

    pipeline_config = scriptparse.startup(
        args,
        "tracer_history",
        type_flag,
        figures_subdirectory=".",
        data_subdirectory=".",
        suppress_sim_name_in_files=False,
    )

    # update pipeline config here
    if args.what == "bar-chart":
        pipeline_config.update({"fractions": args.fractions})
        pipeline = ParentCategoryBarPlotPipeline(**pipeline_config)
    elif args.what == "tracer-fraction":
        pipeline = PlotTracerFractionInRadius(**pipeline_config)
    elif args.what == "mass-plot":
        pipeline_config.update({"combine_panels": args.combine})
        pipeline = ParentCategoryWithClusterMass(**pipeline_config)
    else:
        raise KeyError(f"Unsupported plot type: {args.what}")

    sys.exit(pipeline.run())


if __name__ == "__main__":
    parser = scriptparse.BaseScriptParser(
        prog=f"python {Path(__file__).name}",
        description=(
            "Plot the final plots of the thesis, utilizing all data in the "
            "archive file."
        ),
        allowed_sims=["TNG-Cluster"],
    )
    # remove unnecessary args
    parser.remove_argument("sim")
    parser.remove_argument("processes")
    parser.remove_argument("from_file")
    parser.remove_argument("to_file")

    # add new args
    parser.add_argument(
        "what",
        help="Which plot type to plot.",
        choices=["bar-chart", "tracer-fraction", "mass-plot"],
    )
    parser.add_argument(
        "--fractions",
        help=(
            "When set, the plot will show fractions instead of tracer mass "
            "on the y-axis. Only has an effect when plotting bar chart."
        ),
        dest="fractions",
        action="store_true",
    )
    parser.add_argument(
        "--combine",
        help=(
            "Combine the three different panels in the mass trend plots into "
            "one single figure with three panels. If not set, the mass "
            "dependence of each category with cluster mass at each of the "
            "three time points will be a separate figure. Only has an "
            "effect when choosing `mass-plot` as plot type."
        ),
        dest="combine",
        action="store_true",
    )

    # parse arguments
    try:
        args_ = parser.parse_args()
        main(args_)
    except KeyboardInterrupt:
        print("\nExecution forcefully stopped.\n")
        sys.exit(1)
