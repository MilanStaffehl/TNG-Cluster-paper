import argparse
import sys
from pathlib import Path

root_dir = Path(__file__).parents[2].resolve()
sys.path.insert(0, str(root_dir / "src"))

from library import scriptparse
from pipelines.tracer_history.final_plots import ParentCategoryBarPlotPipeline


def main(args: argparse.Namespace) -> None:
    """Plot finalizing points using all existing data."""
    args.sim = "TNG-Cluster"

    # type flag, subdirectories, and other config changes
    type_flag = "final_plots"

    pipeline_config = scriptparse.startup(
        args,
        "tracer_history",
        type_flag,
        figures_subdirectory=".",
        data_subdirectory=".",
        suppress_sim_name_in_files=False,
    )

    # update pipeline config here
    pipeline_config.update({"fractions": args.fractions})

    # select and build pipeline
    pipeline = ParentCategoryBarPlotPipeline(**pipeline_config)
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
        "--fractions",
        help=(
            "When set, the plot will show fractions instead of tracer mass "
            "on the y-axis."
        ),
        dest="fractions",
        action="store_true",
    )

    # parse arguments
    try:
        args_ = parser.parse_args()
        main(args_)
    except KeyboardInterrupt:
        print("\nExecution forcefully stopped.\n")
        sys.exit(1)
