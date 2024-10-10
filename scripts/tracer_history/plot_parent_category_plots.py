import argparse
import sys
from pathlib import Path

root_dir = Path(__file__).parents[2].resolve()
sys.path.insert(0, str(root_dir / "src"))

from library import scriptparse
from pipelines.tracer_history.parent_categories import PlotParentCategoryPlots


def main(args: argparse.Namespace) -> None:
    """Plots for parent categories of traced particles"""
    args.sim = "TNG-Cluster"

    # type flag, subdirectories, and other config changes
    type_flag = "parent_category"

    pipeline_config = scriptparse.startup(
        args,
        "tracer_history",
        type_flag,
        figures_subdirectory=f"./{type_flag}",
        data_subdirectory="TNG_Cluster",
        suppress_sim_name_in_files=False,
    )

    # update pipeline config here

    # select and build pipeline
    pipeline = PlotParentCategoryPlots(**pipeline_config)
    sys.exit(pipeline.run())


if __name__ == "__main__":
    parser = scriptparse.BaseScriptParser(
        prog=f"python {Path(__file__).name}",
        description=(
            "Plot various plots related to the parent category of the traced "
            "particles that eventually make up the cool gas at redshift 0."
        ),
        allowed_sims=["TNG-Cluster"],
    )
    # remove unnecessary args
    parser.remove_argument("sim")
    parser.remove_argument("processes")
    parser.remove_argument("to_file")
    parser.remove_argument("from_file")

    # add new args

    # parse arguments
    try:
        args_ = parser.parse_args()
        main(args_)
    except KeyboardInterrupt:
        print("\nExecution forcefully stopped.\n")
        sys.exit(1)
