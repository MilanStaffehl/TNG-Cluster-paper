import argparse
import sys
from pathlib import Path

root_dir = Path(__file__).parents[2].resolve()
sys.path.insert(0, str(root_dir / "src"))

from library import scriptparse
from pipelines.tracer_history.crossing_times import PLOT_TYPES, PlotCrossingTimesPlots


def main(args: argparse.Namespace) -> None:
    """Plot how different gas quantities evolve with time"""
    args.sim = "TNG-Cluster"

    # type flag, subdirectories, and other config changes
    type_flag = "crossing_times"

    # figures subdir
    figures_subdir = f"./{type_flag}"

    pipeline_config = scriptparse.startup(
        args,
        "tracer_history",
        type_flag,
        figures_subdirectory=figures_subdir,
        data_subdirectory="TNG_Cluster",
        suppress_sim_name_in_files=False,
    )

    # select pipeline and config
    try:
        plot_types = args.plot_types.split(",")
    except AttributeError:
        plot_types = None
    pipeline_config.update({"plot_types": plot_types})

    pipeline = PlotCrossingTimesPlots(**pipeline_config)
    sys.exit(pipeline.run())


if __name__ == "__main__":
    parser = scriptparse.BaseScriptParser(
        prog=f"python {Path(__file__).name}",
        description=(
            "Plot the various plots containing the crossing times of the "
            "traced gas particles into their respective cluster."
        ),
        allowed_sims=["TNG-Cluster"],
    )
    # remove unnecessary args
    parser.remove_argument("sim")
    parser.remove_argument("processes")
    parser.remove_argument("to_file")
    parser.remove_argument("from_file")

    # add new args
    parser.add_argument(
        "-pt",
        "--plot-types",
        help=(
            f"Comma-separated list of plot types to create. When not set, all "
            f"available plot types are plotted. Must be one of the following "
            f"valid plot types: {', '.join(PLOT_TYPES)}. Has no effect when "
            f"using `--zoom-in`."
        ),
        dest="plot_types",
        metavar="LIST",
        type=str,
        default=None,
    )

    # parse arguments
    try:
        args_ = parser.parse_args()
        main(args_)
    except KeyboardInterrupt:
        print("\nExecution forcefully stopped.\n")
        sys.exit(1)
