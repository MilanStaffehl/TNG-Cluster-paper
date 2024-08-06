import argparse
import sys
from pathlib import Path

root_dir = Path(__file__).parents[2].resolve()
sys.path.insert(0, str(root_dir / "src"))

from library import scriptparse
from pipelines.merger_tree.merger_tree_properties import (
    PlotMergerTreePropertiesPipeline,
)


def main(args: argparse.Namespace) -> None:
    """Plot a subhalo quantity of the primary subhalo of all clusters"""
    args.sim = "TNG-Cluster"

    # type flag, subdirectories, and other config changes
    type_flag = "main_prog_quantity"

    pipeline_config = scriptparse.startup(
        args,
        "merger_tree",
        type_flag,
        figures_subdirectory=".",
        data_subdirectory=".",
        suppress_sim_name_in_files=False,
    )

    # update with custom args
    pipeline_config.update({"field_name": args.field_name})

    # select and build pipeline
    pipeline = PlotMergerTreePropertiesPipeline(**pipeline_config)
    sys.exit(pipeline.run())


if __name__ == "__main__":
    parser = scriptparse.BaseScriptParser(
        prog=f"python {Path(__file__).name}",
        description=(
            "Plot a subhalo quantity for the primary subhalo of every cluster,"
            " following their main progenitor branch."
        ),
        allowed_sims=["TNG-Cluster"],
    )
    # remove unnecessary args
    parser.remove_argument("processes")
    parser.remove_argument("to_file")
    parser.remove_argument("from_file")
    parser.remove_argument("sim")
    parser.remove_argument("no_plots")

    parser.add_argument(
        "-fn",
        "--field-name",
        help="The subhalo or halo field name to plot",
        choices=["HaloMass", "HaloRadius", "SubhaloPos", "SubhaloVel"],
        required=True,
    )

    # parse arguments
    try:
        args_ = parser.parse_args()
        main(args_)
    except KeyboardInterrupt:
        print("\nExecution forcefully stopped.\n")
        sys.exit(1)
