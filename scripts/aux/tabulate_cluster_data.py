import argparse
import logging.config
import sys
from pathlib import Path

root_dir = Path(__file__).parents[2].resolve()
sys.path.insert(0, str(root_dir / "src"))

from library import scriptparse
from library.config import config, logging_config
from pipelines.tabulate_cluster_data import TabulateClusterDataPipeline


def main(args: argparse.Namespace) -> None:
    """Write data for TNG300-1 clusters to file"""
    # config
    cfg = config.get_default_config("TNG300-1")

    # logging
    log_level = scriptparse.parse_verbosity(args)
    log_cfg = logging_config.get_logging_config(log_level)
    logging.config.dictConfig(log_cfg)
    logging.addLevelName(18, "MEMORY")  # custom level

    # paths
    figure_path = cfg.figures_home / "radial_profiles"
    if args.datapath:
        data_path = Path(args.datapath)
    else:
        data_path = cfg.data_home / "radial_profiles" / "individuals"

    file_data = {
        "figures_dir": figure_path.resolve(),
        "data_dir": data_path.resolve(),
        "figures_file_stem": None,
        "data_file_stem": None,
    }

    pipeline_config = {
        "config": cfg,
        "paths": file_data,
        "processes": args.processes,
        "to_file": True,
        "no_plots": False,
        "fig_ext": "png",
        "forbid_tree": args.forbid_tree,
        "force_tree": args.force_tree,
    }
    pipeline = TabulateClusterDataPipeline(**pipeline_config)
    sys.exit(pipeline.run())


if __name__ == "__main__":
    parser = scriptparse.BaseScriptParser(
        prog=f"python {Path(__file__).name}",
        description="Tabulate data for TNG300-1 clusters."
    )
    parser.remove_argument("sim")
    parser.remove_argument("to_file")
    parser.remove_argument("from_file")
    parser.remove_argument("no_plots")
    parser.remove_argument("fig_ext")
    parser.remove_argument("figurespath")
    exclusive_group = parser.add_mutually_exclusive_group(required=False)
    exclusive_group.add_argument(
        "--force-tree",
        help="Force the creation and querying of a KDTree.",
        dest="force_tree",
        action="store_true",
    )
    exclusive_group.add_argument(
        "--forbid-tree",
        help="Forbid the creation of a KDTree.",
        dest="forbid_tree",
        action="store_true",
    )

    # parse arguments
    try:
        args = parser.parse_args()
        main(args)
    except KeyboardInterrupt:
        print("Execution forcefully stopped.")
        sys.exit(1)
