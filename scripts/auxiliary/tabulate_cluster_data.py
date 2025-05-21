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
    cfg = config.get_default_config(args.sim)

    # logging
    log_level = scriptparse.parse_verbosity(args)
    log_cfg = logging_config.get_logging_config(log_level)
    logging.config.dictConfig(log_cfg)
    logging.addLevelName(18, "MEMORY")  # custom level

    # paths
    figure_path = cfg.figures_home / "radial_profiles"
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
        "no_plots": True,
        "fig_ext": "png",
        "forbid_tree": args.forbid_tree,
        "force_tree": args.force_tree,
    }
    pipeline = TabulateClusterDataPipeline(**pipeline_config)
    sys.exit(pipeline.run())


DESCRIPTION = """Tabulate particle data for TNG300-1 clusters.

Script identifies all particles around halos in TNG300-1 which have a
mass M_200c of more than 10^14 solar masses and saves their indices to
file. It additionally then saves properties of the particles to file as
well (velocities, temperatures, temperature regime flag, distance to
cluster center, particle mass).

These files are saved under the data home directory specified in the
config file, with each property being assigned its own subdirectory
(e.g. particle indices are saved under a `particle_indices` subdir).

If particle indices already exist, for example because they have been
previously saved to file by the radial profile scripts, the process of
identifying and tabulating particle data is sped up considerably. If
they have not previously been saved, the indices must first be found by
creating a KDTree of all particles in the TNG300-1 volume. Consequently,
this will take considerable time and memory.
"""

if __name__ == "__main__":
    parser = scriptparse.BaseScriptParser(
        prog=f"python {Path(__file__).name}",
        description=DESCRIPTION,
        allowed_sims=["TNG300", "TNG100"],
        requires_parallel=True,
        required_memory=900,
    )
    parser.remove_argument("to_file")
    parser.remove_argument("from_file")
    parser.remove_argument("no_plots")
    parser.remove_argument("fig_ext")
    parser.remove_argument("figurespath")
    parser.remove_argument("datapath")
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
