import argparse
import sys
from pathlib import Path

root_dir = Path(__file__).parents[2].resolve()
sys.path.insert(0, str(root_dir / "src"))

from library import scriptparse

# from pipelines.>>YOUR.MODULE


def main(args: argparse.Namespace) -> None:
    """>>DESCRIPTION"""

    # type flag, subdirectories, and other config changes
    type_flag = ">>TYPE_FLAG"

    pipeline_config = scriptparse.startup(
        args,
        ">>MILESTONE",
        type_flag,
        figures_subdirectory=".",
        data_subdirectory=".",
        suppress_sim_name_in_files=False,
    )

    # update pipeline config here

    # select and build pipeline
    pipeline = None  # >> FILL IN
    sys.exit(pipeline.run())


if __name__ == "__main__":
    parser = scriptparse.BaseScriptParser(
        prog=f"python {Path(__file__).name}",
        description=(">>DESCRIPTION OF SCRIPT PURPOSE"),
        allowed_sims=["TNG300", "TNG-Cluster"],
    )
    # >>remove unnecessary args

    # >>add new args

    # parse arguments
    try:
        args_ = parser.parse_args()
        main(args_)
    except KeyboardInterrupt:
        print("\nExecution forcefully stopped.\n")
        sys.exit(1)
