import argparse
import logging
import sys
from pathlib import Path

root_dir = Path(__file__).parents[2].resolve()
sys.path.insert(0, str(root_dir / "src"))

from library import scriptparse
from library.config import config
from pipelines.mass_trends.temperatures import (
    FromFilePipeline,
    IndividualsMassTrendPipeline,
)


def main(args: argparse.Namespace) -> None:
    """Create plot of gas mass trends for individual halos"""
    # config
    try:
        cfg = config.get_default_config(args.sim)
    except config.InvalidSimulationNameError:
        logging.fatal(f"Unsupported simulation: {args.sim}")

    # temperature divisions
    if args.normalize:
        temperature_divs = [-100.0, -2.0, -1.0, 100.0]
    else:
        temperature_divs = [0.0, 4.5, 5.5, 10.0]

    # whether to use median or mean
    if args.average:
        statistics = "mean"
    else:
        statistics = "median"

    # file name type flag
    if args.normalize:
        type_flag = "normalized"
    else:
        type_flag = "standard"
    type_flag = f"{type_flag}_{statistics}"
    if args.running_median:
        type_flag = f"{type_flag}_rm"

    # paths
    file_data = scriptparse.assemble_path_dict(
        "mass_trends",
        cfg,
        type_flag,
        args.normalize,
        args.figurespath,
        args.datapath,
    )

    pipeline_config = {
        "config": cfg,
        "paths": file_data,
        "processes": args.processes,
        "fig_ext": args.extension,
        "mass_bin_edges": [1e8, 1e9, 1e10, 1e11, 1e12, 1e13, 1e14, 1e15],
        "temperature_divisions": temperature_divs,
        "normalize": args.normalize,
        "statistic_method": statistics,
        "running_median": args.running_median,
        "quiet": args.quiet,
        "to_file": args.to_file,
        "no_plots": args.no_plots,
    }
    if args.from_file:
        pipeline = FromFilePipeline(**pipeline_config)
    else:
        pipeline = IndividualsMassTrendPipeline(**pipeline_config)
    pipeline.run()


if __name__ == "__main__":
    parser = scriptparse.BaseScriptParser(
        prog=f"python {Path(__file__).name}",
        description="Plot mass trends of gas of halos in TNG",
    )
    parser.add_argument(
        "-n",
        "--normalize-temperatures",
        help="Normalize temperatures to virial temperature",
        dest="normalize",
        action="store_true",
    )
    parser.add_argument(
        "-a",
        "--use-average",
        help="Plot averages instead of medians in the plot",
        dest="average",
        action="store_true",
    )
    parser.add_argument(
        "-r",
        "--running-median",
        help=(
            "Plot a continuous median with confidence region instead of "
            "binned data points. Also works for averages."
        ),
        dest="running_median",
        action="store_true",
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
