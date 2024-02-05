import argparse
import logging
import sys
from pathlib import Path

root_dir = Path(__file__).parents[2].resolve()
sys.path.insert(0, str(root_dir / "src"))

from library import scriptparse
from library.config import config
from pipelines.temperature_distribution.histogram_galleries import (
    FromFilePipeline,
    GalleriesPipeline,
)


def main(args: argparse.Namespace) -> None:
    """Create histograms of temperature distribution"""
    # config
    try:
        cfg = config.get_default_config(args.sim)
    except config.InvalidSimulationNameError:
        logging.fatal(f"Unsupported simulation: {args.sim}")

    # type flag
    type_flag = "gallery"
    if args.normalize:
        type_flag = f"norm_{type_flag}"

    # paths
    file_data = scriptparse.assemble_path_dict(
        "temperature_hist",
        cfg,
        type_flag,
        False,
        args.figurespath,
        args.datapath,
        "galleries",
    )

    pipeline_config = {
        "config": cfg,
        "paths": file_data,
        "processes": 1,
        "fig_ext": args.extension,
        "plots_per_bin": args.plots_per_bin,
        "mass_bin_edges": [1e8, 1e9, 1e10, 1e11, 1e12, 1e13, 1e14, 1e15],
        "n_temperature_bins": args.bins,
        "temperature_range": (3., 8.),
        "normalize": args.normalize,
        "quiet": args.quiet,
        "no_plots": args.no_plots,
        "to_file": args.to_file,
    }
    if args.from_file:
        gallery_plotter = FromFilePipeline(**pipeline_config)
    else:
        gallery_plotter = GalleriesPipeline(**pipeline_config)
    gallery_plotter.run()


if __name__ == "__main__":
    parser = scriptparse.BaseScriptParser(
        prog=f"python {Path(__file__).name}",
        description="Plot temperature distribution galleries of halos in TNG",
    )
    parser.add_argument(
        "--plots-per-bin",
        help=(
            "The number of halos to select and plot per mass bin. Default "
            "is 10"
        ),
        dest="plots_per_bin",
        type=int,
        default=10,
        metavar="NUMBER",
    )
    parser.add_argument(
        "-n",
        "--normalize-temperatures",
        help="Normalize temperatures to virial temperature",
        dest="normalize",
        action="store_true",
    )
    parser.add_argument(
        "-b",
        "--bins",
        help="The number of temperature bins, defaults to 50",
        dest="bins",
        type=int,
        default=50,
        metavar="NUMBER",
    )
    parser.remove_argument("processes")

    # parse arguments
    try:
        args = parser.parse_args()
        main(args)
    except KeyboardInterrupt:
        print("Execution forcefully stopped.")
        sys.exit(1)
