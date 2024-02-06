import argparse
import sys
from pathlib import Path

root_dir = Path(__file__).parents[2].resolve()
sys.path.insert(0, str(root_dir / "src"))

from library import scriptparse
from pipelines.temperature_distribution.histogram_galleries import (
    FromFilePipeline,
    GalleriesPipeline,
)


def main(args: argparse.Namespace) -> None:
    """Create histograms of temperature distribution"""
    # type flag
    type_flag = "gallery"
    if args.normalize:
        type_flag = f"norm_{type_flag}"

    pipeline_config = scriptparse.startup(
        args,
        "temperature_distribution",
        type_flag,
        with_virial_temperatures=False,
        figures_subdirectory="galleries",
    )

    pipeline_config.update(
        {
            "plots_per_bin": args.plots_per_bin,
            "mass_bin_edges": [1e8, 1e9, 1e10, 1e11, 1e12, 1e13, 1e14, 1e15],
            "n_temperature_bins": args.bins,
            "temperature_range": (3., 8.),
            "normalize": args.normalize,
        }
    )
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
