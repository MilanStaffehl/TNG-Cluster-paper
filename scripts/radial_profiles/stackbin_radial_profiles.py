import argparse
import logging.config
import sys
from pathlib import Path

root_dir = Path(__file__).parents[2].resolve()
sys.path.insert(0, str(root_dir / "src"))

from library import scriptparse
from library.config import config, logging_config
from pipelines.radial_profiles.stacks_binned import (
    StackDensityProfilesByVelocityPipeline,
    StackDensityProfilesCombinedPipeline,
    StackProfilesBinnedPipeline,
)


def main(args: argparse.Namespace) -> None:
    """Create stacks of TNG300 and TNG Cluster clusters"""
    # sim data
    sim = "TNG-Cluster"  # set arbitrarily, not used

    # config
    cfg = config.get_default_config(sim)

    # logging
    log_level = scriptparse.parse_verbosity(args)
    log_cfg = logging_config.get_logging_config(log_level)
    logging.config.dictConfig(log_cfg)

    # paths
    if args.figurespath:
        figure_path = Path(args.figurepath)
    else:
        figure_path = cfg.figures_home / "radial_profiles"
    if args.datapath:
        data_path = Path(args.datapath)
    else:
        data_path = cfg.data_home / "radial_profiles" / "individuals"

    core = "_core" if args.core_only else ""
    ab = "_absolute_dist" if not args.normalize else ""
    file_data = {
        "figures_dir": figure_path.resolve(),
        "data_dir": data_path.resolve(),
        "figures_file_stem": f"radial_profiles_{args.what}{core}{ab}_clusters",
        "data_file_stem": f"radial_profiles_{args.what}{core}{ab}_clusters",
    }

    pipeline_config = {
        "config": cfg,
        "paths": file_data,
        "processes": -1,
        "to_file": True,
        "no_plots": False,
        "fig_ext": args.fig_ext,
        "log": args.log,
        "what": args.what,
        "method": args.method,
        "core_only": args.core_only,
        "normalize": args.normalize,
    }
    if args.combined and args.what == "density":
        pipeline = StackDensityProfilesCombinedPipeline(**pipeline_config)
    elif args.split:
        pipeline = StackDensityProfilesByVelocityPipeline(**pipeline_config)
    else:
        pipeline = StackProfilesBinnedPipeline(**pipeline_config)
    pipeline.run()


if __name__ == "__main__":
    parser = scriptparse.BaseScriptParser(
        prog=f"python {Path(__file__).name}",
        description=(
            "Stack individual radial profiles of all clusters in TNG300-1 and "
            "TNG Cluster, binned by mass."
        ),
    )
    parser.remove_argument("sim")
    parser.remove_argument("processes")
    parser.remove_argument("to_file")
    parser.remove_argument("from_file")
    parser.remove_argument("no_plots")
    parser.add_argument(
        "-w",
        "--what",
        help=(
            "What type of radial profile to plot: temperature or density. Defaults "
            "to temperature."
        ),
        dest="what",
        type=str,
        default="temperature",
        choices=["temperature", "density"],
    )
    parser.add_argument(
        "-m",
        "--method",
        help=(
            "How to stack the histograms, i.e. what statistical method to use "
            "for the stacking. Options are mean (with standard deviation) or "
            "median (with 16th and 84th percentile as error). Defaults to "
            "mean."
        ),
        dest="method",
        type=str,
        default="mean",
        choices=["mean", "median"],
    )
    parser.add_argument(
        "--log",
        help="Plot the figures in log scale instead of linear scale.",
        action="store_true",
        dest="log",
    )
    parser.add_argument(
        "-cc",
        "--cluster-core",
        help=(
            "Plot the core region of the cluster only. This will restrict the "
            "radial range of the plot to around 50 kpc physical size. Might "
            "not work with the -c or -b flags."
        ),
        action="store_true",
        dest="core_only",
    )
    parser.add_argument(
        "-a",
        "--absolute-distances",
        help=(
            "Instead of loading the data where halocentric distances are "
            "normalized to the virial radius, use the data where distance is "
            "measured in kpc. Might not work with the -c or -b flags."
        ),
        dest="normalize",
        action="store_false",
    )
    exclusive_group = parser.add_mutually_exclusive_group(required=False)
    exclusive_group.add_argument(
        "-c",
        "--combined",
        help=(
            "Combine median and mean lines into one plot without error "
            "regions in the density plot. Has no effect when `-w temperature` "
            "is set. When using this option, the -m flag has no effect."
        ),
        dest="combined",
        action="store_true",
    )
    exclusive_group.add_argument(
        "-b",
        "--split-by-velocity",
        help=(
            "Split the density profiles by velocity instead of plotting the "
            "total profile. Only works for density, has no effect when used "
            "with temperature. When used, the -m flag has no effect."
        ),
        dest="split",
        action="store_true",
    )

    # parse arguments
    try:
        args = parser.parse_args()
        main(args)
    except KeyboardInterrupt:
        print("Execution forcefully stopped.")
        sys.exit(1)
