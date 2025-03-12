import argparse
import sys
from pathlib import Path

root_dir = Path(__file__).parents[2].resolve()
sys.path.insert(0, str(root_dir / "src"))

from library import scriptparse
from pipelines.mass_trends.temperatures_all import (
    FromFilePipeline,
    IndividualsMassTrendPipeline,
)


def main(args: argparse.Namespace) -> None:
    """Create plot of gas mass trends for individual halos"""
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

    pipeline_config = scriptparse.startup(
        args,
        "mass_trends",
        type_flag,
        with_virial_temperatures=args.normalize,
    )

    pipeline_config.update(
        {
            "mass_bin_edges": [1e8, 1e9, 1e10, 1e11, 1e12, 1e13, 1e14, 1e15],
            "temperature_divisions": temperature_divs,
            "normalize": args.normalize,
            "statistic_method": statistics,
            "running_median": args.running_median,
        }
    )
    if args.from_file:
        pipeline = FromFilePipeline(**pipeline_config)
    else:
        pipeline = IndividualsMassTrendPipeline(**pipeline_config)
    sys.exit(pipeline.run())


DESCRIPTION = """Plot mass trends of gas of halos in a TNG simulation.

Script plots the relationship of gas mass and gas mass fraction of gas
in three phases (cool, warm, and hot gas, divided at 10^4.5 K and 10^5.5 K
respectively) with the mass M_200c of the corresponding halos that host
the gas. Only gas that is part of the halo FoF is considered. The script
creates a figure consisting of six panels, with three rows showing the
trends for hot, warm, and cool gas respectively, and the columns showing
gas mass fraction and total gas mass respectively. All halos down to a
mass of 10^8 solar masses are considered.

To indicate general trends, the median gas mass/gas fraction is marked
in bins of 1 dex halo mass, with error bars showing the 16th and 84th
percentiles in both halo mass and gas mass/fraction. Optionally, this can
also be replaced by a running median line with an error region, or the
median can be replaced by a mean.

Note that empty mass bins can lead to exceptions, i.e. simulations without
cluster-mass halos might cause failure of the script. This is chiefly the
case for certain lower resolution runs of TNG50.
"""

if __name__ == "__main__":
    parser = scriptparse.BaseScriptParser(
        prog=f"python {Path(__file__).name}",
        description=DESCRIPTION,
        required_memory=220,
        requires_parallel=True,
    )
    parser.add_argument(
        "-n",
        "--normalize-temperatures",
        help=(
            "Normalize temperatures to virial temperature and apply a "
            "division between the temperature regimes based on virial "
            "temperature rather than using absolute temperatures."
        ),
        dest="normalize",
        action="store_true",
    )
    parser.add_argument(
        "-a",
        "--use-average",
        help="Plot averages instead of medians in the plot.",
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
