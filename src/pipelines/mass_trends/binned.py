"""
Pipeline for plotting the mass trend of gas fractions.
"""
from __future__ import annotations

import logging
import logging.config
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

import library.data_acquisition as daq
import library.loading.temperature_histograms as ldt
import library.plotting.mass_trends as ptm
import library.processing as prc
from library.config import logging_config
from pipelines import base

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass
class MassTrendPipeline(base.Pipeline):
    """
    Plot the trend of gas fraction with halo mass for bin averages.

    Gas is divided into three regimes (cool, warm, hot gas) and the
    fraction of each plotted against halo mass (virial mass).
    """

    mass_bin_edges: NDArray
    normalize: bool
    cool_bins: int
    warm_bins: int

    def __post_init__(self):
        # set up logging
        log = logging_config.get_logging_config("INFO")
        logging.config.dictConfig(log)

    def run(self) -> int:
        """
        Run the pipeline to produce the mass trend plots.

        Steps:

        0. Create required directories
        1. Acquire halo mass and IDs
        2. Load gas data for every halo, and determine datapoints
           (gas fraction of cool, warm, and hot gas vs. halo mass)
        3. Bin data points in mass bins
        4. Determine running average and percentiles
        5. Plot points and mean/percentiles

        :return: Exit code, zero signifies success, non-zero exit code
            signifies an error occured. Exceptions will be raised
            normally, resulting in execution interruption.
        """
        # Step 1: acquire halo data
        fields = [self.config.mass_field, "GroupMassType"]
        halo_data = daq.halos.get_halo_properties(
            self.config.base_path, self.config.snap_num, fields=fields
        )

        # Step 2: get bin mask
        mass_bin_mask = prc.statistics.sort_masses_into_bins(
            halo_data[self.config.mass_field], self.mass_bin_edges
        )

        # Step 3: find average gas mass per bin
        n_mass_bins = len(self.mass_bin_edges) - 1
        avg_gas_mass = []
        std_gas_mass = []
        for gas_masses in prc.statistics.bin_quantitiy(
                halo_data["GroupMassType"],
                mass_bin_mask,
                n_mass_bins,
        ):
            avg_gas_mass.append(np.average(gas_masses))
            std_gas_mass.append(np.std(gas_masses))

        # Step 4: load the mean, median and errors
        mean, median, perc = ldt.load_histogram_data(
            self.paths["data_dir"] / f"{self.paths['data_file_stem']}.npz",
            (n_mass_bins, 50)
        )

        # Step 5: for every mass bin, find the gas mass fractions
        # Shape (2, 2, M): Two weight types (fraction and mass), two
        # values each (mean, median) and M values for M mass bins.
        logging.info("Gathering gas fraction data.")
        cool_gas = np.zeros((2, 2, n_mass_bins))
        warm_gas = np.zeros((2, 2, n_mass_bins))
        hot_gas = np.zeros((2, 2, n_mass_bins))
        for i in range(n_mass_bins):
            # add together all gas
            hot_edge = self.cool_bins + self.warm_bins
            # means
            cool_gas[0][0][i] = np.sum(mean[i][0:self.cool_bins])
            warm_gas[0][0][i] = np.sum(mean[i][self.cool_bins:hot_edge])
            hot_gas[0][0][i] = np.sum(mean[i][hot_edge:])
            # median
            cool_gas[0][1][i] = np.sum(median[i][0:self.cool_bins])
            warm_gas[0][1][i] = np.sum(median[i][self.cool_bins:hot_edge])
            hot_gas[0][1][i] = np.sum(median[i][hot_edge:])

            # Estimate the same values for the gas by multiplying the gas
            # fraction with the average gas mass per bin:
            # means
            cool_gas[1][0][i] = cool_gas[0][0][i] * avg_gas_mass[i]
            warm_gas[1][0][i] = warm_gas[0][0][i] * avg_gas_mass[i]
            hot_gas[1][0][i] = hot_gas[0][0][i] * avg_gas_mass[i]
            # median
            cool_gas[1][1][i] = cool_gas[0][1][i] * avg_gas_mass[i]
            warm_gas[1][1][i] = warm_gas[0][1][i] * avg_gas_mass[i]
            hot_gas[1][1][i] = hot_gas[0][1][i] * avg_gas_mass[i]

        # Step 6: Find the error
        cool_gas_err = np.ones((2, 2, n_mass_bins))
        warm_gas_err = np.ones((2, 2, n_mass_bins))
        hot_gas_err = np.ones((2, 2, n_mass_bins))

        # Step 7: plot
        logging.info("Plotting mass trend plot.")
        avg_halo_masses = []
        std_halo_masses = []
        for halo_masses in prc.statistics.bin_quantitiy(
                halo_data[self.config.mass_field],
                mass_bin_mask,
                n_mass_bins,
        ):
            avg_halo_masses.append(np.average(halo_masses))
            std_halo_masses.append(np.std(halo_masses))

        f, _ = ptm.plot_gas_mass_trends(
            np.log10(avg_halo_masses),
            cool_gas,
            warm_gas,
            hot_gas,
            cool_gas_err,
            warm_gas_err,
            hot_gas_err
        )
        # f, _ = ptm.plot_gas_mass_trends_individuals(
        #     np.log10(avg_halo_masses),
        #     np.array(std_halo_masses) / np.array(avg_halo_masses) / np.log(10),
        #     cool_gas[:, 0, :],
        #     warm_gas[:, 0, :],
        #     hot_gas[:, 0, :],
        #     cool_gas_err[:, 0, :],
        #     warm_gas_err[:, 0, :],
        #     hot_gas_err[:, 0, :]
        # )

        # save figure
        filename = f"{self.paths['figures_file_stem']}.pdf"
        filepath = Path(self.paths["figures_dir"])
        if not filepath.exists():
            logging.info("Creating missing figures directory.")
            filepath.mkdir(parents=True)
        f.savefig(filepath / filename, bbox_inches="tight")
        return 0
