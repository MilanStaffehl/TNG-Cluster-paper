"""
Pipeline for creating temperature distribution plots, weighted by gas fraction.
"""
from __future__ import annotations

import logging.config
import sys
import time
from dataclasses import InitVar, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import compute
import data_acquisition as daq
import plotting as pt
import processing as prc
from config import config, logging_config

if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from numpy.typing import NDArray


@dataclass
class Pipeline:
    """
    Pipeline to creat histograms of gas temperature distribution.

    Pipeline creates histograms of the temperature distribution of gas
    in all halos of a chosen simulation and saves the plots and data to
    file.

    The pipeline only considers gas particles that belong to the FoF
    group of a halo, not the fuzz particles around it.
    """

    simulation: InitVar[str]  # init-only var, not a field
    processes: int
    mass_bin_edges: Sequence[float]
    n_temperature_bins: int
    temperature_range: tuple[float, float] = (3., 8.)
    with_virial_temperatures: bool = True
    quiet: bool = False
    to_file: bool = True
    no_plots: bool = False
    figures_dir: Path | None = None
    data_dir: Path | None = None

    def __post_init__(self, simulation: str) -> None:
        # set up logging
        log_cfg = logging_config.get_logging_config("INFO")
        logging.config.dictConfig(log_cfg)
        # set up config
        self.config = config.get_default_config(simulation)
        # update paths (not done directly, so that None is a valid arg)
        if self.figures_dir is None:
            self.figures_dir = Path("temperature_distribution/histograms/")
        if self.data_dir is None:
            self.data_dir = Path("temperature_distributions/")

    def run(self) -> int:
        """
        Run the pipeline to produce histogram plots.

        :return: Exit code, zero signifies success, non-zero exit code
            signifies an error occured. Exceptions will be raised
            normally, resulting in execution interruption.
        """
        # Step 0: create directories if needed
        if self.to_file:
            data_path = self.config.data_home / self.data_dir
            if not data_path.exists():
                logging.info(
                    f"Creating missing data directory {str(data_path)}."
                )
                data_path.mkdir(parents=True)
        # Step 1: acquire halo data
        fields = [self.config.mass_field, self.config.radius_field]
        halo_data = daq.halos.get_halo_properties(
            self.config.base_path, self.config.snap_num, fields=fields
        )
        # Step 2: get bin mask
        mass_bin_mask = prc.statistics.sort_masses_into_bins(
            halo_data[self.config.mass_field], self.mass_bin_edges
        )
        # Step 3: acquire virial temperatures
        logging.info("Calculating virial temperatures.")
        if self.with_virial_temperatures:
            if self.processes > 0:
                virial_temperatures = prc.parallelization.process_halo_data_starmap(
                    compute.get_virial_temperature,
                    self.processes,
                    halo_data[self.config.mass_field],
                    halo_data[self.config.radius_field],
                )
            else:
                virial_temperatures = prc.sequential.process_halo_data_multiargs(
                    compute.get_virial_temperature,
                    (self.n_temperature_bins, ),
                    halo_data[self.config.mass_field],
                    halo_data[self.config.radius_field],
                    quiet=self.quiet,
                )
        logging.info("Finished calculating virial temperatures.")
        if self.to_file:
            logging.info("Writing virial temperatures to file.")
            filename = f"virial_temperatures_{self.config.sim_path}.npy"
            np.save(data_path / filename, virial_temperatures)
        # Step 4: get primary data - histograms for every halo
        begin = time.time()
        logging.info("Calculating temperature histograms for all halos.")
        callback = self._get_callback(halo_data[self.config.mass_field])
        if self.processes > 0:
            hists = prc.parallelization.process_halo_data_parallelized(
                callback,
                halo_data["IDs"],
                self.processes,
            )
        else:
            hists = prc.sequential.process_halo_data_sequentially(
                callback,
                halo_data["IDs"],
                (self.n_temperature_bins, ),
                quiet=self.quiet,
            )
        # Step 5: post-processing - stack histograms per mass bin
        mean, median, perc = prc.statistics.stack_histograms_per_mass_bin(
            hists, len(self.mass_bin_edges) - 1, mass_bin_mask
        )
        if self.to_file:
            logging.info("Writing histogram data to file.")
            filename = f"temperature_hists_frac_{self.config.sim_path}.npz"
            np.savez(
                data_path / filename,
                hist_mean=mean,
                hist_median=median,
                hist_percentiles=perc,
            )
        end = time.time()
        # get time spent on computation
        time_diff = end - begin
        time_fmt = time.strftime('%H:%M:%S', time.gmtime(time_diff))
        logging.info(f"Spent {time_fmt} hours on execution.")
        # Step 6: plot the data
        if self.no_plots:
            return 0
        for i in range(len(self.mass_bin_edges) - 1):
            error_bars = pt.temperature_histograms.get_errorbar_lengths(
                median[i], perc[i]
            )
            f, a = pt.temperature_histograms.plot_temperature_distribution(
                mean[i],
                median[i],
                error_bars,
                (self.mass_bin_edges[i], self.mass_bin_edges[i + 1]),
                self.temperature_range,
                "lightblue",
                "Gas temperature [log K]",
                "Gas mass fraction",
            )
            if self.with_virial_temperatures:
                pt.temperature_histograms.overplot_virial_temperatures(
                    f, a, virial_temperatures, i, mass_bin_mask
                )
            # save figure
            self._save_figure(f, i)
        return 0

    def _get_callback(self, masses: NDArray) -> Callable[[int], int]:
        """
        Return a callable that calculates temperature histograms.

        Since the functions for multiprocessing and sequential processing
        expect a Callable that takes as input only the halo ID and returns
        the histogram as array, the corresponding functions must be
        bundled together. Additionally, some attributes of this class
        must be supplied to these functions. This helper method builds
        such a Callable by concatenating the required functions and
        supplying vars to them where needed.

        :return: Callable that takes halo ID as argument and returns
            the temperature distribution histogram data for that halo.
        """

        # Skip condition: skip halos with masses outside the considered
        # mass range
        def skip_condition(halo_id: int) -> bool:
            m = masses[halo_id]
            return not self.mass_bin_edges[0] <= m <= self.mass_bin_edges[-1]

        # the actual callback
        def callback_func(halo_id: int) -> NDArray:
            gas_data = daq.gas.get_halo_temperatures(
                halo_id,
                self.config.base_path,
                self.config.snap_num,
                additional_fields=None,
                skip_condition=skip_condition,
            )
            if gas_data["count"] == 0:
                fallback = np.empty(self.n_temperature_bins)
                fallback.fill(np.nan)
                return fallback
            hist = prc.gas_temperatures.get_temperature_distribution_histogram(
                gas_data,
                "frac",
                self.n_temperature_bins,
                self.temperature_range,
            )
            return hist

        # return the callable
        return callback_func

    def _save_figure(self, figure: Figure, mass_bin: int) -> None:
        """
        Save the given figure to file.

        Method will attempt to iteratively create directories that do not
        exist to save the file under the given path.

        :param figure: The figure to save.
        :param mass_bin: The index of the mass bin for the figure.
        """
        path = self.config.figures_home / self.figures_dir
        filename = (
            f"temperature_hist_frac_{self.config.sim_path}_{mass_bin}.pdf"
        )
        if not path.exists():
            path.mkdir(parents=True)
        figure.savefig(str(path / filename), bbox_inches="tight")
