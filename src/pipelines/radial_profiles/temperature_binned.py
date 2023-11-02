"""
Pipeline for plotting 2D-histograms of radial temperature distribution.
"""
from __future__ import annotations

import logging
import logging.config
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Sequence

import numpy as np

import library.data_acquisition as daq
import library.loading.radial_profiles as ldr
import library.plotting.radial_profiles as ptr
import library.processing as prc
from library import constants
from library.config import logging_config
from pipelines import base

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from library.config import config  # noqa: F401


@dataclass
class BinnedTemperatureProfilePipeline(base.Pipeline):
    """
    Plot a 2D histogram of temperature vs radial distance.

    Halos are binned by mass before processing.

    The radial distance is measured as the distance of every gas cell
    position w.r.t. the position coordinates of their halo, normalized
    to the virial radius R200c of the halo. Histograms of T vs R weighted
    by gas fration are created for every halo in a mass bin, then all
    histograms in a mass bin are averaged to create one final 2D hist
    per mass bin.
    """

    mass_bin_edges: Sequence[float]
    n_temperature_bins: int
    n_radial_bins: int
    temperature_range: tuple[float, float] = (3., 8.)
    radial_range: tuple[float, float] = (0., 1.5)  # units of R200c
    quiet: bool = False
    to_file: bool = True
    no_plots: bool = False

    def __post_init__(self):
        # set up logging
        log = logging_config.get_logging_config("INFO")
        logging.config.dictConfig(log)

    def run(self) -> int:
        """
        Run the pipeline to produce the 2D histograms.

        Steps:

        0. Create required directories
        1. Acquire halo data, including virial radius
        2. Get bin mask
        3. Get 2D histograms for every halo
        4. Stack histograms per mass bin, acquire running average
        5. Plot

        :return: Exit code, zero signifies success, non-zero exit code
            signifies an error occured. Exceptions will be raised
            normally, resulting in execution interruption.
        """
        # Step 0: create required directories
        self._create_directories()

        # Step 1: acquire halo data
        fields = [self.config.mass_field, self.config.radius_field, "GroupPos"]
        halo_data = daq.halos.get_halo_properties(
            self.config.base_path, self.config.snap_num, fields=fields
        )

        # Step 2: Get bin mask
        mass_bin_mask = prc.statistics.sort_masses_into_bins(
            halo_data[self.config.mass_field], self.mass_bin_edges
        )

        # Step 3: Acquire main data - 2D histograms of every halo
        begin = time.time()
        logging.info("Calculating temperature histograms for all halos.")
        callback = self._get_callback(
            halo_data[self.config.mass_field],
            halo_data["GroupPos"],
            halo_data[self.config.radius_field],
        )
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
                (self.n_radial_bins, self.n_temperature_bins),
                quiet=self.quiet,
            )

        # Step 4: Stack histograms per mass bin, get average
        n_mass_bins = len(self.mass_bin_edges) - 1
        histograms = prc.statistics.stack_2d_histograms_per_mass_bin(
            hists, n_mass_bins, mass_bin_mask
        )
        averages = np.zeros((n_mass_bins, self.n_radial_bins))
        for i, hist in enumerate(histograms):
            averages[i] = prc.statistics.get_2d_histogram_running_average(
                hist, self.temperature_range
            )
        # save data to file
        if self.to_file:
            logging.info("Writing histogram data to file.")
            filename = f"{self.paths['data_file_stem']}.npz"
            np.savez(
                self.paths["data_dir"] / filename,
                hist_mean=histograms,
                running_avg=averages,
            )
        end = time.time()
        # get time spent on computation
        time_diff = end - begin
        time_fmt = time.strftime('%H:%M:%S', time.gmtime(time_diff))
        logging.info(f"Spent {time_fmt} hours on execution.")

        # Step 5: Plot histograms
        if self.no_plots:
            return 0
        self._plot(histograms, averages)
        return 0

    def _get_callback(
        self, masses: NDArray, centers: NDArray, virial_radii: NDArray
    ) -> Callable[[int], int]:
        """
        Return a callable that calculates temperature histograms.

        Since the functions for multiprocessing and sequential processing
        expect a Callable that takes as input only the halo ID and returns
        the histogram as array, the corresponding functions must be
        bundled together. Additionally, some attributes of this class
        must be supplied to these functions. This helper method builds
        such a Callable by concatenating the required functions and
        supplying vars to them where needed.

        :param masses: The array of halo masses.
        :param centers: The position vectors of the centers of all halos
            in units of comoving kpc.
        :param virial_radii: The virial radii (R200c) in kpc of all halos.
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
                additional_fields=["Coordinates"],
                skip_condition=skip_condition,
            )
            if gas_data["count"] == 0:
                fallback = np.empty(
                    (self.n_radial_bins, self.n_temperature_bins)
                )
                fallback.fill(np.nan)
                return fallback
            # find the distance of all gas cells to halo center
            positions = gas_data["Coordinates"] / constants.HUBBLE  # unit conv
            distance_vec = (positions - centers[halo_id])
            distances = np.linalg.norm(distance_vec, axis=1)
            # append to gas_data
            gas_data["Distances"] = distances
            hist = prc.gas_temperatures.get_temperature_2d_histogram(
                gas_data,
                "Distances",
                np.array((self.radial_range, self.temperature_range)),
                self.n_temperature_bins,
                self.n_radial_bins,
                convert_units=False,
                normalization_x=virial_radii[halo_id],
            )
            return hist

        # return the callable
        return callback_func

    def _plot(self, histograms: NDArray, running_averages: NDArray) -> None:
        """
        Helper function; gathers all instructions to create plots.

        Final plots - one histogram per mass bin - will be saved to
        file.
        """
        for i in range(len(self.mass_bin_edges) - 1):
            f, a = ptr.plot_radial_temperature_profile(
                histograms[i],
                running_averages[i],
                (self.mass_bin_edges[i], self.mass_bin_edges[i + 1]),
                self.radial_range + self.temperature_range,
            )
            # save figure
            filename = f"{self.paths['figures_file_stem']}_{i}.pdf"
            filepath = Path(self.paths["figures_dir"])
            if not filepath.exists():
                logging.info("Creating missing figures directory.")
                filepath.mkdir(parents=True)
            f.savefig(filepath / filename, bbox_inches="tight")


class FromFilePipeline(BinnedTemperatureProfilePipeline):
    """
    Pipeline to create radial temperature profiles from file.

    Pipeline creates the same plots as created by the normal pipeline,
    but loads the data from file instead of recalculating it.

    If any of the required data is missing, a FileNotFound exception is
    raised and the execution terminated.
    """

    def run(self) -> int:
        """
        Run the pipeline to load data and produce radial profile plots.

        :raises FileNotFoundError: When any of the required data files
            are missing.
        :return: Exit code, zero signifies success, all other values
            mean an error occurred. Exceptions will be raised normally,
            interrupting the execution.
        """
        # Step 0: verify the required data exists
        data_path = (
            self.paths["data_dir"] / f"{self.paths['data_file_stem']}.npz"
        )
        if not data_path.exists():
            raise FileNotFoundError(
                f"Data file {str(data_path)} does not exist."
            )
        # Step 1: load the data
        histograms, averages = ldr.load_radial_profile_data(
            data_path,
            len(self.mass_bin_edges) - 1,
            self.n_radial_bins,
            self.n_temperature_bins,
        )
        # Step 3: plot the data
        if self.no_plots:
            logging.warning(
                "Was asked to load data without plotting it. This is pretty "
                "pointless and probably not what you wanted."
            )
            return 0
        self._plot(histograms, averages)
        return 0
