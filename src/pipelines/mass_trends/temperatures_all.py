"""
Pipelines for creating gas fraction mass dependence plots.
"""
from __future__ import annotations

import logging.config
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, ClassVar, Literal, Sequence

import numpy as np

from library import compute
from library.data_acquisition import gas_daq, halos_daq
from library.loading import load_mass_trends
from library.plotting import common, plot_mass_trends
from library.processing import gas_temperatures, parallelization, sequential, statistics
from pipelines import base

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass
class IndividualsMassTrendPipeline(base.Pipeline):
    """
    Pipeline to create plots of gas temperature fractions with halo mass.

    Pipeline creates plots of the relationship between the gas fraction
    and gas mass with halo mass for three temperture regimes: cool,
    warm and hot gas. The datapoints for every halo are plotted as a
    2D histogram, and additionally, the median is plotted for every mass
    bin.

    Pipeline only considers the gas belonging to a halo, not the fuzz
    around the FoF.
    """

    mass_bin_edges: Sequence[float]
    temperature_divisions: tuple[float, float, float, float]
    normalize: bool = False
    statistic_method: Literal["mean", "median"] = "median"
    running_median: bool = False

    data_points_hist_bins: ClassVar[int] = 60

    def run(self) -> int:
        """
        Run the pipeline to produce mass trend plots.

        Steps:

        0. Create directories
        1. Acquire halo data
        2. Get bin mask
        3. Acquire virial temperatures (if needed)
        4. Get gas fraction and masses per regime of every halo
        5. Average fractions per mass bin
        6. Compute running average
        7. Plot

        :return: Exit code, zero signifies success, non-zero exit code
            signifies an error occured. Exceptions will be raised
            normally, resulting in execution interruption.
        """
        # Step 0: create directories if needed
        self._create_directories()

        # Step 1: acquire halo data
        fields = [self.config.mass_field, self.config.radius_field]
        halo_data = halos_daq.get_halo_properties(
            self.config.base_path, self.config.snap_num, fields=fields
        )

        # Step 2: get bin mask
        mass_bin_mask = np.digitize(
            halo_data[self.config.mass_field], self.mass_bin_edges
        )

        # Step 3: acquire virial temperatures
        if self.normalize:
            virial_temperatures = self._get_virial_temperatures(halo_data)
        else:
            logging.info(
                "Skipping virial temperature calculation (not required)."
            )

        # Step 4: get primary data - histograms for every halo
        begin = time.time()
        logging.info("Calculating gas fraction and mass for all halos.")
        if self.normalize:
            norm = virial_temperatures
        else:
            norm = np.ones(len(halo_data["IDs"]))
        callback = self._get_callback(halo_data[self.config.mass_field], norm)
        if self.processes > 0:
            points = parallelization.process_data_parallelized(
                callback,
                halo_data["IDs"],
                self.processes,
            )
        else:
            points = sequential.process_data_sequentially(
                callback,
                halo_data["IDs"],
                (2, 3),
            )
        # points has shape (H, 2, 3) where the first axis is the halos,
        # second axis is the weight type and third axis is the regime,
        # i.e. cold, warm and hot gas.

        # Step 5: post-processing - average data points per mass bin
        if self.running_median:
            n_mass_bins = self.data_points_hist_bins  # one median for every bin
        else:
            n_mass_bins = len(self.mass_bin_edges) - 1
        if self.statistic_method == "mean":
            getter = statistics.get_binned_averages
        elif self.statistic_method == "median":
            getter = statistics.get_binned_medians
        else:
            logging.error(
                f"Unrecognized request for statistics function: "
                f"{self.statistic_method}."
            )
            return 1
        masses = getter(
            halo_data[self.config.mass_field], mass_bin_mask, n_mass_bins
        )
        # returned arrays have shape (3, M) where M is number of mass bins
        cold_by_frac = getter(points[:, 0, 0], mass_bin_mask, n_mass_bins)
        warm_by_frac = getter(points[:, 0, 1], mass_bin_mask, n_mass_bins)
        hot_by_frac = getter(points[:, 0, 2], mass_bin_mask, n_mass_bins)
        cold_by_mass = getter(points[:, 1, 0], mass_bin_mask, n_mass_bins)
        warm_by_mass = getter(points[:, 1, 1], mass_bin_mask, n_mass_bins)
        hot_by_mass = getter(points[:, 1, 2], mass_bin_mask, n_mass_bins)
        if self.to_file:
            logging.info("Writing mass trend data to file.")
            filename = f"{self.paths['data_file_stem']}.npz"
            np.savez(
                self.paths["data_dir"] / filename,
                gas_data_points=points,
                avg_masses=masses,
                cold_by_frac=cold_by_frac,
                warm_by_frac=warm_by_frac,
                hot_by_frac=hot_by_frac,
                cold_by_mass=cold_by_mass,
                warm_by_mass=warm_by_mass,
                hot_by_mass=hot_by_mass,
            )
        end = time.time()
        # get time spent on computation
        time_diff = end - begin
        time_fmt = time.strftime('%H:%M:%S', time.gmtime(time_diff))
        logging.info(f"Spent {time_fmt} hours on execution.")

        # Step 6: plot the data
        if self.no_plots:
            return 0
        self._plot(
            halo_data[self.config.mass_field],
            points,
            masses,
            cold_by_frac,
            warm_by_frac,
            hot_by_frac,
            cold_by_mass,
            warm_by_mass,
            hot_by_mass
        )
        return 0

    def _get_callback(self, masses: NDArray,
                      normalization: NDArray) -> Callable[[int], int]:
        """
        Return a callable that calculates gas fractions and masses.

        Since the functions for multiprocessing and sequential processing
        expect a Callable that takes as input only the halo ID and returns
        the data points as array, the corresponding functions must be
        bundled together. Additionally, some attributes of this class
        must be supplied to these functions. This helper method builds
        such a Callable by concatenating the required functions and
        supplying vars to them where needed.

        :param masses: The array of halo masses.
        :param normalization: An array of normalizations for the
            temperature values. Must be of the same length as ``masses``.
            To use non-normalized temperatures, give an array of ones.
        :return: Callable that takes halo ID as argument and returns
            the temperature distribution histogram data for that halo.
        """

        # Skip condition: skip halos with masses outside the considered
        # mass range
        def skip_condition(halo_id: int) -> bool:
            m = masses[halo_id]
            return m < 1e8

        # the actual callback
        def callback_func(halo_id: int) -> NDArray:
            gas_data = gas_daq.get_halo_temperatures(
                halo_id,
                self.config.base_path,
                self.config.snap_num,
                additional_fields=None,
                skip_condition=skip_condition,
            )
            if gas_data["count"] == 0:
                fallback = np.empty((2, 3))
                fallback.fill(np.nan)
                return fallback
            temperature_range = (
                self.temperature_divisions[0], self.temperature_divisions[-1]
            )
            frac = gas_temperatures.get_temperature_distribution_histogram(
                gas_data,
                "frac",
                self.temperature_divisions,
                temperature_range,
                normalization[halo_id],
            )
            mass = gas_temperatures.get_temperature_distribution_histogram(
                gas_data,
                "mass",
                self.temperature_divisions,
                temperature_range,
                normalization[halo_id],
            )
            return np.array([frac, mass])

        # return the callable
        return callback_func

    def _get_virial_temperatures(
        self, halo_data: dict[str, NDArray]
    ) -> NDArray | None:
        """
        Return an array of virial temperatures for all halos in the sim.

        :param halo_data: The dictionary holding masses and radii of the
            halos, required for calculation.
        :return: An array of virial temperatures for every halo in Kelvin
            if virial temperatures need to be plotted, None otherwise.
        """
        logging.info("Calculating virial temperatures.")
        if self.processes > 0:
            virial_temperatures = parallelization.process_data_starmap(
                compute.get_virial_temperature,
                self.processes,
                halo_data[self.config.mass_field],
                halo_data[self.config.radius_field],
            )
        else:
            virial_temperatures = sequential.process_data_multiargs(
                compute.get_virial_temperature,
                tuple(),
                halo_data[self.config.mass_field],
                halo_data[self.config.radius_field],
            )
        logging.info("Finished calculating virial temperatures.")
        # optionally write data to file
        if self.to_file:
            logging.info("Writing virial temperatures to file.")
            filename = f"{self.paths['virial_temp_file_stem']}.npy"
            np.save(self.paths["data_dir"] / filename, virial_temperatures)
        return virial_temperatures

    def _plot(
        self,
        halo_masses: NDArray,
        gas_fraction_data: NDArray,
        average_bin_masses: NDArray,
        cold_by_frac: NDArray,
        warm_by_frac: NDArray,
        hot_by_frac: NDArray,
        cold_by_mass: NDArray,
        warm_by_mass: NDArray,
        hot_by_mass: NDArray,
    ) -> None:
        """
        Save plot of mass trend to file.

        All gas data arrays of shape (3, M) are expected to have as the
        first row the values and as second and third row the standard
        deviation or asymmetric errors (e.g. percentiles).

        :param halo_masses: Array of all halo masses, shape (M, ).
        :param gas_fraction_data: Data point for gas fraction data.
        :param average_bin_masses: Average mass and standard deviation
            in every mass bin.
        :param cold_by_frac: Array of cold gas fraction, shape (3, M)
        :param warm_by_frac: Array of warm gas fraction, shape (3, M)
        :param hot_by_frac: Array of hot gas fraction, shape (3, M)
        :param cold_by_mass: Array of cool gas mass, shape (3, M)
        :param warm_by_mass: Array of warm gas mass, shape (3, M)
        :param hot_by_mass: Array of hot gas mass, shape (3, M)
        :return: None
        """
        logging.info("Plotting mass trend data.")
        # construct error arrays as expected by the plotting function
        mass_err_left = (
            average_bin_masses[1] / average_bin_masses[0] / np.log(10)
        )
        mass_err_right = (
            average_bin_masses[2] / average_bin_masses[0] / np.log(10)
        )
        binned_halo_masses = np.log10(average_bin_masses[0])
        binned_halo_masses_err = np.array([mass_err_left, mass_err_right])

        # Plot the individual halo dta points
        f, a = plot_mass_trends.plot_gas_mass_trends_individuals(
            halo_masses=np.log10(halo_masses),
            gas_data=gas_fraction_data,
        )

        # overplot the mean/median data points
        if self.running_median:
            overplotter = common.plot_curve_with_error_region
        else:
            overplotter = common.overplot_datapoints
        overplotter(
            binned_halo_masses,
            cold_by_frac[0],
            binned_halo_masses_err,
            np.array([cold_by_frac[1], cold_by_frac[2]]),
            axes=a[0][0],
        )
        overplotter(
            binned_halo_masses,
            cold_by_mass[0],
            binned_halo_masses_err,
            np.array([cold_by_mass[1], cold_by_mass[2]]),
            axes=a[0][1],
        )
        overplotter(
            binned_halo_masses,
            warm_by_frac[0],
            binned_halo_masses_err,
            np.array([warm_by_frac[1], warm_by_frac[2]]),
            axes=a[1][0],
        )
        overplotter(
            binned_halo_masses,
            warm_by_mass[0],
            binned_halo_masses_err,
            np.array([warm_by_mass[1], warm_by_mass[2]]),
            axes=a[1][1],
        )
        overplotter(
            binned_halo_masses,
            hot_by_frac[0],
            binned_halo_masses_err,
            np.array([hot_by_frac[1], hot_by_frac[2]]),
            axes=a[2][0],
        )
        overplotter(
            binned_halo_masses,
            hot_by_mass[0],
            binned_halo_masses_err,
            np.array([hot_by_mass[1], hot_by_mass[2]]),
            axes=a[2][1],
        )

        # add text labels for temperature range
        a[0][1].text(
            8.5,
            3e13,
            r"Cool ($T \leq 10^{4.5}\,\rm K$)",
            color=common.temperature_colors_rgb["cool"],
            fontsize="small",
        )
        a[1][1].text(
            8.5,
            3e13,
            r"Warm",
            color=common.temperature_colors_rgb["warm"],
            fontsize="small",
        )
        a[1][1].text(
            8.5,
            5e12,
            r"($10^{4.5}\,\rm K < T \leq 10^{5.5}\,\rm K$)",
            color=common.temperature_colors_rgb["warm"],
            fontsize=8,
        )
        a[2][1].text(
            8.5,
            3e13,
            r"Hot ($> 10^{5.5}\,\rm K$)",
            color=common.temperature_colors_rgb["hot"],
            fontsize="small",
        )

        # save figure
        self._save_fig(f, tight_layout=False)
        logging.info("Successfully saved mass trend plot to file!")
        return


class FromFilePipeline(IndividualsMassTrendPipeline):
    """
    Pipeline to creat histograms of gas temperature distribution from file.

    Pipeline creates histograms of the temperature distribution of gas
    in all halos of a chosen simulation and saves the plots and data to
    file. It takes the data previously saved to file by a run of an
    instance of the parent class :class:`Pipeline`.

    If any of the required data is missing, a FileNotFound exception is
    raised and the execution terminated.
    """

    def run(self) -> int:
        """
        Run the pipeline to load data and produce plots.

        :raises FileNotFoundError: When any of the required data files
            are missing.
        :return: Exit code, zero signifies success, all other values
            mean an error occurred. Exceptions will be raised normally,
            interrupting the execution.
        """
        # Step 0: verify all required data exists
        status = self._verify_directories()
        if status > 0:
            return status

        if self.no_plots:
            logging.warning(
                "Was asked to load data without plotting it. This is pretty "
                "pointless and probably not what you wanted."
            )
            return 0

        # Step 1: acquire halo data
        fields = [self.config.mass_field, self.config.radius_field]
        halo_data = halos_daq.get_halo_properties(
            self.config.base_path, self.config.snap_num, fields=fields
        )
        # Step 2: load virial temperatures
        data_dir = self.paths["data_dir"]
        # Step 3: get primary data - histograms for every halo
        if self.running_median:
            n_mass_bins = self.data_points_hist_bins
        else:
            n_mass_bins = len(self.mass_bin_edges) - 1
        data_file = data_dir / f"{self.paths['data_file_stem']}.npz"
        data = load_mass_trends.load_mass_trend_data(data_file, n_mass_bins)
        if data is None:
            logging.error("Could not load mass trend data from file.")
            return 1
        # Step 4: plot the data
        self._plot(halo_data[self.config.mass_field], *data)
        return 0
