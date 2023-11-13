"""
Pipelines for creating temperature distribution plots.
"""
from __future__ import annotations

import logging.config
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Sequence

import numpy as np

import library.data_acquisition as daq
import library.loading.temperature_histograms as ldt
import library.plotting.temperature_histograms as ptt
import library.processing as prc
from library import compute
from library.plotting import util
from pipelines import base

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass
class TemperatureHistogramsPipeline(base.Pipeline):
    """
    Pipeline to creat histograms of gas temperature distribution.

    Pipeline creates histograms of the temperature distribution of gas
    in all halos of a chosen simulation and saves the plots and data to
    file.

    The pipeline only considers gas particles that belong to the FoF
    group of a halo, not the fuzz particles around it.
    """

    mass_bin_edges: Sequence[float]
    n_temperature_bins: int
    temperature_range: tuple[float, float] = (3., 8.)
    weights: str = "frac"
    normalize: bool = False
    with_virial_temperatures: bool = True
    temperature_divisions: tuple[float, float] | None = None

    def run(self) -> int:
        """
        Run the pipeline to produce histogram plots.

        Steps:

        0. Create directories
        1. Acquire halo data
        2. Get bin mask
        3. Acquire virial temperatures
        4. Get histograms of every halo
        5. Stack histograms per mass bin
        6. Plot

        :return: Exit code, zero signifies success, non-zero exit code
            signifies an error occured. Exceptions will be raised
            normally, resulting in execution interruption.
        """
        # Step 0: create directories if needed
        self._create_directories()

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
        virial_temperatures = self._get_virial_temperatures(halo_data)

        # Step 4: get primary data - histograms for every halo
        begin = time.time()
        logging.info("Calculating temperature histograms for all halos.")
        if self.normalize:
            norm = virial_temperatures
        else:
            norm = np.ones(len(halo_data["IDs"]))
        callback = self._get_callback(halo_data[self.config.mass_field], norm)
        if self.processes > 0:
            hists = prc.parallelization.process_data_parallelized(
                callback,
                halo_data["IDs"],
                self.processes,
            )
        else:
            hists = prc.sequential.process_data_sequentially(
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
            filename = f"{self.paths['data_file_stem']}.npz"
            np.savez(
                self.paths["data_dir"] / filename,
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
        self._plot(mean, median, perc, virial_temperatures, mass_bin_mask)
        return 0

    def _get_callback(self, masses: NDArray,
                      normalization: NDArray) -> Callable[[int], int]:
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
                self.weights,
                self.n_temperature_bins,
                self.temperature_range,
                normalization[halo_id],
            )
            return hist

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
        # normalization requires vt, even if with_virial_temperatures is
        # set to False
        if not self.with_virial_temperatures and not self.normalize:
            return None
        logging.info("Calculating virial temperatures.")
        if self.processes > 0:
            virial_temperatures = prc.parallelization.process_data_starmap(
                compute.get_virial_temperature,
                self.processes,
                halo_data[self.config.mass_field],
                halo_data[self.config.radius_field],
            )
        else:
            virial_temperatures = prc.sequential.process_data_multiargs(
                compute.get_virial_temperature,
                tuple(),
                halo_data[self.config.mass_field],
                halo_data[self.config.radius_field],
                quiet=self.quiet,
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
        mean: NDArray,
        median: NDArray,
        percentiles: NDArray,
        virial_temperatures: NDArray | None,
        mass_bin_mask: NDArray | None
    ) -> None:
        """
        Helper function; gathers all instructions to create plots.

        Final plots - one histogram per mass bin - will be saved to
        file.

        :param mean: The array of histogram means of shape (M, T) where
            M is the number of mass bins and T the number of temperature
            bins.
        :param median: The array of histogram medians of shape (M, T)
            where M is the number of mass bins and T the number of
            temperature bins.
        :param percentiles: The array of 16th and 84th percentile of every
            temperature bin. Must be of shape (M, 2, T) where M is the
            number of mass bins and T the number of temperature bins.
        :param virial_temperatures: Array of virial temperatures for all
            halos. Can be set to None if ``self.with_virial_temperatures``
            is False.
        :param mass_bin_mask: The array containing the mass bin number
            of every halo, i.e. the number of the mass bin into which the
            halo with the corresponding array index falls.
        """
        # labels y axis
        if self.weights == "mass":
            ylabel = r"Gas mass [$M_\odot$]"
        else:
            ylabel = "Gas mass fraction"
        # labels x axis
        if self.normalize:
            xlabel = r"Gas temperature $T / T_{vir}$ [dex]"
        else:
            xlabel = "Gas temperature [log K]"
        facecolor = "lightblue" if self.weights == "frac" else "pink"
        # plot all mass bins
        for i in range(len(self.mass_bin_edges) - 1):
            error_bars = ptt.get_errorbar_lengths(median[i], percentiles[i])
            f, a = ptt.plot_temperature_distribution(
                mean[i],
                median[i],
                error_bars,
                (self.mass_bin_edges[i], self.mass_bin_edges[i + 1]),
                self.temperature_range,
                facecolor,
                xlabel,
                ylabel,
            )
            if self.with_virial_temperatures:
                ptt.overplot_virial_temperatures(
                    f, a, virial_temperatures, i, mass_bin_mask
                )
            if self.temperature_divisions:
                ptt.overplot_temperature_divisions(
                    f, a, self.temperature_divisions
                )
            # save figure
            filename = f"{self.paths['figures_file_stem']}_{i}.pdf"
            filepath = Path(self.paths["figures_dir"])
            if not filepath.exists():
                logging.info("Creating missing figures directory.")
                filepath.mkdir(parents=True)
            f.savefig(filepath / filename, bbox_inches="tight")


class FromFilePipeline(TemperatureHistogramsPipeline):
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
        hist_data_path = (
            self.paths["data_dir"] / f"{self.paths['data_file_stem']}.npz"
        )
        vt_data_path = (
            self.paths["data_dir"]
            / f"{self.paths['virial_temp_file_stem']}.npy"
        )
        for path in [hist_data_path, vt_data_path]:
            if not path.exists():
                raise FileNotFoundError(
                    f"Data file {str(path)} does not exist."
                )
        # Step 1: acquire halo data
        fields = [self.config.mass_field, self.config.radius_field]
        halo_data = daq.halos.get_halo_properties(
            self.config.base_path, self.config.snap_num, fields=fields
        )
        # Step 2: get bin mask
        mass_bin_mask = prc.statistics.sort_masses_into_bins(
            halo_data[self.config.mass_field], self.mass_bin_edges
        )
        # Step 3: load virial temperatures
        if self.with_virial_temperatures:
            virial_temperatures = ldt.load_virial_temperatures(vt_data_path)
        else:
            virial_temperatures = None
        # Step 4: get primary data - histograms for every halo
        mean, median, perc = ldt.load_histogram_data(
            hist_data_path,
            (len(self.mass_bin_edges) - 1, self.n_temperature_bins)
        )
        # Step 6: plot the data
        if self.no_plots:
            logging.warning(
                "Was asked to load data without plotting it. This is pretty "
                "pointless and probably not what you wanted."
            )
            return 0
        self._plot(mean, median, perc, virial_temperatures, mass_bin_mask)
        return 0


class CombinedPlotsPipeline(TemperatureHistogramsPipeline):
    """
    Pipeline to create combined plot of temperature distribution.

    Pipeline will create one single plot containing all temperature
    distributions in it, without the error bars.
    """

    # method only needs to overwrite the plotting method
    def _plot(
        self,
        mean: NDArray,
        median: NDArray,
        percentiles: NDArray,
        virial_temperatures: NDArray | None,
        mass_bin_mask: NDArray | None
    ) -> None:
        """
        Plot a single figure with all mass bin histograms in it.

        :param mean: The array of histogram means of shape (M, T) where
            M is the number of mass bins and T the number of temperature
            bins.
        :param median: The array of histogram medians of shape (M, T)
            where M is the number of mass bins and T the number of
            temperature bins.
        :param percentiles: The array of 16th and 84th percentile of every
            temperature bin. Must be of shape (M, 2, T) where M is the
            number of mass bins and T the number of temperature bins.
        :param virial_temperatures: Array of virial temperatures for all
            halos. Can be set to None if ``self.with_virial_temperatures``
            is False.
        :param mass_bin_mask: The array containing the mass bin number
            of every halo, i.e. the number of the mass bin into which the
            halo with the corresponding array index falls.
        """
        # colormap choice
        colormap = "cividis"
        # labels y axis
        if self.weights == "mass":
            ylabel = r"Gas mass per cell [$M_\odot$]"
        else:
            ylabel = "Gas mass fraction"
        # labels x axis
        if self.normalize:
            xlabel = r"Gas temperature $T / T_{vir}$ [dex]"
        else:
            xlabel = "Gas temperature [log K]"

        fig, axes = ptt.plot_temperature_distributions_in_one(
            mean,
            self.temperature_range,
            self.mass_bin_edges,
            xlabel,
            ylabel,
            colormap=colormap,
        )

        # overplot virial temperatures, if desired
        if self.with_virial_temperatures:
            logging.info("Overplotting virial temperatures.")
            for mass_bin in range(len(self.mass_bin_edges) - 1):
                c = util.sample_cmap(colormap, 1 / len(mean), mass_bin)
                ptt.overplot_virial_temperatures(
                    fig, axes, virial_temperatures, mass_bin, mass_bin_mask, c
                )
        if self.temperature_divisions:
            ptt.overplot_temperature_divisions(
                fig, axes, self.temperature_divisions
            )

        # save figure
        filename = f"{self.paths['figures_file_stem']}_combined.pdf"
        filepath = Path(self.paths["figures_dir"])
        if not filepath.exists():
            logging.info("Creating missing figures directory.")
            filepath.mkdir(parents=True)
        fig.savefig(filepath / filename, bbox_inches="tight")


class CombinedPlotsFromFilePipeline(FromFilePipeline):
    """
    Plot the combined temperature distributions from file.
    """

    def run(self) -> int:
        return super().run()

    # method only needs to overwrite the plotting method
    def _plot(
        self,
        mean: NDArray,
        median: NDArray,
        percentiles: NDArray,
        virial_temperatures: NDArray | None,
        mass_bin_mask: NDArray | None
    ) -> None:
        """
        Plot a single figure with all mass bin histograms in it.

        :param mean: The array of histogram means of shape (M, T) where
            M is the number of mass bins and T the number of temperature
            bins.
        :param median: The array of histogram medians of shape (M, T)
            where M is the number of mass bins and T the number of
            temperature bins.
        :param percentiles: The array of 16th and 84th percentile of every
            temperature bin. Must be of shape (M, 2, T) where M is the
            number of mass bins and T the number of temperature bins.
        :param virial_temperatures: Array of virial temperatures for all
            halos. Can be set to None if ``self.with_virial_temperatures``
            is False.
        :param mass_bin_mask: The array containing the mass bin number
            of every halo, i.e. the number of the mass bin into which the
            halo with the corresponding array index falls.
        """
        # colormap choice
        colormap = "jet"
        # labels y axis
        if self.weights == "mass":
            ylabel = r"Gas mass per cell [$M_\odot$]"
        else:
            ylabel = "Gas mass fraction"
        # labels x axis
        if self.normalize:
            xlabel = r"Gas temperature $T / T_{vir}$ [dex]"
        else:
            xlabel = "Gas temperature [log K]"

        fig, axes = ptt.plot_temperature_distributions_in_one(
            mean,
            self.temperature_range,
            self.mass_bin_edges,
            xlabel,
            ylabel,
            colormap=colormap,
        )

        # overplot virial temperatures, if desired
        if self.with_virial_temperatures:
            logging.info("Overplotting virial temperatures.")
            for mass_bin in range(len(self.mass_bin_edges) - 1):
                c = util.sample_cmap(colormap, len(mean), mass_bin)
                ptt.overplot_virial_temperatures(
                    fig,
                    axes,
                    virial_temperatures,
                    mass_bin,
                    mass_bin_mask,
                    c,
                    True
                )
        if self.temperature_divisions:
            ptt.overplot_temperature_divisions(
                fig, axes, self.temperature_divisions
            )

        # save figure
        filename = f"{self.paths['figures_file_stem']}_combined.pdf"
        filepath = Path(self.paths["figures_dir"])
        if not filepath.exists():
            logging.info("Creating missing figures directory.")
            filepath.mkdir(parents=True)
        fig.savefig(filepath / filename, bbox_inches="tight")
