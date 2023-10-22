"""
Pipelines for creating galleries of temperature distribution histograms.
"""
from __future__ import annotations

import logging.config
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

import numpy as np

import library.data_acquisition as daq
import library.loading.temperature_histograms as ldt
import library.plotting.temperature_histograms as ptt
import library.processing as prc
from library import compute
from library.config import logging_config
from pipelines import base

if TYPE_CHECKING:
    # Reason for noqa: https://github.com/PyCQA/pyflakes/issues/648
    from numpy.typing import NDArray

    from library.config import config  # noqa: F401


@dataclass
class GalleriesPipeline(base.Pipeline):
    """
    Pipeline to create gallery plots of temperature distribution.

    Pipeline selects the given number of halos for every mass bin and
    plots their temperature distribution histogram in a gallery plot,
    that is, a grid of individual histograms for these halos.
    """

    plots_per_bin: int
    mass_bin_edges: Sequence[float]
    n_temperature_bins: int
    temperature_range: tuple[float, float] = (3., 8.)
    normalize: bool = False
    quiet: bool = False
    no_plots: bool = False
    to_file: bool = True

    def __post_init__(self) -> None:
        # set up logging
        log_cfg = logging_config.get_logging_config("INFO")
        logging.config.dictConfig(log_cfg)

    def run(self) -> int:
        """
        Run the pipeline to produce gallery plots.

        :return: Exit code, zero signifies success, non-zero exit code
            signifies an error occured. Exceptions will be raised
            normally, resulting in execution interruption.
        """
        # Step 0: create directories, if needed
        self._create_directories()

        # Step 1: acquire halo data
        fields = [self.config.mass_field, self.config.radius_field]
        halo_data = daq.halos.get_halo_properties(
            self.config.base_path, self.config.snap_num, fields=fields
        )
        # Step 2: select halos from every mass bin
        mass_bin_mask = prc.statistics.sort_masses_into_bins(
            halo_data[self.config.mass_field], self.mass_bin_edges
        )
        # for every mass bin, select twice as many halos as needed (to
        # have a backup when empty halos are selected by accident)
        n_mass_bins = len(self.mass_bin_edges) - 1
        selected_halo_ids = prc.selection.select_halos_from_mass_bins(
            2 * self.plots_per_bin,
            halo_data["IDs"],
            n_mass_bins,
            mass_bin_mask,
        )
        # assign masses and radii to attributes
        selected_masses = halo_data[self.config.mass_field][selected_halo_ids]
        selected_radii = halo_data[self.config.radius_field][selected_halo_ids]
        logging.info("Finished loading and selecting halo masses & radii.")
        # Step 3: acquire virial temperatures
        virial_temperatures = prc.sequential.process_halo_data_multiargs(
            compute.get_virial_temperature,
            tuple(),
            selected_masses,
            selected_radii,
            quiet=self.quiet,
        )
        # Step 4: get temperature histograms
        logging.info("Calculating temperature histograms for all halos.")
        histograms = np.zeros(
            (len(selected_halo_ids), self.n_temperature_bins)
        )
        for i, halo_id in enumerate(selected_halo_ids):
            if not self.quiet:
                total = len(selected_halo_ids)
                perc = i / total * 100
                print(f"Processing halo {i}/{total} ({perc:.1f}%)", end="\r")
            norm = virial_temperatures[i] if self.normalize else 1
            histograms[i] = self._get_histogram(halo_id, norm)
        logging.info("Finished calculating temperature histograms.")
        # Step 5: reshape arrays
        selected_halo_ids = selected_halo_ids.reshape(n_mass_bins, -1)
        selected_masses = selected_masses.reshape(n_mass_bins, -1)
        selected_radii = selected_radii.reshape(n_mass_bins, -1)
        virial_temperatures = virial_temperatures.reshape(n_mass_bins, -1)
        histograms = histograms.reshape(
            n_mass_bins, -1, self.n_temperature_bins
        )
        # save data to file
        if self.to_file:
            filename = f"{self.paths['data_file_stem']}.npz"
            np.savez(
                self.paths["data_dir"] / filename,
                selected_masses=selected_masses,
                selected_halo_ids=selected_halo_ids,
                selected_radii=selected_radii,
                virial_temperatures=virial_temperatures,
                histograms=histograms,
            )
            logging.info("Saved data to file.")
        # Step 6: plot histograms
        if self.no_plots:
            return 0
        self._plot(selected_halo_ids, histograms, virial_temperatures)
        return 0

    def _get_histogram(self, halo_id: int, normalization: float) -> NDArray:
        """
        Return the histogram of the selected halo.

        Optionally, the temperatures may be normalized to the given value.
        Returns the historgram of the temperature distribution for the
        selected halo.

        :param halo_id: ID of the halo to process.
        :param normalization: A value by which to normalize the gas
            temperatures before creating the histogram. Usually, this
            is the virial temperature in Kelvin or 1 (no normalization).
        :return: The histogram of the temperature distribution as an
            array of length ``self.n_temperature_bins``.
        """
        gas_data = daq.gas.get_halo_temperatures(
            halo_id,
            self.config.base_path,
            self.config.snap_num,
            additional_fields=None,
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
            normalization,
        )
        return hist

    def _plot(
        self,
        halo_ids: NDArray,
        histograms: NDArray,
        virial_temperatures: NDArray,
    ) -> None:
        """
        Helper function; gathers all instructions to create plots.

        Final plots - one gallery per mass bin - will be saved to
        file.

        :param halo_ids: An array of halo IDs to plot, sorted by mass
            bin. Must have shape (M, 2P) where M is the number of mass
            bins and P is the number of plots per mass bin.
        :param histograms: The array of histogram data to plot. Must
            have shape (M, 2P, T) where T is the number of temperature
            bins.
        :param virial_temperatures: Array of virial temperatures of the
            selected halos. Must also have shape (M, 2P).
        :return: None
        """
        # labels x axis
        if self.normalize:
            xlabel = r"Gas temperature $T / T_{vir}$ [dex]"
        else:
            xlabel = "Gas temperature [log K]"
        # plot all mass bins
        for i in range(len(self.mass_bin_edges) - 1):
            f, _ = ptt.plot_temperature_distribution_gallery(
                self.plots_per_bin,
                halo_ids[i],
                histograms[i],
                virial_temperatures[i],
                self.temperature_range,
                (self.mass_bin_edges[i], self.mass_bin_edges[i + 1]),
                xlabel=xlabel,
            )
            # save figure
            filename = f"{self.paths['figures_file_stem']}_{i}.pdf"
            filepath = Path(self.paths["figures_dir"])
            if not filepath.exists():
                logging.info("Creating missing figures directory.")
                filepath.mkdir(parents=True)
            f.savefig(filepath / filename, bbox_inches="tight")


class FromFilePipeline(GalleriesPipeline):
    """
    Pipeline to creat galleries of gas temperature distribution from file.

    Pipeline creates galleries of the temperature distribution of gas
    for a selected number of halos of a chosen simulation and saves the
    plots and data to file. It takes the data previously saved to file
    by a run of an instance of the parent class :class:`Pipeline`.

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
        gallery_data_path = (
            self.paths["data_dir"] / f"{self.paths['data_file_stem']}.npz"
        )
        if not gallery_data_path.exists():
            raise FileNotFoundError(
                f"Data file {str(gallery_data_path)} does not exist."
            )
        # Step 1: Load data from file
        _m, _r, ids, vt, histograms = ldt.load_gallery_plot_data(
            gallery_data_path,
            (len(self.mass_bin_edges) - 1, 2 * self.plots_per_bin),
        )
        # Step 2: plot the data
        if self.no_plots:
            logging.warning(
                "Was asked to load data without plotting it. This is pretty "
                "pointless and probably not what you wanted."
            )
            return 0
        self._plot(ids, histograms, vt)
        return 0
