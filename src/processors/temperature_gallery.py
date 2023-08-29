"""
Processor for plotting temperature distributions of individual halos.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

import illustris_python as il
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma

import compute
import constants
from processors import base_processor

if TYPE_CHECKING:
    import logging

    from numpy.typing import NDArray


class TemperatureDistributionGalleryProcessor(base_processor.BaseProcessor):
    """
    Provides an interface to plot a gallery of temperature distributions.

    The plots in the gallery are the temperature distribution histograms
    of individual halos. For every mass bin, a number of halos is chosen
    either randomly or by ID and their individual temperature distribution
    is plotted as a histogram. The histogams are then arranged into a
    gallery of plots.
    """

    # range of temperatures to plot in log10 K
    temperature_range: ClassVar[tuple[float]] = (3.0, 8.0)
    # number of plots displayed per gallery (i.e. per mass bin)
    plots_per_bin: ClassVar[int] = 10

    def __init__(
        self,
        sim: str,
        logger: logging.Logger,
        data_length: int,
        mass_bins: list[float],
    ) -> None:
        """
        :param sim: Simulation name.
        :param logger: Logger for the instance.
        :param n_temperature_bins: Number of temperature bins.
        :param mass_bins: Edges of mass bins.
        """
        super().__init__(sim, logger, data_length)
        self.mass_bins = mass_bins
        self.n_mass_bins = len(mass_bins) - 1
        # attributes for data
        self.masses = None
        self.radii = None
        self.indices = None
        self.virial_temperatures = None

    def plot_data(self, mass_bin: int, output: Path | str) -> None:
        """
        Plot a gallery of temperature distributions for the given mass bin.

        The function creates a gallery of temperature distribution plots
        for all selected halos in the given mass bin and saves it to file.

        :param mass_bin: mass bin index, starting from zero
        :param output: The file path and file name for the produced plot
        """
        self.logger.info(f"Plotting gallery for mass bin {mass_bin}.")
        fig, axes = plt.subplots(figsize=(8, 10), ncols=2, nrows=int(self.plots_per_bin / 2))
        fig.set_tight_layout(True)
        axes = axes.flatten()  # allows simple iteration
        drawn_plots = 0
        for i, halo_id in enumerate(self.indices[mass_bin]):
            if np.any(np.isnan(self.data[mass_bin][i])):
                self.logger.debug(
                    f"Skipped halo {halo_id} due to being empty."
                )
                continue  # halo had no gas
            if drawn_plots == self.plots_per_bin:
                break  # all subplots have been populated.
            # axes config
            axes[drawn_plots].set_xlabel("Gas temperature [log K]")
            axes[drawn_plots].set_ylabel("Gas mass fraction [dex]")
            axes[drawn_plots].set_ylim(1e-5, 1)
            axes[drawn_plots].set_title(f"Halo ID: {halo_id}")
            # calculate bin positions
            _, bins = np.histogram(
                np.array([0]), bins=self.len_data, range=self.temperature_range
            )
            centers = (bins[:-1] + bins[1:]) / 2

            # plot data
            plot_config = {
                "histtype": "stepfilled",
                "facecolor": "lightblue",
                "edgecolor": "black",
                "log": True,
            }
            # hack: produce exactly one entry for every bin, but weight it
            # by the histogram bar length, to achieve a "fake" bar plot
            axes[drawn_plots].hist(
                centers,
                bins=bins,
                range=self.temperature_range,
                weights=self.data[mass_bin][i],
                **plot_config
            )
            # overplot virial temperature
            line_config = {
                "color": "blue",
                "linewidth": 1.0,
                "alpha": 0.6,
                "linestyle": "dashed",
            }
            axes[drawn_plots].axvline(
                np.log10(self.virial_temperatures[mass_bin][i]), **line_config
            )
            drawn_plots += 1  # increment counter

        # save figure
        fig.savefig(output, bbox_inches="tight")

    def load_data(self, filepath: str | Path) -> None:
        """
        Load data for gallery plots from file.

        The file needs to be a numpy .npz archive, as saved by the method
        ``_post_process_data``. The resulting NpzFile instance must have
        keys 'masses', 'radii', 'indices', 'virial_temperatures' and
        'hists'.
        The first axis of the arrays must match in length the expected
        length, defined by two times the number of bins times the number
        of plots per gallery.

        The loaded data is placed into the respective attributes.

        :param file: file name of the numpy data file
        :return: None
        """
        self.logger.info("Loading saved gallery data from file.")
        if not isinstance(filepath, Path):
            filepath = Path(filepath)

        if not filepath.is_file():
            self.logger.error(
                f"The given file {str(filepath)} is not a valid file."
            )
            return

        # attempt to load the data
        with np.load(filepath) as data:
            masses = data["masses"]
            radii = data["radii"]
            indices = data["indices"]
            virial_temperatures = data["virial_temperatures"]
            hists = data["hists"]

        attrs = [masses, radii, indices, virial_temperatures]
        expected_shape = (self.n_mass_bins, 2 * self.plots_per_bin)
        if not all([x.shape == expected_shape for x in attrs]):
            self.logger.error(
                "Some of the loaded data does not have the expected "
                "number of entries. Data could not be loaded."
            )
            return
        elif not hists.shape[:-1] == expected_shape:
            self.logger.error(
                f"The histogram data does not provide the expected number of "
                f"histrograms: Expected {expected_shape} hists, but got "
                f"{hists.shape} hists instead. Aborting loading of data."
            )
            return
        else:
            self.masses = masses
            self.radii = radii
            self.indices = indices
            self.virial_temperatures = virial_temperatures
            self.data = hists
            self.logger.info("Successfully loaded data.")

    def _get_auxilary_data(self, processes: int, quiet: bool) -> None:
        """
        Calculate virial temperatures for the selected halos.

        :param processes: Stub
        :param quiet: Stub
        """
        self.logger.info("Calculating virial temperatures.")
        self.virial_temperatures = compute.get_virial_temperature(
            self.masses, self.radii
        )
        self.logger.info("Successfully calculated virial temperatures.")

    def _get_halo_data(self) -> None:
        """
        Load and bin the halos from the simulation, save binned masses.

        Method loads the halo data from the simulation and bins all halos
        by mass. IDs of the halos are binned as well and saved in attrs
        as well.

        :return: None
        """
        self.logger.info("Loading halo masses & radii.")
        halo_data = il.groupcat.loadHalos(
            self.config.base_path,
            self.config.snap_num,
            fields=[self.config.mass_field, self.config.radius_field],
        )
        num_halos = len(halo_data[self.config.mass_field])
        indices = np.indices([num_halos], sparse=True)[0]
        masses = (halo_data[self.config.mass_field] * 1e10 / constants.HUBBLE)
        radii = halo_data[self.config.radius_field] / constants.HUBBLE
        mass_bin_mask = np.digitize(masses, self.mass_bins)

        # for every mass bin, select twice as many halos as needed (to
        # have a backup when empty halos are selected by accident)
        self.logger.info("Selecting subset of halos for gallery.")
        self.indices = np.zeros(
            self.n_mass_bins * 2 * self.plots_per_bin, dtype=int
        )
        for bin_num in range(self.n_mass_bins):
            mask = np.where(mass_bin_mask == bin_num + 1, 1, 0)
            masked_indices = ma.masked_array(indices).compress(mask)
            masked_indices = masked_indices.compressed()
            n = 2 * self.plots_per_bin  # number of halos to select per bin
            # choose entries randomly
            rng = np.random.default_rng()
            self.indices[bin_num * n:(bin_num + 1) * n] = rng.choice(
                masked_indices, size=n, replace=False
            )
        # assign masses and radii to attributes
        self.masses = masses[self.indices]
        self.radii = radii[self.indices]
        self.logger.info("Finished loading and selecting halo masses & radii.")

    def _post_process_data(
        self,
        processes: int,
        quiet: bool,
        to_file: bool = False,
        output: Path | str | None = None
    ) -> None:
        """
        Reshape the obtained data and save it to file, if desired.

        :param processes: Stub
        :param quiet: Stub
        :param to_file: Whether to save the data to file, defaults to False
        :param output: File path and name for the data file. Needs to be
            specified if ``to_file`` is True. Defaults to None.
        """
        self.indices = self.indices.reshape(self.n_mass_bins, -1)
        self.masses = self.masses.reshape(self.n_mass_bins, -1)
        self.radii = self.radii.reshape(self.n_mass_bins, -1)
        self.virial_temperatures = self.virial_temperatures.reshape(
            self.n_mass_bins, -1
        )
        self.data = self.data.reshape(self.n_mass_bins, -1, self.len_data)

        if to_file:
            np.savez(
                output,
                masses=self.masses,
                indices=self.indices,
                radii=self.radii,
                virial_temperatures=self.virial_temperatures,
                hists=self.data,
            )
            self.logger.info("Saved data to file.")
        else:
            self.logger.debug("Skipping saving data to file.")

    def _process_temperatures(
        self,
        halo_id: int,
        temperatures: NDArray,
        gas_data: dict[str, NDArray]
    ) -> NDArray:
        """
        Calculate gas temperature and bin them into histogram data.

        :param gas_data: dictionary holding arrays with gas data. Must
            have keys for internal energy, electron abundance and for
            gas mass
        :return: histogram data, i.e. the binned temperature counts,
            weighted by gas mass fraction
        """
        total_gas_mass = np.sum(gas_data["Masses"])
        weights = gas_data["Masses"] / total_gas_mass
        # generate and assign hist data
        hist, _ = np.histogram(
            np.log10(temperatures),
            bins=self.len_data,
            range=self.temperature_range,
            weights=weights,
        )
        return hist

    def _skip_halo(self, halo_id: int) -> bool:
        """
        Return True if the halo has a mass outside of the mass bin range.

        :param halo_id: ID of the halo to check
        :return: Whether the calclation of halo gas temperature can be
            skipped.
        """
        idx = np.where(self.indices == halo_id)[0][0]
        if not self.mass_bins[0] <= self.masses[idx] <= self.mass_bins[-1]:
            return True
        else:
            return False
