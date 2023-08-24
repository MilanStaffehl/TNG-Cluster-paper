"""
Processor for plotting temperature distributions.
"""
from __future__ import annotations

import multiprocessing as mp
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

    from numpy.typing import ArrayLike, NDArray


class TemperatureDistributionProcessor(base_processor.BaseProcessor):
    """
    Provides an interface to plot the temperature distribution of halos.

    Instances of this class can separately load halo data, binned by
    mass, as well as calculate and store data on cell temperatures for
    these halos. It also provides methods to plot the distribution of
    temperatures within these halos, both for individual halos as well
    as for stacked halos from a single mass bin.
    """

    # range of temperatures to plot in log10 K
    temperature_range: ClassVar[tuple[float]] = (3.0, 8.0)

    def __init__(
        self,
        sim: str,
        logger: logging.Logger,
        n_temperature_bins: int,
        mass_bins: list[float],
        weight: str = "frac",
    ) -> None:
        """
        :param sim: Simulation name.
        :param logger: Logger for the instance.
        :param n_temperature_bins: Number of temperature bins.
        :param mass_bins: Edges of mass bins.
        :param weight: Weight for the histograms, defaults to "frac"
        """
        super().__init__(sim, logger, n_temperature_bins)
        self.weight = weight
        self.mass_bins = mass_bins
        self.n_mass_bins = len(mass_bins) - 1
        # create attributes for data
        self.indices = None
        self.masses = None
        self.radii = None
        self.bin_masker = None  # set by get_mask
        self.histograms_mean = None  # stacked histogram mean per mass bin
        self.histograms_median = None  # stacked hist median per mass bin
        self.histograms_percentiles = None  # perentiles per bin
        self.virial_temperatures = None

    def plot_data(
        self,
        bin_num: int,
        suffix: str = "",
        log: bool = True,
        plot_vir_temp: bool = True,
    ) -> None:
        """
        Plot the distribution of temperatures for all halos of the mass bin.

        Plots a histogram using the data of all halos in the specified
        mass bin. The mass bin must be given as an array index, i.e.
        starting from zero.

        :param bin_num: mass bin index, starting from zero
        :param suffix: suffix appended to output file name, defaults to
            an empty string (no suffix)
        :param log: whether to plot the histograms in logarithmic scale
            on the y-axis, defaults to True
        :param plot_vir_temp: whether to overplot the range of virial
            temperatures possible in the mass bin, defaults to True
        :return: None
        """
        if any([x is None for x in (self.histograms_mean,
                                    self.histograms_median,
                                    self.histograms_percentiles)]):
            self.logger.error("Data is not loaded yet!")
            return

        self.logger.info(f"Plotting temperature hist for mass bin {bin_num}.")
        fig, axes = plt.subplots(figsize=(5, 4))
        fig.set_tight_layout(True)
        axes.set_title(
            r"$M_{200c}$: "
            rf"${np.log10(self.mass_bins[bin_num])} < \log \ M_\odot "
            rf"< {np.log10(self.mass_bins[bin_num + 1])}$"
        )
        labelsize = 12
        axes.set_xlabel("Gas temperature [log K]", fontsize=labelsize)
        if self.weight == "frac":
            axes.set_ylabel("Average gas mass fraction", fontsize=labelsize)
        else:
            axes.set_ylabel(
                r"Average gas mass per cell [$M_\odot$]", fontsize=labelsize
            )

        # calculate bin positions
        _, bins = np.histogram(
            np.array([0]), bins=self.len_data, range=self.temperature_range
        )
        centers = (bins[:-1] + bins[1:]) / 2

        # plot data
        facecolor = "lightblue" if self.weight == "frac" else "pink"
        plot_config = {
            "histtype": "stepfilled",
            "facecolor": facecolor,
            "edgecolor": "black",
            "log": log,
        }
        # hack: produce exactly one entry for every bin, but weight it
        # by the histogram bar length, to achieve a "fake" bar plot
        axes.hist(
            centers,
            bins=bins,
            range=self.temperature_range,
            weights=self.histograms_mean[bin_num],
            **plot_config
        )
        # plot error bars
        error_config = {
            "marker": "x",
            "linestyle": "none",
            "ecolor": "dimgrey",
            "color": "dimgrey",
            "alpha": 0.8,
            "capsize": 2.0,
        }
        axes.errorbar(
            centers,
            self.histograms_median[bin_num],
            yerr=self._get_errorbars(bin_num),
            **error_config
        )

        # overplot virial temperatures, if desired and possible:
        if plot_vir_temp and self.virial_temperatures is None:
            self.logger.warning(
                "Virial temperatures have not been calculated, skipping "
                "overplotting of virial temperatures!"
            )
            plot_vir_temp = False
        if plot_vir_temp:
            self._overplot_virial_temperatures(axes, bin_num)

        # save figure
        filename = f"temperature_hist_{bin_num}{suffix}.pdf"
        sim = self.config.sim.replace("-", "_")
        fig.savefig(
            self.config.figures_home / "001" / sim / filename,
            bbox_inches="tight"
        )

    def load_data(self, filepath: str | Path) -> None:
        """
        Load stacked (averaged) histogram data from file.

        The file needs to be a numpy .npz archive, as saved by the method
        ``stack_bins``. The resulting NpzFile instance must have keys
        'hist_mean' and 'hist_std'. For both arrays, the first axis
        must match in length the number of mass bins and the second axis
        must match the number of histogram bins ``self.len_data``.

        The loaded data is placed into the ``histograms`` and
        ``histograms_percentiles`` attributes respectively.

        :param file: file name of the numpy data file
        :return: None
        """
        self.logger.info("Loading saved histogram data from file.")
        if not isinstance(filepath, Path):
            filepath = Path(filepath)

        if not filepath.is_file():
            self.logger.error(
                f"The given file {str(filepath)} is not a valid file."
            )
            return

        # attempt to load the data
        with np.load(filepath) as hist_data:
            hist_mean = hist_data["hist_mean"]
            hist_median = hist_data["hist_median"]
            hist_perc = hist_data["hist_percentiles"]

        # verify data:
        if not hist_median.shape == (self.n_mass_bins, self.len_data):
            self.logger.error(
                f"Loaded histogram data does not match expected data in "
                f"shape:\nExpected shape: {(self.n_mass_bins, self.len_data)}, "
                f"received shape: {hist_mean.shape}"
            )
            return
        elif not hist_mean.shape == hist_median.shape:
            self.logger.error(
                f"The histogram mean array doe not match the median array in "
                f"shape: mean has shape {hist_mean.shape} but the median has "
                f"shape {hist_median.shape}. Median has correct shape, so the "
                f"mean data must have been saved wrong or is corrupted."
            )
        elif not hist_median.shape[0] == hist_perc.shape[
                0] or not hist_median.shape[1] == hist_perc.shape[2]:
            self.logger.error(
                f"Shape of histogram median is different from shape of "
                f"histogram percentiles:\nMedian shape: {hist_mean.shape}, "
                f"percentiles shape: {hist_perc.shape}\n"
                f"The medians have the expected shape; the percentilesmust "
                f"have been saved wrong or are corrupted."
            )
            return
        else:
            self.histograms_mean = hist_mean
            self.histograms_median = hist_median
            self.histograms_percentiles = hist_perc
            self.logger.info("Successfully loaded data!")

    def load_virial_temperatures(self, np_file: str | Path) -> None:
        """
        Load virial temperature data from file.

        The file needs to be a numpy .npy file, as saved by the method
        ``get_virial_temperature``. The loaded data is placed into the
        ``virial_temperature`` attribute for use in plotting.

        :param file: file name of the numpy data file
        :return: None
        """
        self.logger.info("Loading saved virial temperature data from file.")
        if not isinstance(np_file, Path):
            np_file = Path(np_file)

        if not np_file.is_file():
            self.logger.error(
                f"The given file {str(np_file)} is not a valid file."
            )
            return

        # attempt to load the data
        self.virial_temperatures = np.load(np_file)
        self.logger.info("Successfully loaded virial temperatures.")

    def _get_auxilary_data(
        self,
        processes: int,
        quiet: bool,
        *,
        virial_temperatures: bool = True,
        to_file: bool = False,
        suffix: str = ""
    ) -> None:
        """
        Load auxilary data: bin mask, virial temperatures.

        :param processes: Number of processes to use. To calculate the
            data sequentially, set to zero.
        :param quiet: Whether to suppress status reports. Useful when
            redirecting program output to file.
        :param virial_temperatures: Whether virial temperatures need to
            be calculated. Keyword only argument.
        :param to_file: whether to write the resulting virial temperatures
            to file. Keyword only argument.
        :param suffix: Suffix to append to the file name of the virial
            temperature data file. Keyword only argument.
        """
        # create a mask array for the mass bins
        self.bin_masker = np.digitize(self.masses, self.mass_bins)
        # calculate virial temperatures
        if virial_temperatures:
            if processes > 0:
                self._get_virial_temperatures_multiprocessed(
                    processes=processes, to_file=to_file, suffix=suffix
                )
            else:
                self._get_virial_temperatures_sequentially(
                    quiet=quiet, to_file=to_file, suffix=suffix
                )

    def _get_errorbars(self, mass_bin: int) -> NDArray:
        """
        Return the histogram error bars for the percentiles.

        The method calculates, from the 18th and 84th percentile, the
        length of the error bars for the given mass bin and returns them
        as an array of shape (2, N), which can be immediately passed to
        the ``yerr`` argument of the errorbar plot function of pyplot.

        :param mass_bin: Bin index of the mass bin to plot
        :return: Shape (2, N) array of error bar sizes, with the first
            row being the lower and the second row being the upper error
            bar lengths.
        """
        median = self.histograms_median[mass_bin]
        percentiles = self.histograms_percentiles[mass_bin]
        lower_ebars = median - percentiles[0]
        upper_ebars = percentiles[1] - median
        return np.array([lower_ebars, upper_ebars])

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
        self.indices = np.indices([num_halos], sparse=True)[0]
        self.masses = (
            halo_data[self.config.mass_field] * 1e10 / constants.HUBBLE
        )
        self.radii = halo_data[self.config.radius_field] / constants.HUBBLE
        self.logger.info("Finished loading halo masses & radii.")

    def _get_virial_temperatures_multiprocessed(
        self,
        processes: int = 16,
        to_file: bool = True,
        suffix: str = ""
    ) -> None:
        """
        Calculate the virial temperature of all halos using multiprocessing.

        Method calculates the virial temperatures for all halos and
        assigns it to the ``virial_temperatures`` attribute. It utilises
        the common version of the virial theorem which sets K ~ T and
        assumes that the halo is virialized, i.e. there is a relation
        between potential energy U and kinetic energy K of 2U = K.

        The number of processes should be set such that the method can
        take full advantage of all available CPU cores. It will default
        to 16 processes. The number of processes should be at most equal
        to the number of CPU cores on the executing host, otherwise the
        performance will suffer. Chunksize for the processes is calculated
        automatically.

        This method will take considerable computation time. Computation
        is roughly reduced by a factor equal to the number of processes
        compared to usage of ``get_virial_temperatures_lin``, provided
        the file system is fast and does not add considerable overhead
        through file IO.

        After calculating the temperatures, the method cn optionally
        save them to .npy file.

        :param processes: number of processes to use for calculation
            with multiprocessing (i.e. number of CPU cores to use),
            defaults to 16
        :param to_file: whether to write the virial temperatures to
            numpy-readable file
        :param suffix: suffix to append to the file name
        :return: None
        """
        if self.masses is None or self.radii is None:
            self.logger.info("No data loaded yet, start loading now.")
            self.get_data()

        self.logger.info("Calculating virial temperatures.")
        # splice masses and radii together
        mass_radius_pairs = np.array([self.masses, self.radii]).transpose()
        chunksize = round(len(self.indices) / processes / 4, -2)
        self.logger.info(
            f"Starting {processes} subprocesses with chunksize {chunksize}."
        )
        with mp.Pool(processes=processes) as pool:
            results = pool.starmap(
                self._get_virial_temperatures_step,
                mass_radius_pairs,
                chunksize=int(chunksize)
            )
            pool.close()
            pool.join()

        # assign array of hist data to attribute
        self.virial_temperatures = np.array(results)
        self.logger.info("Finished calculating virial temperatures.")

        # write to file
        if to_file:
            file_name = f"virial_temperatures{suffix}.npy"
            file_path = self.config.data_home / "001" / file_name
            np.save(file_path, self.virial_temperatures)
            self.logger.info("Wrote virial temperatures to file.")

    def _get_virial_temperatures_sequentially(
        self,
        quiet: bool = False,
        to_file: bool = True,
        suffix: str = ""
    ) -> None:
        """
        Calculate the virial temperature of all halos without multiprocessing.

        Method calculates the virial temperatures for all halos and
        assigns it to the ``virial_temperatures`` attribute. It utilises
        the common version of the virial theorem which sets K ~ T and
        assumes that the halo is virialized, i.e. there is a relation
        between potential energy U and kinetic energy K of 2U = K.

        This method will take considerable computation time. For large
        simulations with many halos, it is recommended to use ``get_hists``
        instead, setting an appropriate number of subprocesses. This is
        typically much faster, provided the file system is fast.

        After calculating the temperatures, the method cn optionally
        save them to .npy file.

        :param quiet: whether to suppress writing progress report to
            stdout
        :param to_file: whether to write the virial temperatures to
            numpy-readable file
        :param suffix: suffix to append to the file name
        :return: None
        """
        if self.indices is None or self.masses is None:
            self.logger.info("No data loaded yet, start loading now.")
            self.get_data()

        self.logger.info("Calculating virial temperatures.")
        n_halos = len(self.indices)
        self.virial_temperatures = np.zeros(n_halos)
        for i, halo_id in enumerate(self.indices):
            if not quiet:
                perc = i / n_halos * 100
                print(f"Processing halo {i}/{n_halos} ({perc:.1f}%)", end="\r")
            # some halos have no radius data available
            if self.radii[halo_id] == 0:
                continue
            self.virial_temperatures[halo_id] = compute.get_virial_temperature(
                self.masses[halo_id], self.radii[halo_id]
            )
        self.logger.info("Finished calculating virial temperatures.")

        # write to file
        if to_file:
            file_name = f"virial_temperatures{suffix}.npy"
            file_path = self.config.data_home / "001" / file_name
            np.save(file_path, self.virial_temperatures)
            self.logger.info("Wrote virial temperatures to file.")

    def _get_virial_temperatures_step(
        self, mass: float, radius: float
    ) -> float:
        """
        Helper method, return virial temperature for a halo.

        This method is needed as the decorated function within the
        compute module cannot be used in multiprocessing directly.
        It also is needed to avoid division by zero when the radius
        of a given halo is zero.

        :param mass: mass of halo in solar masses
        :param radius: radius of halo in kpc
        :return: virial temperature in Kelvin
        """
        if radius == 0:
            return 0.0
        return compute.get_virial_temperature(mass, radius)

    def _overplot_virial_temperatures(
        self, axes: plt.Axes, mass_bin: int
    ) -> plt.Axes:
        """
        Overplot the range of virial temperatures onto the given axes.

        Method calculates the virial temperatures for all halos in the
        given mass bin and overplots onto the given axes object the
        range from lowest to highest virial temperature found as well as
        the mean virial temperature in the bin. The region between min
        and max virial temperature is shaded in color, to signify the
        region as the possible ranges of virial temperature.

        Returns the updated axes for convenience, but since the object
        is changed in place, re-assigning it is not necessary.

        :param axes: axes object upon which to overplot the virial
            temperature range
        :param mass_bin: mass bin index, starting from zero
        :return: the updated axes
        """
        if self.bin_masker is None:
            self.logger.debug("No halo data loaded yet, loading it now.")
            self._get_halo_data()
            self.bin_masker = np.digitize(self.masses, self.mass_bins)
        # find virial temperatures, only for current bin
        mask = np.where(self.bin_masker == mass_bin + 1, 1, 0)
        virial_temperatures = (
            ma.masked_array(self.virial_temperatures).compress(mask, axis=0)
        )

        # find min and max as well as the average temperature
        min_temp = np.min(virial_temperatures)
        max_temp = np.max(virial_temperatures)
        mean_temp = np.average(virial_temperatures)

        # overplot these into the plot
        self.logger.debug("Overplotting virial temperature region.")
        plot_config = {
            "color": "blue",
            "linewidth": 1.0,
            "alpha": 0.6,
        }
        axes.axvline(np.log10(min_temp), linestyle="solid", **plot_config)
        axes.axvline(np.log10(max_temp), linestyle="solid", **plot_config)
        axes.axvline(np.log10(mean_temp), linestyle="dashed", **plot_config)
        # shade region
        xs = np.arange(np.log10(min_temp), np.log10(max_temp), 0.01)
        fill_config = {
            "transform": axes.get_xaxis_transform(),
            "alpha": 0.1,
            "color": "blue",
        }
        axes.fill_between(xs, 0, 1, **fill_config)
        return axes

    def _post_process_data(
        self,
        _processes: int,
        _quiet: bool,
        *,
        to_file: bool = False,
        suffix: str = ""
    ) -> None:
        """
        Stack all histograms per mass bin for average histogram.

        The method will average all histograms in every mass bin and
        assign the resulting average histogram data to the ``histograms``
        attribute. It also calculates the 18th and 84th percentiles of
        the bins and assigns them to the ``histograms_percentiles``
        attribute.

        Optionally, the data can also be written to a numpy readable
        .npz archive file.

        :param to_file: whether to write the resulting array of histograms
            to file, defaults to False
        :param suffix: suffix to append to the file name
        :return: None
        """
        if self.data is None:
            self.logger.error(
                "No histogram data loaded yet. Use the 'get_data' method to "
                "load the data."
            )
            return

        self.logger.info("Start post-processing of data (stacking hists).")
        self.histograms_mean = np.zeros((self.n_mass_bins, self.len_data))
        self.histograms_median = np.zeros_like(self.histograms_mean)
        self.histograms_percentiles = np.zeros(
            (self.n_mass_bins, 2, self.len_data)
        )
        for bin_num in range(self.n_mass_bins):
            # mask histogram data
            mask = np.where(self.bin_masker == bin_num + 1, 1, 0)
            masked_hists = ma.masked_array(self.data).compress(mask, axis=0)
            # masked arrays need to be compressed into standard arrays
            halo_hists = masked_hists.compressed().reshape(masked_hists.shape)
            self.histograms_mean[bin_num] = np.nanmean(halo_hists, axis=0)
            self.histograms_median[bin_num] = np.nanmedian(halo_hists, axis=0)
            self.histograms_percentiles[bin_num] = np.nanpercentile(
                halo_hists,
                (16, 84),
                axis=0,
            )
            # diagnostics TODO: set to debug
            self.logger.info(
                f"Empty halos in mass bin {bin_num}: "
                f"{np.sum(np.any(np.isnan(halo_hists), axis=1))}"
            )

        if to_file:
            file_name = f"temperature_hists{suffix}.npz"
            file_path = self.config.data_home / "001" / file_name
            np.savez(
                file_path,
                hist_mean=self.histograms_mean,
                hist_median=self.histograms_median,
                hist_percentiles=self.histograms_percentiles,
            )

        self.logger.info("Finished post-processing data.")

    def _process_temperatures(
        self, temperatures: ArrayLike, gas_data: ArrayLike
    ) -> NDArray:
        """
        Calculate gas temperature and bin them into histogram data.

        :param gas_data: dictionary holding arrays with gas data. Must
            have keys for internal energy, electron abundance and for
            gas mass
        :return: histogram data, i.e. the binned temperature counts,
            weighted by gas mass fraction
        """
        # determine weights for hist
        if self.weight == "frac":
            total_gas_mass = np.sum(gas_data["Masses"])
            weights = gas_data["Masses"] / total_gas_mass
        else:
            weights = gas_data["Masses"]

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
        if not self.mass_bins[0] <= self.masses[halo_id] <= self.mass_bins[-1]:
            return True
        else:
            return False

    def __str__(self) -> str:
        """
        Return a string containing information on the current mass bins.

        :return: information on currently loaded mass bins
        """
        if self.indices is None or self.masses is None:
            return "No data loaded yet."
        if self.bin_masker is None:
            return "No bin mask created yet."

        ret_str = ""
        unique, counts = np.unique(self.bin_masker, return_counts=True)
        halos_per_bin = dict(zip(unique, counts))
        for i in range(self.n_mass_bins):
            ret_str += (
                f"Bin {i} [{self.mass_bins[i]:.2e}, "
                f"{self.mass_bins[i + 1]:.2e}]): "
                f"{halos_per_bin[i + 1]} halos\n"
            )
        ret_str += f"Total halos: {np.sum(counts[1:])}\n"
        return ret_str
