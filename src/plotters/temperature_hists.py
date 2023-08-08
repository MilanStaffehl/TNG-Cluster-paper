from __future__ import annotations

import multiprocessing as mp
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

import illustris_python as il
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
from numpy.typing import ArrayLike

import compute
import config
import constants

if TYPE_CHECKING:
    import logging


class TemperatureDistributionPlotter:
    """
    Provides an interface to plot the temperature distribution of halos.

    Instances of this class can separately load halo data, binned by
    mass, as well as calculate and store data on cell temperatures for
    these halos. It also provides methods to plot the distribution of
    temperatures within these halos, both for individual halos as well
    as for stacked halos from a single mass bin.
    """

    # number of bins in temperature histogram
    n_bins: ClassVar[int] = 50
    temperature_range: ClassVar[tuple[float]] = (3.0, 8.0)

    def __init__(
        self,
        sim: str,
        mass_bins: list[float],
        logger: logging.Logger,
        weight: str = "frac",
    ) -> None:
        self.logger = logger
        self.config = config.get_config(sim=sim)
        self.sim = sim
        self.weight = weight
        self.mass_bins = mass_bins
        self.n_mass_bins = len(mass_bins) - 1
        # create attributes for data
        self.indices = None
        self.masses = None
        self.radii = None
        self.bin_masker = None  # set by get_mask
        self.hist_data = None  # histograms of temperature
        self.histograms = None  # stacked histograms per mass bin
        self.histograms_std = None  # standard deviation of bins
        self.virial_temperatures = None

    def get_data(self) -> None:
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
        self.logger.info("Finished setting up data.")

    def get_mask(self) -> None:
        """
        Create an array of bin indices matching the halos to their bin.

        The array will hold numbers from 0 to ``self.n_mass_bins`` - 1,
        where each number corresponds to the mass bin into which the
        corresponding halo with the same array indx falls into, i.e.
        if the i-th entry of the list is equals to 1, then the i-th
        halo falls into the second mass bin (that is, it has a mass
        between ``self.bins[1]`` and ``self.bins[2]``).
        """
        if self.masses is None or self.indices is None:
            self.logger.info(
                "No masses to create mask from yet. Loading masses now."
            )
            self.get_data()
        self.bin_masker = np.digitize(self.masses, self.mass_bins)

    def get_hists(self, processes: int = 16) -> None:
        """
        Load the histogram data for every halo using multiprocessing.

        Requires that the halo mass data has already been loaded with
        ``get_data``. It loads, for every bin, the gas cells of every
        halo and computes both the gas mass fraction as well as the
        temperature of the every gas cell. It then calculates histogram
        data for every halo (for a gas mass fration vs. temperature
        histogram). The histograms are packed into a list which in turn
        is placed into a tuple of as many members as there are mass
        bins. This tuple is then assigned to the ``binned_temperature_hists``
        attribute.

        The number of processes should be set such that the method can
        take full advantage of all available CPU cores. It will default
        to 16 processes. The number of processes should be at most equal
        to the number of CPU cores on the executing host, otherwise the
        performance will suffer.

        Chunksize for the processes is calculated automatically. It
        takes into account that not all chunks of a list of halos will
        take equal amount of computation time: lower mass halos will
        typically be orders of magnitude faster. Therefore, the chunk
        size is only a quarter of the maximum possible chunk size, to
        allow processes that have finished their chunk faster to be
        assigned a new chunk instead of idling.

        This method will take considerable computation time. Computation
        is roughly reduced by a factor equal to the number of processes
        compared to usage of `get_hists_lin`, provided the file system
        is fast and does not add considerable overhead through file IO.

        :param processes: number of processes to use for calculation
            with multiprocessing (i.e. number of CPU cores to use)
        :return: None
        """
        if self.indices is None or self.masses is None:
            self.logger.info("No data loaded yet, start loading now.")
            self.get_data()
            self.get_mask()

        self.logger.info("Start processing halo data.")
        # multiprocess the entire problem
        chunksize = round(len(self.indices) / processes / 4, -2)
        self.logger.info(f"Starting subprocesses with chunksize {chunksize}.")
        with mp.Pool(processes=processes) as pool:
            results = pool.map(
                self._get_hists_step, self.indices, chunksize=int(chunksize)
            )
            pool.close()
            pool.join()
        self.logger.info("Finished processing halo data.")

        # assign array of hist data to attribute
        self.hist_data = np.array(results)

    def get_hists_lin(self, quiet: bool = False):
        """
        Load the histogram data for every halo without multiprocessing.

        Requires that the halo mass data has already been loaded with
        ``get_data``. It loads, for every bin, the gas cells of every
        halo and computes both the gas mass fraction as well as the
        temperature of the every gas cell. It then calculates histogram
        data for every halo (for a gas mass fration vs. temperature
        histogram). The histograms are packed into a list which in turn
        is placed into a tuple of as many members as there are mass
        bins. This tuple is then assigned to the ``binned_temperature_hists``
        attribute.

        This method will take considerable computation time. For large
        simulations with many halos, it is recommended to use ``get_hists``
        instead, setting an appropriate number of subprocesses. This is
        typically much faster, provided the file system is fast.

        :param quiet: whether to suppress writing progress report to
            stdout
        :return: None
        """
        if self.indices is None or self.masses is None:
            self.logger.info("No data loaded yet, start loading now.")
            self.get_data()
            self.get_mask()

        self.logger.info("Start processing halo data.")
        n_halos = len(self.indices)
        self.hist_data = np.zeros((n_halos, self.n_bins))
        for i, halo_id in enumerate(self.indices):
            if not quiet:
                perc = i / n_halos * 100
                print(
                    f"Processing halo {halo_id}/{n_halos} ({perc:.1f}%)",
                    end="\r"
                )
            # halos outside of the mass bin needn't be procesed
            if (self.masses[halo_id] < self.mass_bins[0]
                    or self.masses[halo_id] > self.mass_bins[-1]):
                continue
            self.hist_data[halo_id] = self._get_hists_step(halo_id)
        self.logger.info("Finished processing halo data.")

    def get_virial_temperatures(
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
        self.logger.info(f"Starting subprocesses with chunksize {chunksize}.")
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
            cur_dir = Path(__file__).parent.resolve()
            file_name = f"virial_temperatures{suffix}.npy"
            file_path = (cur_dir.parent.parent / "data" / "001" / file_name)
            np.save(file_path, self.virial_temperatures)
            self.logger.info("Wrote virial temperatures to file.")

    def get_virial_temperatures_lin(
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
            cur_dir = Path(__file__).parent.resolve()
            file_name = f"virial_temperatures{suffix}.npy"
            file_path = (cur_dir.parent.parent / "data" / "001" / file_name)
            np.save(file_path, self.virial_temperatures)
            self.logger.info("Wrote virial temperatures to file.")

    def stack_bins(self, to_file: bool = False, suffix: str = "") -> None:
        """
        Stack all histograms per mass bin for average histogram.

        The method will average all histograms in every mass bin and
        assign the resulting average histogram data to the ``histograms``
        attribute. It also calculates the standard deviation of the bins
        and assigns it to the ``histograms_std`` attribute.

        Optionally, the data can also be written to a numpy readable
        .npz archive file.

        :param to_file: whether to write the resulting array of histograms
            to file, defaults to False
        :param suffix: suffix to append to the file name
        :return: None
        """
        if self.hist_data is None:
            self.logger.error(
                "No histogram data loaded yet. Load data using either the "
                "'get_hists' or 'get_hists_lin' methods to load the data."
            )
            return

        self.histograms = np.zeros((self.n_mass_bins, self.n_bins))
        self.histograms_std = np.zeros((self.n_mass_bins, self.n_bins))
        for bin_num in range(self.n_mass_bins):
            # mask histogram data
            mask = np.where(self.bin_masker == bin_num + 1, 1, 0)
            halo_hists = ma.masked_array(self.hist_data).compress(mask, axis=0)
            self.histograms[bin_num] = np.average(halo_hists, axis=0)
            self.histograms_std[bin_num] = np.std(halo_hists, axis=0)

        if to_file:
            cur_dir = Path(__file__).parent.resolve()
            file_name = f"temperature_hists{suffix}.npz"
            file_path = (cur_dir.parent.parent / "data" / "001" / file_name)
            np.savez(
                file_path,
                hist_mean=self.histograms,
                hist_std=self.histograms_std,
            )

    def load_stacked_hist(self, np_file: str | Path) -> None:
        """
        Load stacked (averaged) histogram data from file.

        The file needs to be a numpy .npz archive, as saved by the method
        ``stack_bins``. The resulting NpzFile instance must have keys
        'hist_mean' and 'hist_std'. For both arrays, the first axis
        must match in length the number of mass bins and the second axis
        must match the number of histogram bins ``self.n_bins``.

        The loaded data is placed into the ``histograms`` and
        ``histograms_std`` attributes respectively.

        :param file: file name of the numpy data file
        :return: None
        """
        self.logger.info("Loading saved histogram data from file.")
        if not isinstance(np_file, Path):
            np_file = Path(np_file)

        if not np_file.is_file():
            self.logger.error(
                f"The given file {str(np_file)} is not a valid file."
            )
            return

        # attempt to load the data
        with np.load(np_file) as hist_data:
            hist_mean = hist_data["hist_mean"]
            hist_std = hist_data["hist_std"]

        # verify data:
        if not hist_mean.shape == (self.n_mass_bins, self.n_bins):
            self.logger.error(
                f"Loaded histogram data does not match expected data in "
                f"shape:\nExpected shape: {(self.n_mass_bins, self.n_bins)}, "
                f"received shape: {hist_mean.shape}"
            )
            return
        elif not hist_mean.shape == hist_std.shape:
            self.logger.error(
                f"Shape of histogram means is different from shape of "
                f"histogram standard deviations:\nMeans shape: "
                f"{hist_mean.shape}, std shape: {hist_std.shape}\n"
                f"The means have the expected shape; the standard deviations "
                f"must have been saved wrong or are corrupted."
            )
            return
        else:
            self.histograms = hist_mean
            self.histograms_std = hist_std
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

    def plot_stacked_hist(
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
        if self.histograms is None or self.histograms_std is None:
            self.stack_bins(suffix=suffix)

        self.logger.info(f"Plotting temperature hist for mass bin {bin_num}.")
        fig, axes = plt.subplots(figsize=(5, 4))
        fig.set_tight_layout(True)
        axes.set_title(
            rf"${np.log10(self.mass_bins[bin_num])} < \log \ M_\odot "
            rf"< {np.log10(self.mass_bins[bin_num + 1])}$"
        )
        axes.set_xlabel("Gas temperature [log K]")
        if self.weight == "frac":
            axes.set_ylabel("Average gas mass fraction")
        else:
            axes.set_ylabel(r"Average gas mass per cell [$M_\odot$]")

        # calculate bin positions
        _, bins = np.histogram(
            np.array([0]), bins=self.n_bins, range=self.temperature_range
        )
        centers = (bins[:-1] + bins[1:]) / 2

        # plot data
        facecolor = "lightblue" if self.weight == "frac" else "lightcoral"
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
            weights=self.histograms[bin_num],
            **plot_config
        )
        # plot error bars
        error_config = {
            "fmt": "",
            "linestyle": "none",
            "ecolor": "grey",
            "color": "grey",
            "alpha": 0.7,
            "capsize": 2.0,
        }
        axes.errorbar(
            centers,
            self.histograms[bin_num],
            yerr=self.histograms_std[bin_num],
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
        fig.savefig(
            f"./../../figures/001/temperature_hist_{bin_num}{suffix}.pdf",
            bbox_inches="tight"
        )

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
            self.get_mask()
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

    def _get_hists_step(self, halo_id: int) -> ArrayLike:
        """
        Calculate the hist data for a single halo and place it into attr.

        Calculates the temperature for all gas cells of the halo and
        from the temperature calculates the temperature distribution
        weighted by gas mas fraction as histogram data. The resulting
        histogram data - an array of shape ``(N, self.n_bins)`` is then
        assigned to its index position of the ``self.hist_data`` array,
        which must be created first by calling ``get_data``.

        Returns numpy array containing the histogram data.

        :param halo_id: ID of the halo to process
        :return: numpy array of hist data
        """
        # check if the halo needs to be loaded
        if (self.masses[halo_id] < self.mass_bins[0]
                or self.masses[halo_id] > self.mass_bins[-1]):
            return np.zeros(self.n_bins)
        # load required halo data
        gas_data = il.snapshot.loadHalo(
            self.config.base_path,
            self.config.snap_num,
            halo_id,
            partType=0,  # gas
            fields=["InternalEnergy", "ElectronAbundance", "Masses"],
        )

        # some halos do not contain gas
        if gas_data["count"] == 0:
            self.logger.debug(
                f"Halo {halo_id} contains no gas. Returning an empty hist."
            )
            return np.zeros(self.n_bins)

        # calculate hist and return it
        return self._calculate_hist_data(gas_data)

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

    def _calculate_hist_data(self, gas_data: ArrayLike) -> ArrayLike:
        """
        Calculate gas temperature and bin them into histogram data.

        :param gas_data: dictionary holding arrays with gas data. Must
            have keys for internal energy, electron abundance and for
            gas mass
        :return: histogram data, i.e. the binned temperature counts,
            weighted by gas mass fraction
        """
        temperatures = compute.get_temperature(
            gas_data["InternalEnergy"], gas_data["ElectronAbundance"]
        )
        # determine weights for hist
        if self.weight == "frac":
            total_gas_mass = np.sum(gas_data["Masses"])
            weights = gas_data["Masses"] / total_gas_mass
        else:
            weights = gas_data["Masses"]

        # generate and assign hist data
        hist, _ = np.histogram(
            np.log10(temperatures),
            bins=self.n_bins,
            range=self.temperature_range,
            weights=weights,
        )
        return hist

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
