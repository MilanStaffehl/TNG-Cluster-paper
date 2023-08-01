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
        self, sim: str, mass_bins: list[float], logger: logging.Logger
    ) -> None:
        self.logger = logger
        self.config = config.get_config(sim=sim)
        self.sim = sim
        self.mass_bins = mass_bins
        self.n_mass_bins = len(mass_bins) - 1
        # create attributes for data
        self.indices = None
        self.masses = None
        self.bin_masker = None  # set by get_mask
        self.hist_data = None  # histograms of temp
        self.histograms = None  # stacked histograms per mass bin

    def get_data(self) -> None:
        """
        Load and bin the halos from the simulation, save binned masses.

        Method loads the halo data from the simulation and bins all halos
        by mass. IDs of the halos are binned as well and saved in attrs
        as well.

        :return: None
        """
        self.logger.info("Loading halo masses.")
        halo_masses = il.groupcat.loadHalos(
            self.config.base_path,
            self.config.snap_num,
            fields=self.config.mass_field,
        )
        num_halos = len(halo_masses)
        self.indices = np.indices([num_halos], sparse=True)[0]
        self.masses = halo_masses * 1e10 / constants.HUBBLE
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
        Load the histogram data for every halo in the dataset.

        Requires that the halo mass data has already been loaded with
        ``get_data``. It loads, for every bin, the gas cells of every
        halo and computes both the gas mass fraction as well as the
        temperature of the every gas cell. It then calculates histogram
        data for every halo (for a gas mass fration vs. temperature
        histogram). The histograms are packed into a list which in turn
        is placed into a tuple of as many members as there are mass
        bins. This tuple is then assigned to the ``binned_temperature_hists``
        attribute.

        This method will take considerable computation time.

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
        Load the histogram data for every halo in the dataset.

        Requires that the halo mass data has already been loaded with
        ``get_data``. It loads, for every bin, the gas cells of every
        halo and computes both the gas mass fraction as well as the
        temperature of the every gas cell. It then calculates histogram
        data for every halo (for a gas mass fration vs. temperature
        histogram). The histograms are packed into a list which in turn
        is placed into a tuple of as many members as there are mass
        bins. This tuple is then assigned to the ``binned_temperature_hists``
        attribute.

        This method will take considerable computation time.

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

    def stack_bins(self, to_file: bool = False, suffix: str = "") -> None:
        """
        Stack all histograms per mass bin for average histogram.

        The method will average all histograms in every mass bin and
        assign the resulting average histogram data to the ``histograms``
        attribute. Optionally, the data can also be written to a numpy
        readable binary file.

        :param to_file: whether to write the resulting array of histograms
            to file, defaults to False
        :return: None
        """
        self.histograms = np.zeros((self.n_mass_bins, self.n_bins))
        for bin_num in range(self.n_mass_bins):
            # mask histogram data
            mask = np.where(self.bin_masker == bin_num + 1, 1, 0)
            halo_hists = ma.masked_array(self.hist_data).compress(mask, axis=0)
            self.histograms[bin_num] = np.average(halo_hists, axis=0)

        if to_file:
            cur_dir = Path(__file__).parent.resolve()
            file_path = (
                cur_dir.parent / "data" / f"temperature_hists{suffix}.npy"
            )
            np.save(file_path, self.histograms)

    def plot_stacked_hist(self, bin_num: int, suffix: str = "") -> None:
        """
        Plot the distribution of temperatures for all halos of the mass bin.

        Plots a histogram using the data of all halos in the specified
        mass bin. The mass bin must be given as an array index, i.e.
        starting from zero.

        :param bin_num: mass bin index, starting from zero
        """
        if self.histograms is None:
            self.stack_bins(suffix=suffix)

        self.logger.info(f"Plotting temperature hist for mass bin {bin_num}.")
        fig, axes = plt.subplots(figsize=(4, 4))
        fig.set_tight_layout(True)
        axes.set_title(
            rf"${np.log10(self.mass_bins[bin_num])} < \log \ M_\odot "
            rf"< {np.log10(self.mass_bins[bin_num + 1])}$"
        )
        axes.set_xlabel("Gas temperature [log K]")
        axes.set_ylabel("Average gas mass fraction")

        # calculate bin positions
        _, bins = np.histogram(
            np.array([0]), bins=self.n_bins, range=self.temperature_range
        )
        width = bins[1] - bins[0]
        centers = (bins[:-1] + bins[1:]) / 2

        # plot data
        plot_config = {
            "align": "center",
            "color": "lightblue",
            "edgecolor": "black",
            "log": True,
        }
        axes.bar(centers, self.histograms[bin_num], width=width, **plot_config)

        # save figure
        fig.savefig(
            f"./../figures/001/temperature_hist_{bin_num}{suffix}.pdf",
            bbox_inches="tight"
        )

    def _get_hists_step(self, halo_id: int) -> int:
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

    def _calculate_hist_data(
        self, gas_data: dict[str, ArrayLike]
    ) -> ArrayLike:
        """
        Calculate gas temperature and bin them into histogram data.

        :param gas_data: dictionary holding arrays with gas data. Must
            have keys for internal energy, electron abundance and for
            gas mass
        :return: histogram data, i.e. the binned temperature counts,
            weighted by gas mass fraction
        """
        # calculate helper quantities
        total_gas_mass = np.sum(gas_data["Masses"])
        gas_mass_fracs = gas_data["Masses"] / total_gas_mass
        temperatures = compute.get_temperature(
            gas_data["InternalEnergy"], gas_data["ElectronAbundance"]
        )

        # generate and assign hist data
        hist, _ = np.histogram(
            np.log10(temperatures),
            bins=self.n_bins,
            range=self.temperature_range,
            weights=gas_mass_fracs,
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
