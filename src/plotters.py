from __future__ import annotations

from typing import TYPE_CHECKING

import illustris_python as il
import numpy as np

import compute
import config
import loader

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

    def __init__(
        self, sim: str, mass_bins: list[float], logger: logging.Logger
    ) -> None:
        self.logger = logger
        self.config = config.get_config(sim=sim)
        self.sim = sim
        self.mass_bins = mass_bins
        self.n_mass_bins = len(mass_bins) - 1
        # create attributes for data
        self.binned_indices = None
        self.binned_masses = None
        self.binned_temperature_hists = None  # histograms of temp

    def get_data(self) -> None:
        """
        Load and bin the halos from the simulation, save binned masses.

        Method loads the halo data from the simulation and bins all halos
        by mass. IDs of the halos are binned as well and saved in attrs
        as well.

        :return: None
        """
        data = loader.get_halos_binned_by_mass(self.mass_bins, self.config)
        self.binned_indices = data[0]
        self.binned_masses = data[1]

    def get_hists(self) -> list[int]:
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

        Some halos might not have any gas (shallow dark halos). these
        are skipped in the pocessing. This means that the final lists
        of histogram data might not match one-to-one with the lists of
        halo masses and indices anymore and instead are shorter! To be
        able to account for this issue, the method will return a list of
        all halo IDs of those halos that were skipped due to not having
        any gas particles.

        This method will take considerable computation time.

        :return: list of halo IDs that are not present in the hist
            data due to not having any gas particles.
        """
        N_BINS = 45

        if self.binned_indices is None or self.binned_masses is None:
            raise TypeError(
                "Data is not loaded yet, please use 'get_data' to load it first."
            )
        
        temperature_hists = []
        skipped_halo_ids = []

        self.logger.info("Beginning histogram data generation.")
        for mass_bin_index in range(self.n_mass_bins):
            self.logger.info(
                f"Processing halos in mass bin "
                f"{mass_bin_index}/{self.n_mass_bins}."
            )
            cur_mass_bin_hist_data = []
            num_halos = len(self.binned_masses[mass_bin_index])

            for i, halo_index in enumerate(self.binned_indices[mass_bin_index]):
                # status report
                print(
                    (f"Processing halo with ID {halo_index} ({i}/{num_halos}) "
                     f"{i / num_halos * 100:.0f} %"), 
                    end="\r"
                )
                # load halo gas cell fields
                fields = ["InternalEnergy", "ElectronAbundance", "Masses"]
                gas_data = il.snapshot.loadHalo(
                    self.config.base_path,
                    self.config.snap_num,
                    halo_index,
                    partType=0,  # gas
                    fields=fields,
                )
                # sort out all halos without gas particles
                if gas_data["count"] == 0:
                    self.logger.debug(
                        f"Halo {halo_index} does not contain any gas cells, "
                        "skipped."
                    )
                    skipped_halo_ids.append(halo_index)
                    continue
                # calculate gas mass fraction and temperature
                total_gas_mass = np.sum(gas_data["Masses"])
                gas_mass_frac = gas_data["Masses"] / total_gas_mass
                temperatures = compute.get_temperature(
                    gas_data["InternalEnergy"], gas_data["ElectronAbundance"]
                )
                # generate hist data
                hist = np.histogram(
                    temperatures, N_BINS, weights=gas_mass_frac
                )
                # add hist data to list of hist data for current mass bin
                cur_mass_bin_hist_data.append(hist)

            self.logger.info(
                f"Finished processing mass bin {mass_bin_index}/"
                f"{self.n_mass_bins}, processed {len(cur_mass_bin_hist_data)} "
                "halos."
            )
            # add list of hist data to final binned tuple
            temperature_hists.append(cur_mass_bin_hist_data)

        # assign generated data to attribute
        self.binned_temperature_hists = tuple(temperature_hists)
        return skipped_halo_ids
    
    def __str__(self) -> str:
        """
        Return a string containing information on the current mass bins.

        :return: information on currently loaded mass bins
        """
        if self.binned_indices is None or self.binned_masses is None:
            return "No data loaded yet."
        
        ret_str = ""
        for i, bin_ in enumerate(self.binned_indices):
            ret_str += (
                f"Bin {i} [{self.mass_bins[i]:.2e}, "
                f"{self.mass_bins[i + 1]:.2e}]): "
                f"{len(bin_)} halos\n"
            )
        return ret_str

