from __future__ import annotations

import multiprocessing as mp
import sys
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

import illustris_python as il
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma

import compute
import config
import constants

if TYPE_CHECKING:
    import logging


class MassDependenceOfGasMassPlotter:
    """
    Provides an interface to plot gas mass (fraction) vs halo mass.

    Instances of this class can separately load halo data, gas data,
    calculate the gas mass (faction) within a certain temperature
    range and plot, for every halo, the gas mass (fraction) vs its
    mass.
    """

    # upper temperature limit for regimes
    temperature_regimes: ClassVar[dict[str, tuple[float, float]]] = {
        "cold": (0.0, 10**4.5),
        "warm": (10**4.5, 10**5.0),
        "hot": (10**5, np.inf),
    }

    def __init__(
        self,
        sim: str,
        logger: logging.Logger,
        use_mass: bool = False,
    ) -> None:
        self.logger = logger
        self.config = config.get_config(sim=sim)
        self.sim = sim
        self.y_axis_data = "mass" if use_mass else "frac"
        # attributes for data, filled in later
        self.n_halos = 0
        self.limits = (0.0, np.inf)  # temperature limits
        self.gas_data = None
        self.halo_data = None
        self.plot_data = None

    def get_gas_data(self) -> None:
        """
        Load gas data for all particles and save it to attributes.

        The method loads the fields required for temperature and gas
        mass fraction calculation and saves them to the ``gas_data``
        attribute. The resulting dataset is large (~30GB). This method
        should only be executed on machines capable of providing enough
        memory for the operation!

        :return: None
        """
        self.logger.info("Loading gas data. This may take a while.")
        fields = [
            "InternalEnergy",
            "ElectronAbundance",
            "StarFormationRate",
            "Masses"
        ]
        part_type = 0  # gas
        self.gas_data = il.snapshot.loadSubset(
            self.config.base_path,
            self.config.snap_num,
            part_type,
            fields=fields
        )
        size = sys.getsizeof(self.gas_data) / 1024. / 1024.
        self.logger.info(f"Finished loading {size:,.2f} MB of gas data.")

    def get_halo_data(self) -> None:
        """
        Load halo data for all halos and save it to attributes.

        The method loads the mass, particle offsets and number of particles
        per halo and saves them to the ``halo_data`` attribute.
        """
        self.logger.info("Loading halo data.")
        halo_masses = il.groupcat.loadHalos(
            self.config.base_path,
            self.config.snap_num,
            fields=self.config.mass_field,
        )
        self.n_halos = len(halo_masses)
        # create a dictionary with arrays for the required data
        self.halo_data = {
            "HaloMasses": halo_masses * 1e10 / constants.HUBBLE,
            "lenGas": np.zeros(self.n_halos, dtype=np.int32),
            "firstGasPartID": np.zeros(self.n_halos, dtype=np.int64),
        }
        for halo in range(self.n_halos):
            offsets = il.snapshot.getSnapOffsets(
                self.config.base_path,
                self.config.snap_num,
                id=halo,
                type="Group",
            )
            # assign relevant data
            self.halo_data["lenGas"][halo] = offsets["lenType"][0]
            self.halo_data["firstGasPartID"][halo] = offsets["offsetType"][0]
        self.logger.info("Finished loading halo data.")

    def get_plot_data(
        self,
        temperature_regime: str,
        processes: int = 16,
        to_file: bool = True,
        suffix: str = ""
    ) -> None:
        """
        Calculate and save value pairs for the given temperature regime.

        The value pairs consist of gas mass (fraction) and halo mass.
        They are saved into an array of shape (N, 2) and placed into the
        ``plot_data`` attribute.

        This method uses multiprocessing with the specified number of
        CPU cores.

        :param temperature_regime: the regime for which to create data,
            must be 'cold', 'warm', or 'hot'.
        :param processes: number of processes to use for calculation
            with multiprocessing (i.e. number of CPU cores to use)
        :param to_file: whether to write the calculated data to an .npz
            file
        :param suffix: suffix to append to the end of the file name.
        :return: None
        """
        if self.halo_data is None:
            self.get_halo_data()
        if self.gas_data is None:
            self.get_gas_data()

        # set limits and y-axis data type
        try:
            self.limits = self.temperature_regimes[temperature_regime]
        except KeyError:
            self.logger.error(
                f"Temperature regime {temperature_regime} does not exist. "
                f"Choose from {self.temperature_regimes.keys()}"
            )
            return

        chunksize = round(self.n_halos / processes / 4, -2)
        self.logger.info(
            f"Starting subprocesses with chunksize {chunksize} on {processes} "
            "processes."
        )
        # pack data together into one large ierable
        multiproc_data = np.column_stack(
            [
                self.halo_data["lenGas"],
                self.halo_data["firstGasPartID"],
            ]
        )
        with mp.Pool(processes=processes) as pool:
            results = pool.starmap(
                self._get_plot_data_step,
                multiproc_data,
                chunksize=int(chunksize)
            )
            pool.close()
            pool.join()
        self.logger.info("Finished processing halo data.")
        self.plot_data = np.array(results)

        if to_file:
            self._save_plot_data(suffix)

    def get_plot_data_lin(
        self,
        temperature_regime: str,
        quiet: bool = False,
        to_file: bool = True,
        suffix: str = ""
    ) -> None:
        """
        Calculate and save value pairs for the given temperature regime.

        The value pairs consist of gas mass (fraction) and halo mass.
        They are saved into an array of shape (N, 2) and placed into the
        ``plot_data`` attribute.

        This method calculates the data sequentially. This can take a
        considerable amount of time and is not recommended for large
        simulations such as TNG300.

        :param temperature_regime: the regime for which to create data,
            must be 'cold', 'warm', or 'hot'.
        :param quiet: whether to suppress status information during
            calculation.
        :param to_file: whether to write the calculated data to an .npz
            file
        :param suffix: suffix to append to the end of the file name.
        :return: None
        """
        if self.halo_data is None:
            self.get_halo_data()
        if self.gas_data is None:
            self.get_gas_data()

        # set limits and y-axis data type
        try:
            self.limits = self.temperature_regimes[temperature_regime]
        except KeyError:
            self.logger.error(
                f"Temperature regime {temperature_regime} does not exist. "
                f"Choose from {self.temperature_regimes.keys()}"
            )
            return

        # allocate memory space
        self.plot_data = np.zeros(self.n_halos)

        self.logger.info(
            f"Calculating plot data for {temperature_regime} gas."
        )
        for i in range(self.n_halos):
            if not quiet:
                perc = i / self.n_halos * 100
                print(
                    f"Processing halo {i}/{self.n_halos} ({perc:.1f}%)",
                    end="\r"
                )
            self.plot_data[i] = self._get_plot_data_step(
                self.halo_data["lenGas"][i],
                self.halo_data["firstGasPartID"][i],
            )
        self.logger.info("Finished calculating plot data.")

        if to_file:
            self._save_plot_data(suffix)

    def plot_mass_dependence(
        self, temperature_regime: str, suffix: str
    ) -> None:
        """
        Plot the gas mass (fraction) vs halo mass.

        The data must be created already for one of the available
        temperature regimes.

        :param temperature_regime: the regime for which to create data,
            must be 'cold', 'warm', or 'hot'.
        :param suffix: suffix to append to the end of the plot file
            name.
        """
        if self.plot_data is None:
            self.logger.error("No data to plot yet!")
            return

        # plot setup
        fig, axes = plt.subplots(figsize=(5, 4))
        fig.set_tight_layout(True)
        axes.set_title(
            f"Gas trends with halo mass for {temperature_regime} gas"
        )
        axes.set_xlabel(r"Halo Mass $M_{200c}$ [$M_\odot$]")
        if self.y_axis_data == "frac":
            axes.set_ylabel(f"Gas mass fraction of {temperature_regime} gas")
        else:
            axes.set_ylabel(
                rf"Gas mass of {temperature_regime} gas [$M_\odot$]"
            )
        axes.set_xscale("log")
        axes.set_yscale("log")

        # plot the dat as datpoints
        plot_config = {
            "marker": ".",
            "color": "black",
            "markersize": 1,
            "linestyle": "None",
            "alpha": 0.4,
        }
        axes.plot(self.halo_data["HaloMasses"], self.plot_data, **plot_config)

        # save figure
        filename = (f"mass_trends_{suffix}.pdf")
        fig.savefig(f"./../../figures/002/{filename}", bbox_inches="tight")

    def _get_plot_data_step(
        self, num_particles: int, first_particle_id: int
    ) -> float:
        """
        Calculate the plot data points and return them.

        The plot data points consists of the gas mass (fraction) of the
        current regime for the chosen halo.

        :param num_particles: The number of gas particles belonging to
            the halo.
        :param first_particle_id: ID of the first gas particle of the
            halo.
        :return: gas mass (fraction) of the gas within the currently
            set temperature limits (``self.limits``)
        """
        # some halos do not contain gas
        if num_particles == 0:
            return np.nan
        # slice start and stop values
        start = first_particle_id
        stop = first_particle_id + num_particles
        # slice the gas data to get only particles of current halo
        internal_energy = self.gas_data["InternalEnergy"][start:stop]
        electron_abund = self.gas_data["ElectronAbundance"][start:stop]
        sfr = self.gas_data["StarFormationRate"][start:stop]
        mass = self.gas_data["Masses"][start:stop]

        # calculate tempertures
        temperatures = compute.get_temperature(
            internal_energy, electron_abund, sfr
        )

        # determine weights for hist
        if self.y_axis_data == "frac":
            total_gas_mass = np.sum(mass)
            values = mass / total_gas_mass
        else:
            values = mass

        # keep only those values that belong to the correct temperatures
        # yapf: disable
        mask = np.where(
            (temperatures > self.limits[0]) & (temperatures <= self.limits[1]),
            1,
            0,
        )
        # yapf: enable
        masked_values = ma.masked_array(values).compress(mask)
        # sum the remaining values
        return np.sum(masked_values)

    def _save_plot_data(self, suffix: str) -> None:
        """
        Save the plot data to file, including halo masses.

        :param suffix: suffix to append to the end of the file name
        """
        if self.plot_data is None:
            self.logger.error("No data to save exists!")
            return

        self.logger.info("Saving plot data to file.")
        cur_dir = Path(__file__).parent.resolve()
        file_name = f"mass_trend{suffix}.npz"
        file_path = (cur_dir.parent.parent / "data" / "002" / file_name)
        np.savez(
            file_path,
            halo_masses=self.halo_data["HaloMasses"],
            y_axis=self.plot_data,
        )
