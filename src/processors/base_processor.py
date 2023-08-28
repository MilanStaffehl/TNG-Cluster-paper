"""
Base class for processors.
"""
from __future__ import annotations

import multiprocessing as mp
from typing import TYPE_CHECKING, Any

import illustris_python as il
import numpy as np

import compute
import config
import constants

if TYPE_CHECKING:
    import logging
    from pathlib import Path

    from numpy.typing import NDArray


class BaseProcessor:
    """
    The base class for processors.

    This base class provides a modular approach to working with halo gas
    temperatures. At its core, it provides a set of public methods to
    calculate temperatures for halo gas in a given simulation, process
    it into other types of data, and then plot the resulting data.

    This is achieved by subclasses of this method providing implementations
    for the stub methods of this class which get called in a predefined
    order. Another method of this class then implements plotting the
    data and a third allows loading data previously saved to file to be
    loaded.

    The class will load two kinds of data: *primary data* is the data
    that is calculated from the gas temperatures. It is stored inside
    the ``data`` attribute. *Auxilary data* is all kind of data that is
    not calculated from the gas temperatures, but still required for
    the plots (e.g. halo radii).

    In detail, these three public methods provide the building blocks
    for the modular processor subclasses:

    - :meth:`get_data`: This method will gather all the data required
      for plotting, including most crucially that data which is calculated
      from the halo gas temperature, but it can optionally also gather
      additional ("auxiliary") data.
    - :meth:`load_data`: This method is a stub. When implemented, it is
      meant to load data previously gathered by ``get_data`` and saved
      to file, and place it into the corresponding attributes for use
      with ``plot_data`` without the need to process data again from
      scratch.
    - :meth:`plot_data`: This method is a stub. When implemented, it is
      meant to create a plot of the data previously gathered by either
      of the other two public methods.

    Alongside these three methods, the base class also provides other
    methods - some of which are stubs - that are important to understand:

    - :meth:`_get_auxilary_data`: This method is called by `get_data`
      *before* gathering the primary data. Arguments to this method can
      be given to `get_data` in the form of a dict of keyworded args.
      It is meant to load additional data for plotting.
      This method is a stub and *can* be implemented by subclasses.
    - :meth:`_post_process_data`: This method is called by `get_data`
      *before* gathering the primary data. Arguments to this method can
      be given to `get_data` in the form of a dict of keyworded args.
      It is meant to process the primary data further, if required, or
      verify the validity of the loaded data.
      This method is a stub and *can* be implemented by subclasses.
    - :meth:`_fallback`: This method is called when a halo is found that
      has no gas particles or when a halo is skipped by :meth:`_skip_halo`.
      It is meant to return a data array for such a case.
      This method is a stub. It *can* be implemented by a subclass.
    - :meth:`_process_temperatures`: This method takes the array of halo
      gas temperatures and halo gas data and returns the data that is
      desired for the final plot. The exact implementation of the
      calculation is up to the subclass to determine.
      This method is a stub. It *must* be implemented by a subclass.
    - :meth:`_skip_halo`: This method determines whether a halo may be
      skipped during the calculation. This is useful to save processing
      time when a large portion of halos is not going to be part of the
      plot anyway.
      This method is a stub. It *should* be implemented by subclasses.

    All data loaded, be it the primary data or auxilary data, is expected
    to be saved in appropriate attributes. The primary data computed from
    the gas temperatures in ``get_data`` are saved in ``self.data`` and
    has the form of an NDArray of shape (N, M), where N is the number of
    halos in the simulation and M is the length of the data array per
    halo. M is determined upon instntiation with the ``data_length``
    argument.

    A typical use of this classes subclass instances may look as follows:

    1. Instantiate a subclass of ``BaseProcessor``.
    2. Call the ``get_data`` method to retrieve data.
    3. Call the ``plot_data`` method to create a plot of the data.
    """

    def __init__(
        self, sim: str, logger: logging.Logger, data_length: int
    ) -> None:
        self.sim = sim
        self.config = config.get_default_config(sim=sim)
        self.logger = logger
        self.len_data = data_length
        # data fields
        self.indices = None
        self.masses = None
        self.data = None  # results of get_data are assigned here

    def get_data(
        self,
        processes: int = 16,
        quiet: bool = False,
        aux_kwargs: dict[str, Any] | None = None,
        post_kwargs: dict[str, Any] | None = None
    ) -> None:
        """
        Load the data for the processor and place it into attributes.

        This method gathers all the data required for the processor to
        calculate and plot the desired data. It works by calling different
        methods after each other:

        1. Calls :meth:`_get_halo_data` to retrieve halo masses and a
           list of halo indices.
        2. Calls :meth:`_getauxilary_data` to set up any seconary data
           required for plotting.
        3. Gathers primary data, using eithe rmultiprocessing or doing
           so sequentially, depending on the ``processes`` argument:
           a. For every halo, calculates the gas cell temperatures.
           b. Passes the array of temperatures and gas data to the
              :meth:`_process_temperatures` method.
           c. Gathers the results of the processing in an array and
              assigns this array to ``self.data``.
        4. Calls :meth:`_post_process_data` to perform post-processing
           or verification of data.

        This method may use multiprocessing for the caluclation of gas
        cell temperatures. Set the number of processes to 0 to instead
        calculate them sequentially on one process only.

        Subclasses may EXTEND this method, but they should not overwrite
        it under any circumstances!

        :param processes: Number of porcesses to use in multiprocessing.
            Set to 0 to calculate data sequentially instead.
        :param quiet: Whether to suppress status reports in sequential
            processing. This is useful when output is redirected to a
            file instead of a console output.
        :param aux_kwargs: keyworded arguments to pass to the
            ``_get_auxilary_data`` method.
        :param post_kwargs: Keyworded arguments to pass to the
            ``_post_process_data`` method.
        :return: None
        """
        if aux_kwargs is None:
            aux_kwargs = {}
        if post_kwargs is None:
            post_kwargs = {}
        # load required and optional data
        self._get_halo_data()  # hard requirement
        self._get_auxilary_data(processes, quiet, **aux_kwargs)  # optional
        # get the data to plot
        if processes > 0:
            self._get_data_multiprocessed(processes=processes)
        else:
            self._get_data_sequentially(quiet=quiet)
        # post process data
        self._post_process_data(processes, quiet, **post_kwargs)  # optional

    def plot_data(self, *args, **kwargs) -> None:
        """
        Plot the data.

        This method is a stub and needs to be implemented by subclasses.
        """
        pass

    def load_data(self, filepath: str | Path, *args, **kwargs) -> None:
        """
        Load existing data from file.

        This method is a stub and needs to be implemented by subclasses.
        """
        pass

    def _get_auxilary_data(self, processes: int, quiet: bool) -> None:
        """
        Load additional data required for plotting.

        This method is a stub and needs to be implemented by subclasses.
        It allows for data to be loaded using multiprocessing and as
        such takes the appropriate srguments for number of processes and
        verbosity.
        """
        pass

    def _get_data_multiprocessed(self, processes: int) -> None:
        """
        Load the data using multiprocessing.

        This method uses multiprocessing with the given number of processes
        to load the gas cell data, compute temperatures, and process them
        according to the ``process_temperatures`` method. The result is then
        placed into the ``data`` attribute.

        :param processes: The number of processes to use.
        """
        self.logger.info("Start processing halo data on mutliple cores.")
        # multiprocess the entire problem
        chunksize = round(len(self.indices) / processes / 4, -2)
        self.logger.info(
            f"Starting {processes} subprocesses with chunksize {chunksize}."
        )
        with mp.Pool(processes=processes) as pool:
            results = pool.map(
                self._get_data_step, self.indices, chunksize=int(chunksize)
            )
            pool.close()
            pool.join()
        self.logger.info("Finished processing halo data.")

        # assign array of data to attribute
        self.data = np.array(results)

    def _get_data_sequentially(self, quiet: bool) -> None:
        """
        Load the data sequentially.

        This method loads the gas data for all halos, computes the gas
        cell temperatures and then processes it accoriding to the
        ``process_temperatures`` method. The result is then saved in the
        ``data`` attribute.

        This method can optionally print status udate to the output. It
        is recommended to suppress this with ``quiet = True`` when the
        output is redirected to file.

        :param quiet: Whether to suppress status report messages.
        """
        self.logger.info("Start processing halo data sequentially.")
        n_halos = len(self.indices)
        self.data = np.zeros((n_halos, self.len_data))
        for i, halo_id in enumerate(self.indices):
            if not quiet:
                perc = i / n_halos * 100
                print(f"Processing halo {i}/{n_halos} ({perc:.1f}%)", end="\r")
            self.data[i] = self._get_data_step(halo_id)
        self.logger.info("Finished processing halo data.")

    def _get_data_step(self, halo_id: int) -> NDArray:
        """
        Calculate temperatures and return processed data for a single halo.

        This method loads the gas cell data for a single halo and from
        it calculates the temperatures of the gas cells. It then passes
        the temperatures to the ``process_temperatures`` method for
        processing alongside the loaded gas data. The result of the
        processing is returned.

        Optionally, halos cn be skipped by implementing a skip condition
        inside the ``_skip_halo`` method.

        :param halo_id: The ID of the halo to process.
        :return: The data derived from the halo gas cell temperatures.
        """
        # optionally skip a halo under specific conditions
        if self._skip_halo(halo_id):
            return self._fallback(halo_id)

        # load halo gas cell data
        fields = [
            "InternalEnergy",
            "ElectronAbundance",
            "Masses",
            "StarFormationRate"
        ]
        gas_data = il.snapshot.loadHalo(
            self.config.base_path,
            self.config.snap_num,
            halo_id,
            partType=0,  # gas
            fields=fields,
        )

        # some halos do not contain gas
        if gas_data["count"] == 0:
            self.logger.debug(
                f"Halo {halo_id} contains no gas. Returning a fallback array."
            )
            return self._fallback(halo_id)

        # calculate temperatures
        temperatures = compute.get_temperature(
            gas_data["InternalEnergy"],
            gas_data["ElectronAbundance"],
            gas_data["StarFormationRate"],
        )
        # post-process temperatures and return result
        return self._process_temperatures(halo_id, temperatures, gas_data)

    def _get_halo_data(self) -> None:
        """
        Load halo masses and create a list of indices from it.

        This method is required for the ``get_data`` method to work. Any
        subclass may EXTEND it. If a subclass OVERWRITES it, it must at
        least reproduce the assignent of ``self.masses`` with the halo
        masses and ``self.indices`` with a list of indices. Overwriting
        this method rather than extending it can be useful when more
        fields from the halos are required (e.g. the radius) to avoid
        opening and reading from the same files twice.
        """
        self.logger.info("Loading halo masses and indices.")
        halo_data = il.groupcat.loadHalos(
            self.config.base_path,
            self.config.snap_num,
            fields=[self.config.mass_field],
        )
        num_halos = len(halo_data)
        self.indices = np.indices([num_halos], sparse=True)[0]
        self.masses = (halo_data * 1e10 / constants.HUBBLE)
        self.logger.info("Finished loading halo mass data.")

    def _fallback(self, halo_id: int) -> NDArray:
        """
        Return a fallback data array for halos without any gas.

        In the process of loading halo data, some halos will not contain
        any gas. To not break the implementation, even these halos will
        require a data array of length ``self.len_data``. This method
        must be OVERWRITTEN by subclasses to provide such a fallback for
        gas-deficient halos, unless the default defined in the base class
        is sufficient.

        The default implementation returns an array of NaN's of length
        ``self.len_data``.

        :param halo_id: The ID of the halo.
        :return: A fallback data array for halos without any gas particles. If
            not overwritten, this is an array of NaN's.
        """
        fallback = np.empty(self.len_data, dtype=np.float32)
        fallback.fill(np.nan)
        return fallback

    def _post_process_data(
        self, processes: int, quiet: bool, **kwargs
    ) -> None:
        """
        Called after the main data has been loaded.

        This method is a stuband needs to be implemented by subclasses.
        """
        pass

    def _process_temperatures(
        self,
        halo_id: int,
        temperatures: NDArray,
        gas_data: dict[str, NDArray]
    ) -> NDArray:
        """
        Return a data array processed from temperatures and gas data.

        This method performs the processing of halo gas cell temperatures
        and the halo gas data into the desired data array.

        This method is a stub and must be overwritten by subclasses.

        :param halo_id: The ID of the halo.
        :param temperatures: Array of gas cell temperatures of the halo.
        :param gas_data: The dictionary containing gas data for the halo.
        :return: A data array of length ``self.len_data``.
        """
        return temperatures

    def _skip_halo(self, halo_id: int) -> bool:
        """
        Return bool whether the given halo can be skipped in the calclation.

        Some halos might not need to be processed. Skipping them can save
        computation time. This method needs to be overwritten by
        subclasses and determines whether a halo can be skipped or not.

        If the halo with the given halo ID can be skipped, this method
        returns True, otherwise it returns False.

        :param halo_id: The halo ID of the halo to check.
        :return: Whether to skip the halo. True means the halo is skipped,
            False means the halo is processed.
        """
        return False
