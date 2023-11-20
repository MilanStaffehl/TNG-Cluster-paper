"""
Pipeline to plot radial temperature profiles for individual halos.
"""
import logging
import time
import tracemalloc
from typing import Literal

import illustris_python as il
from scipy.spatial import KDTree

import library.data_acquisition as daq
import library.processing as prc
from library import compute
from library.config import logging_config
from pipelines.base import Pipeline


class IndividualTemperatureProfilePipeline(Pipeline):
    """
    Pipeline to create plots of radial temperature distribution.

    Pipeline creates 2D histograms of the temperature distribution with
    radial distance to the center of the halo, including particles not
    bound to the halo. It does this for every halo above 10^14 solar
    masses in virial mass.

    This pipeline must load all particle data in order to be able to
    plot gas particles that do ot belong to halos as well.
    """

    def __post_init__(self):
        super().__post_init__()
        # define cutom logging level for memory infos
        logging.addLevelName(18, "MEMLOG")
        if not self.quiet:
            logging_config.change_level(18)

    def run(self) -> int:
        """
        Create radial temperature profiles for all halos above 10^14 M.

        Steps:

        1. Load halo data.
        2. Restrict halo data to halos above mass threshhold.
        3. Load gas cell data required for temperature calculation.
        4. Calculate gas cell temperature, discard obsolete data.
        5. Load gas cell position data.
        6. Create a KDTree from the positions for fast querying
        7. For every selected halo:
           i. Query gas cells for neighbors.
           ii. Create a 2D histogram of temperature vs. distance.
           iii. Save figure and data to file.
           iv. Discard data in memory.

        :return: Exit code.
        """
        # Step 0: create directories, start memory monitoring, timing
        self._create_directories()
        tracemalloc.start()
        begin = time.time()

        # Step 1: acquire halo data
        fields = [self.config.mass_field, "GroupPos"]
        halo_data = daq.halos.get_halo_properties(
            self.config.base_path, self.config.snap_num, fields=fields
        )
        mem = tracemalloc.get_traced_memory()
        self._memlog("Halo gas data memory usage", mem[0], "MB")

        # Step 2: select only halos above threshhold mass
        logging.info("Restricting halo data to log(M) > 14.")
        mask = prc.statistics.sort_masses_into_bins(
            halo_data[self.config.mass_field], [0, 1e14, 1e25]
        )
        selected_ids = prc.statistics.mask_quantity(  # noqa: F841
            halo_data["IDs"], mask, index=2, compress=True
        )
        selected_masses = prc.statistics.mask_quantity(  # noqa: F841
            halo_data[self.config.mass_field], mask, index=2, compress=True
        )
        selected_positions = prc.statistics.mask_quantity(  # noqa: F841
            halo_data["GroupPos"], mask, index=2, compress=True
        )
        del halo_data  # free memory
        del mask  # free memory
        mem = tracemalloc.get_traced_memory()
        self._memlog("Memory usage after restricting halos", mem[0], "kB")
        timepoint = self._timeit(begin, "loading and selecting halo data")

        # Step 3: Load gas cell data for temperature
        logging.info("Loading gas cell data for all gas particles.")
        fields = ["InternalEnergy", "ElectronAbundance", "StarFormationRate"]
        gas_data = il.snapshot.loadSubset(
            self.config.base_path,
            self.config.snap_num,
            partType=0,
            fields=fields
        )
        mem = tracemalloc.get_traced_memory()
        self._memlog("Memory used after loading particles", mem[0])
        timepoint = self._timeit(timepoint, "loading gas cell data")

        # Step 4: Calculate temperature of every gas cell
        part_shape = gas_data["InternalEnergy"].shape
        logging.info(
            f"Calculating temperature for {part_shape[0]:,} gas cells."
        )
        temps = compute.get_temperature(  # noqa: F841
            gas_data["InternalEnergy"],
            gas_data["ElectronAbundance"],
            gas_data["StarFormationRate"],
        )
        # clean up unneeded data
        del gas_data
        # diagnostics
        timepoint = self._diagnostics(
            timepoint, "calculating gas temperatures"
        )

        # Step 5: Load gas cell position data
        gas_positions = daq.gas.get_gas_properties(  # noqa: F841
            self.config.base_path, self.config.snap_num, ["Coordinates"]
        )
        # diagnostics
        timepoint = self._diagnostics(timepoint, "loading gas cell positions")
        # construct KDTree
        logging.info("Constructing KDTree of gas cell positions.")
        positions_tree = KDTree(gas_positions["Coordinates"])  # noqa: F841
        # diagnostics
        timepoint = self._diagnostics(timepoint, "constructing KDTree")

        tracemalloc.stop()
        return 0

    def _diagnostics(
        self,
        start_time: int,
        step_description: str,
        reset_peak: bool = True
    ) -> int:
        """
        Log diagnostic data.

        :param start_time: The start time of the step to diagnose in
            seconds since the epoch.
        :param step_description: A description of the step for which the
            diagnostics are logged.
        :param reset_peak: Whether to reset the peak of the traced
            memory (so that in the next step, the peak can be determined
            independently of the previous steps).
        :return: The time point of the diagnostic in seconds since the
            epoch.
        """
        # memory diagnostics
        mem = tracemalloc.get_traced_memory()
        self._memlog(f"Peak memory usage during {step_description}", mem[1])
        self._memlog(f"Current memory usage after {step_description}", mem[0])
        if reset_peak:
            tracemalloc.reset_peak()
        # runtime diagnostics
        return self._timeit(start_time, step_description)

    def _memlog(
        self,
        message: str,
        memory_used: float,
        unit: Literal["kB", "MB", "GB"] = "GB"
    ) -> None:
        """
        Helper function; logs memory usage message if set to verbose.

        The function will print the given message, followed by a colon
        and the given memory used, converted into the given unit.

        :param message: The message to log before the converted memory.
        :param memory_used: The memory currently used in units of bytes.
        :param unit: The unit to convert the memory into. Can be one of
            the following: kB, MB, GB. If omitted, the memory is given
            in bytes. Defaults to display in gigabytes.
        :return: None
        """
        match unit:
            case "kB":
                memory = memory_used / 1024.
            case "MB":
                memory = memory_used / 1024. / 1024.
            case "GB":
                memory = memory_used / 1024. / 1024. / 1024.
            case _:
                unit = "Bytes"  # assume the unit is bytes
        logging.log(18, f"{message}: {memory:,.4} {unit}.")
