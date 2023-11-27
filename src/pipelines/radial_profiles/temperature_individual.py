"""
Pipeline to plot radial temperature profiles for individual halos.
"""
from __future__ import annotations

import logging
import time
import tracemalloc
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import illustris_python as il
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import KDTree

import library.data_acquisition as daq
import library.loading.radial_profiles as ld
import library.plotting.radial_profiles as ptr
import library.processing as prc
from library import compute
from library.config import logging_config
from pipelines.base import Pipeline

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass
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

    radial_bins: int
    temperature_bins: int

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
        6. For every selected halo:
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
        fields = [self.config.mass_field, self.config.radius_field, "GroupPos"]
        halo_data = daq.halos.get_halo_properties(
            self.config.base_path, self.config.snap_num, fields=fields
        )
        mem = tracemalloc.get_traced_memory()
        self._memlog("Halo gas data memory usage", mem[0], "MB")

        # Step 2: select only halos above threshhold mass
        logging.info("Restricting halo data to log(M) > 14.")
        mask = np.digitize(halo_data[self.config.mass_field], [0, 1e14, 1e25])
        selected_ids = prc.statistics.mask_quantity(
            halo_data["IDs"], mask, index=2, compress=True
        )
        selected_masses = prc.statistics.mask_quantity(
            halo_data[self.config.mass_field], mask, index=2, compress=True
        )
        selected_positions = prc.statistics.mask_quantity(
            halo_data["GroupPos"], mask, index=2, compress=True
        )
        selected_radii = prc.statistics.mask_quantity(
            halo_data[self.config.radius_field], mask, index=2, compress=True
        )
        del halo_data, mask  # free memory
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
        temps = compute.get_temperature(
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
        gas_data = daq.gas.get_gas_properties(
            self.config.base_path,
            self.config.snap_num, ["Coordinates", "Masses"]
        )
        # diagnostics
        timepoint = self._diagnostics(timepoint, "loading gas cell positions")

        # Step 6: construct KDTree
        logging.info("Constructing KDTree of gas cell positions.")
        positions_tree = KDTree(
            gas_data["Coordinates"],
            balanced_tree=True,
            compact_nodes=True,
        )
        # diagnostics
        timepoint = self._diagnostics(timepoint, "constructing KDTree")

        # Step 7: Create the radial profiles
        workers = self.processes if self.processes else 1
        logging.info(f"Begin processing halos with {workers} workers.")
        for i in range(len(selected_ids)):
            # find all particles within 2 * R_vir
            neighbors = positions_tree.query_ball_point(
                selected_positions[i], 2 * selected_radii[i], workers=workers
            )  # list of indces, can be used for slices
            # slice and normalize distances
            part_positions = gas_data["Coordinates"][neighbors]
            part_distances = np.linalg.norm(
                part_positions - selected_positions[i], axis=1
            ) / selected_radii[i]
            # slice temperatures
            part_temperatures = temps[neighbors]
            # weight by gas mass
            weights = gas_data["Masses"][neighbors]
            weights /= np.sum(gas_data["Masses"][neighbors])
            # plot and save data
            self._plot_halo(
                selected_ids[i],
                selected_masses[i],
                part_distances,
                part_temperatures,
                weights,
            )
            # cleanup
            del part_positions, part_distances, part_temperatures, weights

        timepoint = self._diagnostics(
            timepoint, "plotting individual profiles"
        )
        self._timeit(begin, "total execution")
        tracemalloc.stop()
        return 0

    def _plot_halo(
        self,
        halo_id: int,
        halo_mass: float,
        distances: NDArray,
        temperatures: NDArray,
        weights: NDArray
    ) -> None:
        title = (
            f"Temperature profile of halo {halo_id} "
            f"(10^{np.log10(halo_mass):.2f} log M_sol)"
        )
        f, _, h, xe, ye = ptr.generate_generic_radial_profile(
            distances,
            np.log10(temperatures),
            "Temperature [log K]",
            weights=weights,
            colorbar_label="Gas fraction",
            density=False,
            title=title,
            xbins=self.radial_bins,
            ybins=self.temperature_bins,
        )

        # save data
        if self.to_file:
            logging.debug(
                f"Writing histogram data for halo {halo_id} to file."
            )
            filepath = Path(self.paths["data_dir"])
            filename = (f"{self.paths['data_file_stem']}_halo_{halo_id}.npz")
            np.savez(
                filepath / filename,
                hist=h,
                xedges=xe,
                yedges=ye,
                halo_id=halo_id,
                halo_mass=halo_mass,
            )

        # save figure
        if self.no_plots:
            return
        name = (f"{self.paths['figures_file_stem']}_halo_{halo_id}.pdf")
        path = Path(self.paths["figures_dir"]) / f"halo_{halo_id}"
        if not path.exists():
            logging.debug(
                f"Creating missing figures directory for halo "
                f"{halo_id}."
            )
            path.mkdir(parents=True)
        f.savefig(path / name, bbox_inches="tight")
        plt.close(f)

    def _diagnostics(
        self,
        start_time: int,
        step_description: str,
        reset_peak: bool = True,
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


class ITProfilesFromFilePipeline(IndividualTemperatureProfilePipeline):
    """
    Pipeline to recreate the temp profiles of individual halos from file.
    """

    def __post_init__(self) -> None:
        return super().__post_init__()

    def run(self) -> int:
        """
        Recreate radial temperature profiles from file.

        Steps for every halo:

        1. Load data from file
        2. Plot the halo data

        :return: Exit code.
        """
        # Step 0: verify directories
        if exit_code := self._verify_directories() > 0:
            return exit_code

        # Step 1: load data
        load_generator = ld.load_individuals(
            self.paths["data_dir"], (self.radial_bins, self.temperature_bins)
        )
        for halo_data in load_generator:
            halo_id = halo_data['halo_id']
            ranges = np.concatenate([halo_data["xedges"], halo_data["yedges"]])
            title = (
                rf"Halo {halo_id} "
                rf"($10^{{{np.log10(halo_data['halo_mass']):.2f}}} M_\odot$)"
            )
            f, _ = ptr.plot_radial_temperature_profile(
                halo_data["histogram"],
                f"halo {halo_id}",
                ranges,
                title=title,
                cbar_label="Gas fraction",
            )
            # save figure
            if self.no_plots:
                logging.warning(
                    "Was asked to load data but not plot it. This is pretty "
                    "pointless and probably not what you wanted."
                )
                return 1
            name = (f"{self.paths['figures_file_stem']}_halo_{halo_id}.pdf")
            path = Path(self.paths["figures_dir"]) / f"halo_{halo_id}"
            if not path.exists():
                logging.debug(
                    f"Creating missing figures directory for halo "
                    f"{halo_id}."
                )
                path.mkdir(parents=True)
            f.savefig(path / name, bbox_inches="tight")
