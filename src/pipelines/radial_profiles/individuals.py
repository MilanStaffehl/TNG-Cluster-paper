"""
Pipeline to plot radial temperature profiles for individual halos.
"""
from __future__ import annotations

import logging
import sys
import time
import tracemalloc
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Literal

import illustris_python as il
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import KDTree

from library import compute
from library.data_acquisition import gas_daq, halos_daq
from library.loading import load_radial_profiles
from library.plotting import common, plot_radial_profiles
from library.processing import selection, statistics
from pipelines.base import DiagnosticsPipeline

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass
class IndividualRadialProfilePipeline(DiagnosticsPipeline):
    """
    Pipeline to create plots of radial temperature/density distribution.

    Pipeline creates histograms of the distribution of temperature or
    density with radial distance to the center of the halo, including
    particles not bound to the halo. It does this for every halo above
    10^14 solar masses in virial mass.

    This pipeline must load all particle data in order to be able to
    plot gas particles that do ot belong to halos as well.
    """

    what: Literal["temperature", "density"]
    radial_bins: int
    temperature_bins: int
    log: bool
    forbid_tree: bool = True  # whether KDTree construction is allowed
    ranges: NDArray = np.array([[0, 2], [3, 8.5]])  # hist ranges
    core_only: bool = False
    normalize: bool = True

    divisions: ClassVar[NDArray] = np.array([4.5, 5.5])  # in log K

    def __post_init__(self):
        super().__post_init__()
        self.use_tree = not self.forbid_tree
        if self.config.sim_name == "TNG-Cluster":
            self.group_name = "cluster"
        else:
            self.group_name = "halo"
        # particle id directory and file suffix
        if self.core_only:
            if self.normalize:
                logging.info(
                    f"Received instruction to plot {self.what} profile of "
                    f"core only."
                )
                pid_dir = (
                    self.config.data_home / "particle_ids_core" / "TNG300_1"
                )
                self.suffix = "_core"
            else:
                logging.info(
                    f"Received instructions to plot {self.what} profile of "
                    f"core only, in absolute units."
                )
                # use full halo particles, since for low-mass clusters
                # the 100kpc extend beyond 5% of the virial radius:
                pid_dir = (self.config.data_home / "particle_ids" / "TNG300_1")
                self.suffix = "_core"
        else:
            logging.info(
                f"Received instructions to plot {self.what} profile for full "
                f"halo (possibly in absolute distance units)."
            )
            pid_dir = (self.config.data_home / "particle_ids" / "TNG300_1")
            self.suffix = ""
        if not self.normalize:
            self.suffix += "_absolute"
        self.part_id_dir = pid_dir

    def run(self) -> int:
        """
        Create radial profiles for all halos above 10^14 M.

        Can either be a radial temperature profile or a radial density
        profile, depending on choice of ``self.what``.

        Steps:

        1. Load halo data.
        2. Restrict halo data to halos above mass threshold.
        3. Calculate virial temperature for selected halos
        4. Load gas cell data required for temperature calculation.
        5. Calculate gas cell temperature, discard obsolete data.
        6. Load gas cell position and mass data.
        7. For every selected halo:
           i. Query gas cells for neighbors (either using KDTree or pre-
              saved particle IDs)
           ii. Create a 2D histogram of temperature vs. distance.
           iii. Save figure and data to file.
           iv. Discard data in memory.

        :return: Exit code.
        """
        # Step 0: create directories, start memory monitoring, timing
        self._create_directories(
            subdirs=[
                f"temperature_profiles{self.suffix}",
                f"density_profiles{self.suffix}",
            ],
            force=True
        )
        tracemalloc.start()
        begin = time.time()

        # Step zero-and-a-half: warn if memory intensive
        if self.what == "density" and not self.forbid_tree:
            logging.warning(
                "Was instructed to plot density while tree construction was "
                "not explicitly forbidden. If a tree needs to be constructed, "
                "this will lead to more than 1.5 GB memory use!"
            )

        # Step 1: acquire halo data
        fields = [
            self.config.mass_field,
            self.config.radius_field,
            "GroupPos",
            "GroupVel"
        ]
        halo_data = halos_daq.get_halo_properties(
            self.config.base_path, self.config.snap_num, fields=fields
        )
        mem = tracemalloc.get_traced_memory()
        self._memlog("Halo gas data memory usage", mem[0], "MB")

        # Step 2: select only halos above threshold mass
        logging.info("Restricting halo data to log(M) > 14.")
        mask = np.digitize(halo_data[self.config.mass_field], [0, 1e14, 1e25])
        selected_halos = {
            "ids":
                selection.mask_quantity(
                    halo_data["IDs"], mask, index=2, compress=True
                ),
            "masses":
                selection.mask_quantity(
                    halo_data[self.config.mass_field],
                    mask,
                    index=2,
                    compress=True
                ),
            "positions":
                selection.mask_quantity(
                    halo_data["GroupPos"], mask, index=2, compress=True
                ),
            "velocities":
                selection.mask_quantity(
                    halo_data["GroupVel"], mask, index=2, compress=True
                ),
            "radii":
                selection.mask_quantity(
                    halo_data[self.config.radius_field],
                    mask,
                    index=2,
                    compress=True
                ),
        }
        del halo_data, mask  # free memory
        mem = tracemalloc.get_traced_memory()
        self._memlog("Memory usage after restricting halos", mem[0], "kB")
        timepoint = self._timeit(begin, "loading and selecting halo data")

        # Step 3: calculate virial temperature for halos
        logging.info("Calculating virial temperature for selected halos.")
        selected_halos["virial_temperatures"] = compute.get_virial_temperature(
            selected_halos["masses"], selected_halos["radii"]
        )
        mem = tracemalloc.get_traced_memory()
        self._memlog(
            "Memory used after calculating virial temperatures", mem[0], "kB"
        )
        timepoint = self._timeit(timepoint, "calculating virial temperatures")

        # Step 4: Load gas cell data for temperature
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

        # Step 5: Calculate temperature of every gas cell
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

        # Step 6: Load gas cell position and mass data
        fields = ["Coordinates", "Masses"]
        if self.what == "density":
            fields.append("Velocities")
        gas_data = gas_daq.get_gas_properties(
            self.config.base_path,
            self.config.snap_num,
            fields=fields,
        )
        gas_data["Temperatures"] = temps
        # diagnostics
        timepoint = self._diagnostics(timepoint, "loading gas cell positions")

        # Step 7: check if KDTree construction is required
        workers, positions_tree = self._check_if_tree_required(selected_halos, gas_data)
        if self.use_tree:
            timepoint = self._diagnostics(timepoint, "constructing KDTree")

        # Step 8: Create the radial profiles
        logging.info("Begin processing halos.")
        if self.what == "temperature":
            worker_method = self._process_halo_temperature_profile
        elif self.what == "density":
            worker_method = self._process_halo_density_profile
        else:
            logging.fatal(f"Unrecognized plot type {self.what}.")
            return 3
        for i in range(len(selected_halos["ids"])):
            kwargs = {
                "halo_id": selected_halos["ids"][i],
                "halo_position": selected_halos["positions"][i],
                "halo_velocity": selected_halos["velocities"][i],
                "halo_mass": selected_halos["masses"][i],
                "virial_radius": selected_halos["radii"][i],
                "virial_temperature": selected_halos["virial_temperatures"][i],
                "gas_data": gas_data,
                "positions_tree": positions_tree,
            }
            if self.what == "density":
                kwargs.pop("virial_temperature")  # not required
            worker_method(**kwargs)

        self._diagnostics(timepoint, "plotting individual profiles")

        self._timeit(begin, "total execution")
        tracemalloc.stop()
        return 0

    def _check_if_tree_required(
        self, selected_halos: dict[str, NDArray], gas_data: dict[str, NDArray]
    ) -> tuple[int, KDTree | None]:
        """
        Checks whether the construction of a KDTree is required.

        If the construction of a KDTree is required, the tree will be
        constructed and ``self.use_tree`` is set to True. Otherwise, or
        if three construction is explicitly forbidden, ``self.use_tree``
        will be set to False.

        Returns the number of workers and the KDTree object, if needed,
        otherwise returns the tuple (1, None) as a dummy return value.

        :param selected_halos: The dictionary containing the restricted
            halo data.
        :param gas_data: The dictionary containing the gas cell data.
        :return: The tuple of the number of workers and the KDTree, if
            construction of it is required.
        """
        logging.debug(f"Searching particle ID directory: {self.part_id_dir}")
        try:
            available_ids = set([f.stem for f in self.part_id_dir.iterdir()])
        except IOError:
            logging.warning(
                f"Could not find or read the particle IDs from the directory "
                f"{self.part_id_dir}. Did you delete or move the directory?"
            )
            available_ids = set()

        suffix = self.suffix.removesuffix("_absolute")
        if self.core_only and not self.normalize:
            # also remove the "_core" suffix as all particles are used:
            suffix = suffix.removesuffix("_core")
        required_ids = set(
            [f"particles_halo_{i}{suffix}" for i in selected_halos["ids"]]
        )

        # check whether all halos have particle ID files available
        if required_ids.issubset(available_ids):
            logging.info(
                "Found particle IDs of associated particles for all halos. "
                "Continuing with existing particle ID data."
            )
            self.use_tree = False
            positions_tree = None
        else:
            logging.debug(f"Missing files: {required_ids - available_ids}")
            # if the user explicitly forbade tree creation, cancel execution
            if self.forbid_tree:
                logging.fatal(
                    "Not all selected halos have associated particle IDs on "
                    "file, but tree creation was forbidden. Cannot continue "
                    "with the job at hand, canceling execution."
                )
                sys.exit(2)
            # otherwise, create the tree
            logging.info(
                "Not all selected halos have particle IDs of associated "
                "particles saved. Continuing with KDTree construction."
            )
            logging.info("Constructing KDTree from particle positions.")
            self.use_tree = True
            positions_tree = KDTree(
                gas_data["Coordinates"],
                balanced_tree=True,
                compact_nodes=True,
            )

        # prepare variables for querying
        workers = self.processes if self.processes else 1
        return workers, positions_tree

    def _process_halo_temperature_profile(
        self,
        halo_id: int,
        halo_position: NDArray,
        halo_velocity: NDArray,
        halo_mass: float,
        virial_radius: float,
        virial_temperature: float,
        gas_data: dict[str, NDArray],
        positions_tree: KDTree | None,
    ) -> None:
        """
        Process a single halo into a temperature radial profile.

        :param halo_id: The ID of the halo.
        :param halo_position: The 3D vector pointing to the position
            of the halo, in units of kpc.
        :param halo_velocity: The 3D vector of the halo peculiar velocity
            in units of km/s
        :param halo_mass: The mass of the halo in units of solar masses.
        :param virial_radius: The virial radius of the halo, in units
            of kpc.
        :param virial_temperature: The virial temperature of the halo in
            units of Kelvin.
        :param gas_data: The dictionary of the gas cell data.
        :param positions_tree: If ``self.use_tree`` is True, meaning
            that neighboring particles must be queried using a KDTree,
            this must be the KDTree of all particle positions in the
            simulation. Otherwise, it can be set to None.
        :return: None
        """
        restricted_gas_data = self._restrict_gas_data_to_halo(
            gas_data,
            halo_id,
            halo_position,
            halo_velocity,
            virial_radius,
            positions_tree
        )

        # weight by gas mass
        weights = restricted_gas_data["Masses"]
        weights /= np.sum(restricted_gas_data["Masses"])

        # create histogram
        h, _, _, = np.histogram2d(
            restricted_gas_data["Distances"],
            np.log10(restricted_gas_data["Temperatures"]),
            range=self.ranges,
            bins=(self.radial_bins, self.temperature_bins),
            weights=weights,
        )
        hn, xe, ye = statistics.column_normalized_hist2d(
            restricted_gas_data["Distances"],
            np.log10(restricted_gas_data["Temperatures"]),
            ranges=self.ranges,
            bins=(self.radial_bins, self.temperature_bins),
            values=weights,
            normalization="density",
        )

        # save data
        if self.to_file:
            logging.debug(
                f"Writing histogram data for halo {halo_id} to file."
            )
            filepath = (
                Path(self.paths["data_dir"])
                / f"temperature_profiles{self.suffix}"
            )
            filename = (
                f"{self.paths['data_file_stem']}_{self.group_name}_"
                f"{halo_id}.npz"
            )
            np.savez(
                filepath / filename,
                histogram=hn,
                original_histogram=h,
                xedges=xe,
                yedges=ye,
                halo_id=halo_id,
                halo_mass=halo_mass,
                virial_temperature=virial_temperature,
            )  # yapf: disable

        # plot and save data
        self._plot_temperature_profile(
            halo_id=halo_id,
            halo_mass=halo_mass,
            virial_temperature=virial_temperature,
            histogram=hn,
            xedges=xe,
            yedges=ye,
        )

        # cleanup
        del restricted_gas_data, weights
        del hn, h, xe, ye

    def _process_halo_density_profile(
        self,
        halo_id: int,
        halo_position: NDArray,
        halo_velocity: NDArray,
        halo_mass: float,
        virial_radius: float,
        gas_data: dict[str, NDArray],
        positions_tree: KDTree | None,
    ) -> None:
        """
        Process a single halo into a density profile, split by radial velocity.

        :param halo_id: The ID of the halo.
        :param halo_position: The 3D vector pointing to the position
            of the halo, in units of kpc.
        :param halo_velocity: The 3D vector of the peculiar velocity of
            the halo in km/s.
        :param halo_mass: The mass of the halo in units of solar masses.
        :param virial_radius: The viriral radius of the halo, in units
            of kpc.
        :param gas_data: The dictionary of the gas cell data.
        :param positions_tree: If ``self.use_tree`` is True, meaning
            that neighboring particles must be queried using a KDTree,
            this must be the KDTree of all particle positions in the
            simulation. Otherwise, it can be set to None.
        :return: None
        """
        logging.debug(f"Processing halo {halo_id} into plot.")
        restricted_gas_data = self._restrict_gas_data_to_halo(
            gas_data,
            halo_id,
            halo_position,
            halo_velocity,
            virial_radius,
            positions_tree
        )

        # bin gas particles by temperature:
        mask = np.digitize(
            np.log10(restricted_gas_data["Temperatures"]),
            self.temperature_bins,
        )
        cool_gas_data = selection.mask_data_dict(
            restricted_gas_data, mask, index=1
        )
        warm_gas_data = selection.mask_data_dict(
            restricted_gas_data, mask, index=2
        )
        hot_gas_data = selection.mask_data_dict(
            restricted_gas_data, mask, index=3
        )

        # create a total density profile of infalling gas
        infall_mask = restricted_gas_data["RadialVelocities"] >= 0
        outflow_mask = restricted_gas_data["RadialVelocities"] < 0
        total_in, edges = statistics.volume_normalized_radial_profile(
            restricted_gas_data["Distances"][infall_mask],
            restricted_gas_data["Masses"][infall_mask],
            self.radial_bins,
            virial_radius if self.normalize else None,
            radial_range=self.ranges[0],
        )
        total_out, _ = statistics.volume_normalized_radial_profile(
            restricted_gas_data["Distances"][outflow_mask],
            restricted_gas_data["Masses"][outflow_mask],
            self.radial_bins,
            virial_radius if self.normalize else None,
            radial_range=self.ranges[0],
        )

        # create density profile for cool gas
        infall_mask = cool_gas_data["RadialVelocities"] >= 0
        outflow_mask = cool_gas_data["RadialVelocities"] < 0
        cool_in, _ = statistics.volume_normalized_radial_profile(
            cool_gas_data["Distances"][infall_mask],
            cool_gas_data["Masses"][infall_mask],
            self.radial_bins,
            virial_radius if self.normalize else None,
            radial_range=self.ranges[0],
        )
        cool_out, _ = statistics.volume_normalized_radial_profile(
            cool_gas_data["Distances"][outflow_mask],
            cool_gas_data["Masses"][outflow_mask],
            self.radial_bins,
            virial_radius if self.normalize else None,
            radial_range=self.ranges[0],
        )

        # create density profile for warm gas
        infall_mask = warm_gas_data["RadialVelocities"] >= 0
        outflow_mask = warm_gas_data["RadialVelocities"] < 0
        warm_in, _ = statistics.volume_normalized_radial_profile(
            warm_gas_data["Distances"][infall_mask],
            warm_gas_data["Masses"][infall_mask],
            self.radial_bins,
            virial_radius if self.normalize else None,
            radial_range=self.ranges[0],
        )
        warm_out, _ = statistics.volume_normalized_radial_profile(
            warm_gas_data["Distances"][outflow_mask],
            warm_gas_data["Masses"][outflow_mask],
            self.radial_bins,
            virial_radius if self.normalize else None,
            radial_range=self.ranges[0],
        )

        # create density profile for hot gas
        infall_mask = hot_gas_data["RadialVelocities"] >= 0
        outflow_mask = hot_gas_data["RadialVelocities"] < 0
        hot_in, _ = statistics.volume_normalized_radial_profile(
            hot_gas_data["Distances"][infall_mask],
            hot_gas_data["Masses"][infall_mask],
            self.radial_bins,
            virial_radius if self.normalize else None,
            radial_range=self.ranges[0],
        )
        hot_out, _ = statistics.volume_normalized_radial_profile(
            hot_gas_data["Distances"][outflow_mask],
            hot_gas_data["Masses"][outflow_mask],
            self.radial_bins,
            virial_radius if self.normalize else None,
            radial_range=self.ranges[0],
        )

        # write data to file
        if self.to_file:
            logging.debug(f"Writing data for halo {halo_id} to file.")
            filepath = (
                Path(self.paths["data_dir"]) / f"density_profiles{self.suffix}"
            )
            filename = (
                f"{self.paths['data_file_stem']}_{self.group_name}_"
                f"{halo_id}.npz"
            )
            np.savez(
                filepath / filename,
                total_inflow=total_in,
                total_outflow=total_out,
                cool_inflow=cool_in,
                cool_outflow=cool_out,
                warm_inflow=warm_in,
                warm_outflow=warm_out,
                hot_inflow=hot_in,
                hot_outflow=hot_out,
                edges=edges,
                halo_id=halo_id,
                halo_mass=halo_mass,
                halo_position=halo_position,
            )

        # plot
        self._plot_density_profile(
            halo_id,
            halo_mass,
            edges,
            total_in,
            total_out,
            cool_in,
            cool_out,
            warm_in,
            warm_out,
        )

    def _restrict_gas_data_to_halo(
        self,
        gas_data: dict[str, NDArray],
        halo_id: int,
        halo_pos: NDArray,
        halo_vel: NDArray,
        halo_radius: float,
        positions_tree: KDTree | None
    ) -> dict[str, NDArray]:
        """
        Restrict the given gas data only to the halo of the given ID.

        Appends to the gas data catalogue also the distance to the
        current halo center in units of virial radii.

        :param gas_data: The dictionary containing the gas data to
            constrain to only the particles within 2 R_vir of the given
            halo.
        :param halo_id: ID of the halo.
        :param halo_pos: The 3D cartesian vector giving the coordinates
            of the halo position. In units of kpc.
        :param halo_radius: The virial radius of the halo in units of
            kpc.
        :param positions_tree: If the neighboring particles must be
            queried from a KDTree, this must be the KDTree to use. If
            particle IDs already exist on file, this cna be None.
        :return: The dictionary of gas data, but only containing as
            values arrays, that have been restricted to particles within
            2 R_vir of the given halo. Additionally, also contains a new
            field 'Distances' which contains the distance of every gas
            particle to the halo position in units of virial radii.
        """
        neighbors = self._query_for_neighbors(
            halo_id,
            halo_pos,
            halo_radius,
            positions_tree,
            self.processes,
        )

        # restrict gas data to chosen particles only:
        restricted_gas_data = {}
        for field, value in gas_data.items():
            if field == "count":
                restricted_gas_data["count"] = len(neighbors)
                continue
            restricted_gas_data[field] = gas_data[field][neighbors]

        # calculate distances
        part_distances = np.linalg.norm(
            restricted_gas_data["Coordinates"] - halo_pos, axis=1
        )
        if self.normalize:
            part_distances /= halo_radius
            assert np.max(part_distances) <= self.ranges[0, 1]

        restricted_gas_data.update({"Distances": part_distances})

        # update with radial velocities
        if "Velocities" in gas_data.keys():
            # calculate radial velocities
            radial_vel = compute.get_radial_velocities(
                halo_pos,
                halo_vel,
                restricted_gas_data["Coordinates"],
                restricted_gas_data["Velocities"],
            )
            restricted_gas_data.update({"RadialVelocities": radial_vel})
            if self.to_file:
                logging.debug(
                    f"Writing radial velocities of halo {halo_id} to file."
                )
                filename = f"radial_velocity_halo_{halo_id}.npy"
                filepath = (
                    self.config.data_home / "particle_velocities"
                    / self.config.sim_path
                )
                np.save(filepath / filename, radial_vel)

        return restricted_gas_data

    def _query_for_neighbors(
        self,
        halo_id: int,
        halo_position: NDArray,
        halo_radius: float,
        positions_tree: KDTree | None,
        workers: int
    ) -> NDArray:
        """
        Return the array of indices of particles within the given halo.

        The particles are queried either from the KDTree, or loaded from
        file. All particles within 2 R_vir are chosen and their indices
        in the list of particles is returned.

        :param halo_id: ID of the halo to query for particles.
        :param halo_position: The shape (3, ) array of the halo position
            in ckpc.
        :param halo_radius: The halo radius in units of ckpc.
        :param positions_tree: If ``self.use_tree`` is True and the
            particles are queried from an existing KDTree, this must be
            the KDTree. Otherwise, if no tree is required, this can be
            set to None.
        :param workers: The number of cores used to query the tree. If
            no tree is used, this can be arbitrarily set to 1.
        :return: The array of list indices of particles which belong to
            the chosen halo, i.e. are within 2 R_vir of the halo center.
        """
        suffix = self.suffix.removesuffix("_absolute")
        if self.core_only and not self.normalize:
            # also remove the "_core" suffix as all particles are used
            suffix = suffix.removesuffix("_core")
        # find all particles within 2 * R_vir
        if self.use_tree:
            neighbors = positions_tree.query_ball_point(
                halo_position,
                self.ranges[0][-1] * halo_radius,
                workers=workers
            )
            if self.to_file:
                logging.debug(
                    f"Saving particle indices of halo {halo_id} to file."
                )
                np.save(
                    self.part_id_dir / f"particles_halo_{halo_id}{suffix}.npy",
                    neighbors
                )
        else:
            neighbors = np.load(
                self.part_id_dir / f"particles_halo_{halo_id}{suffix}.npy"
            )
        return neighbors

    def _plot_temperature_profile(
        self,
        halo_id: int,
        halo_mass: float,
        virial_temperature: float,
        histogram: NDArray,
        xedges: NDArray,
        yedges: NDArray,
    ) -> None:
        """
        Plot the temperature histogram of a single halo.

        :param halo_id: The halo ID.
        :param halo_mass: The mass of the halo in units of solar masses.
        :param virial_temperature: Virial temperature of the halo in Kelvin.
        :param histogram: The (N, N) shape array of the histogram data.
        :param xedges: The edges of the x bins.
        :param yedges: The edges of the y bins.
        """
        fig, axes = plt.subplots(figsize=(5, 4))
        fig.set_tight_layout(True)
        title = (
            f"{self.group_name.capitalize()} {halo_id} "
            rf"($10^{{{np.log10(halo_mass):.2f}}} M_\odot$)"
        )
        ranges = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        if self.normalize:
            xlabel = r"Distance from halo center [$R_{200}$]"
        else:
            xlabel = r"Distance from halo center [$kpc$]"
        with np.errstate(invalid="ignore", divide="ignore"):
            if self.log:
                plot_radial_profiles.plot_2d_radial_profile(
                    fig,
                    axes,
                    histogram,
                    ranges,
                    title=title,
                    xlabel=xlabel,
                    cbar_label="Normalized gas mass fraction (log10)",
                    cbar_limits=[-4.2, None],
                    scale="log",
                    cbar_ticks=[0, -1, -2, -3, -4],
                )
            else:
                plot_radial_profiles.plot_2d_radial_profile(
                    fig,
                    axes,
                    histogram,
                    ranges,
                    title=title,
                    xlabel=xlabel,
                    cbar_label="Normalized gas mass fraction"
                )
        # virial temperature and temperature divisions
        if self.log:
            axes.hlines(
                np.log10(virial_temperature),
                xedges[0],
                xedges[-1],
                colors="blue"
            )
            plot_radial_profiles.overplot_temperature_divisions(
                axes, self.divisions, xedges[0], xedges[-1]
            )
        else:
            axes.hlines(
                virial_temperature, xedges[0], xedges[-1], colors="blue"
            )
            plot_radial_profiles.overplot_temperature_divisions(
                axes, 10**self.divisions, xedges[0], xedges[-1]
            )

        # save figure
        supplementary = f"{self.group_name}_{halo_id}"
        self._save_fig(
            fig,
            ident_flag=supplementary,
            subdir=f"./{supplementary}",
        )

    def _plot_density_profile(
        self,
        halo_id: int,
        halo_mass: float,
        edges: NDArray,
        total_inflow: NDArray,
        total_outflow: NDArray,
        cool_inflow: NDArray,
        cool_outflow: NDArray,
        warm_inflow: NDArray,
        warm_outflow: NDArray,
    ) -> None:
        """
        Plot the density profile of a halo split by radial velocity.

        :param halo_id: ID of the halo to plot.
        :param halo_mass: Mass of the halo in solar masses.
        :param edges: The bin edges of the radial bins.
        :param total_inflow: Density histogram of the total inflowing gas.
        :param total_outflow: Density histogram of the total outflowing gas.
        :param cool_inflow: Density histogram of the cool inflowing gas.
        :param cool_outflow: Density histogram of the cool outflowing gas.
        :param warm_inflow: Density histogram of the warm inflowing gas.
        :param warm_outflow: Density histogram of the warm outflowing gas.
        :return: None
        """
        fig, axes = plt.subplots(figsize=(4, 4))
        fig.set_tight_layout(True)

        title = (
            f"{self.group_name.capitalize()} {halo_id} "
            rf"($10^{{{np.log10(halo_mass):.2f}}} M_\odot$)"
        )
        ranges = np.array([edges[0], edges[-1]])
        if self.normalize:
            xlabel = r"Distance from halo center [$R_{200}$]"
        else:
            xlabel = r"Distance from halo center [$kpc$]"

        # total gas content, both split and summed
        plot_radial_profiles.plot_1d_radial_profile(
            axes,
            total_inflow + total_outflow,
            edges,
            xlims=ranges,
            log=self.log,
            title=title,
            xlabel=xlabel,
        )
        plot_radial_profiles.plot_1d_radial_profile(
            axes,
            total_inflow,
            edges,
            xlims=ranges,
            log=self.log,
            linestyle="dotted",
        )
        plot_radial_profiles.plot_1d_radial_profile(
            axes,
            total_outflow,
            edges,
            xlims=ranges,
            log=self.log,
            linestyle="dashed",
        )

        # Warm gas, total, in- and outflow
        plot_radial_profiles.plot_1d_radial_profile(
            axes,
            warm_inflow + warm_outflow,
            edges,
            xlims=ranges,
            log=self.log,
            label=r"Warm ($10^{4.5} - 10^{5.5} K$)",
            color=common.temperature_colors_rgb["warm"],
        )
        plot_radial_profiles.plot_1d_radial_profile(
            axes,
            warm_inflow,
            edges,
            xlims=ranges,
            log=self.log,
            color=common.temperature_colors_rgb["warm"],
            linestyle="dotted",
        )
        plot_radial_profiles.plot_1d_radial_profile(
            axes,
            warm_outflow,
            edges,
            xlims=ranges,
            log=self.log,
            color=common.temperature_colors_rgb["warm"],
            linestyle="dashed"
        )

        # Plot cool gas total, in- and outflow
        plot_radial_profiles.plot_1d_radial_profile(
            axes,
            cool_inflow + cool_outflow,
            edges,
            xlims=ranges,
            log=self.log,
            label=r"Cool ($< 10^{4.5} K$)",
            color=common.temperature_colors_rgb["cool"],
        )
        plot_radial_profiles.plot_1d_radial_profile(
            axes,
            cool_inflow,
            edges,
            xlims=ranges,
            log=self.log,
            color=common.temperature_colors_rgb["cool"],
            linestyle="dotted",
        )
        plot_radial_profiles.plot_1d_radial_profile(
            axes,
            cool_outflow,
            edges,
            xlims=ranges,
            log=self.log,
            color=common.temperature_colors_rgb["cool"],
            xlabel=xlabel,
            linestyle="dashed",
        )

        axes.legend(fontsize=10, frameon=False)
        # save
        supplementary = f"{self.group_name}_{halo_id}"
        self._save_fig(
            fig,
            ident_flag=supplementary,
            subdir=f"./{supplementary}",
        )


class IndividualProfilesFromFilePipeline(IndividualRadialProfilePipeline):
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

        if self.no_plots:
            logging.warning(
                "Was asked to load data but not plot it. This is pretty "
                "pointless and probably not what you wanted."
            )
            return 1

        # Step 1: load data
        logging.info(f"Start loading {self.what} data from file.")
        if self.what == "temperature":
            load_generator = load_radial_profiles.load_individuals_2d_profile(
                self.paths["data_dir"] / f"temperature_profiles{self.suffix}",
                (self.radial_bins, self.temperature_bins),
            )
            plotting_func = self._plot_temperature_profile
        elif self.what == "density":
            load_generator = load_radial_profiles.load_individuals_1d_profile(
                self.paths["data_dir"] / f"density_profiles{self.suffix}",
                self.radial_bins,
            )
            plotting_func = self._plot_density_profile
        else:
            logging.fatal(f"Unrecognized plot type: {self.what}.")
            return 2

        # Step 2: plot data
        logging.info("Plotting individual halo profiles.")
        for halo_data in load_generator:
            if self.what == "temperature":
                halo_data.pop("original_histogram")
            elif self.what == "density":
                halo_data.pop("halo_position")
                halo_data.pop("hot_inflow")
                halo_data.pop("hot_outflow")
            plotting_func(**halo_data)
        logging.info("Done! Finished plotting individual halo profiles.")


class IndividualProfilesTNGClusterPipeline(IndividualRadialProfilePipeline):
    """
    Pipeline to create radial profiles for TNG Cluster.

    Pipeline creates 2D histograms of the temperature distribution with
    radial distance to the center of the halo, including particles not
    bound to the halo. It does this for every halo above 10^14 solar
    masses in virial mass.

    This Pipeline is specific to the TNG Cluster simulation and utilizes
    some of the simulations properties to be more efficient than its
    parent class at handling particles.
    """

    def run(self) -> int:
        """
        Create radial temperature profiles for zoom-in cluster of TNG Cluster.

        Steps:

        1. Load halo data, restricted to zoom-ins.
        2. Calculate virial temperatures.
        3. For every cluster:
           1. Load gas cell data for temperature (only loading particles
              from the zoom).
           2. Calculate gas cell temperature, discard obsolete data.
           3. Load gas cell position data, calculate halocentric distance.
           4. Create a 2D histogram of temperature vs. distance. Let
              the histogram function handle particles beyond 2 R_200
              automatically.
           5. Save data and figure to file.
           6. Clean-up: discard all data for the current halo.

        :return: Exit code.
        """
        # Step 0: create directories, start monitoring, timing
        self._create_directories(
            subdirs=[f"{self.what}_profiles{self.suffix}"], force=True
        )
        tracemalloc.start()
        begin = time.time()

        # Step 1: Load and restrict halo data from TNG Cluster
        fields = [
            self.config.mass_field,
            self.config.radius_field,
            "GroupPos",
            "GroupVel",
        ]
        halo_data = halos_daq.get_halo_properties(
            self.config.base_path,
            self.config.snap_num,
            fields=fields,
            cluster_restrict=True,
        )
        timepoint = self._diagnostics(begin, "loading halo data", unit="kB")

        # Step 2: calculate virial temperatures
        halo_data["VirialTemperature"] = compute.get_virial_temperature(
            halo_data[self.config.mass_field],
            halo_data[self.config.radius_field],
        )
        timepoint = self._diagnostics(
            timepoint, "calculating virial temperature", unit="kB"
        )

        # Step 3: Loop through halos
        logging.info("Start processing individual clusters.")
        for i, halo_id in enumerate(halo_data["IDs"]):
            logging.debug(f"Processing cluster {halo_id} ({i}/352).")
            # Step 3.1: Load gas cell data for temperature
            gas_temperatures = gas_daq.get_cluster_temperature(
                self.config.base_path,
                self.config.snap_num,
                halo_id,
            )

            # Step 3.2: Load gas cell position data, calculate distance
            fields = ["Coordinates", "Masses"]
            if self.what == "density":
                fields.append("Velocities")
            gas_data = gas_daq.get_gas_properties(
                self.config.base_path,
                self.config.snap_num,
                fields=fields,
                cluster=halo_id,
            )
            gas_distances = np.linalg.norm(
                gas_data["Coordinates"] - halo_data["GroupPos"][i], axis=1
            )
            if self.normalize:
                gas_distances /= halo_data[self.config.radius_field][i]

            # Step 3.3: Create histogram
            gas_data.update(
                {
                    "Temperatures": gas_temperatures,
                    "Distances": gas_distances,
                }
            )
            if self.what == "temperature":
                self._process_halo_temperature_profile(
                    halo_id,
                    halo_data["GroupPos"][i],
                    halo_data["GroupVel"][i],
                    halo_data[self.config.mass_field][i],
                    halo_data[self.config.radius_field][i],
                    halo_data["VirialTemperature"][i],
                    gas_data,
                    None,
                )
            elif self.what == "density":
                # add in the radial velocities
                logging.debug(
                    f"Calculating radial velocity for "
                    f"{len(gas_data['Velocities'])} gas cells."
                )
                radial_velocities = compute.get_radial_velocities(
                    halo_data["GroupPos"][i],
                    halo_data["GroupVel"][i],
                    gas_data["Coordinates"],
                    gas_data["Velocities"],
                )
                gas_data.update({"RadialVelocities": radial_velocities})
                self._process_halo_density_profile(
                    halo_id,
                    halo_data["GroupPos"][i],
                    halo_data["GroupVel"][i],
                    halo_data[self.config.mass_field][i],
                    halo_data[self.config.radius_field][i],
                    gas_data,
                    None,
                )
            else:
                logging.fatal(f"Unrecognized plot type {self.what}.")
                return 3

            timepoint = self._diagnostics(
                timepoint, f"processing halo {halo_id} ({i}/352)"
            )

        self._diagnostics(begin, "total execution")
        tracemalloc.stop()
        return 0

    def _restrict_gas_data_to_halo(
        self,
        gas_data: dict[str, NDArray],
        halo_id: int,
        halo_pos: NDArray,
        halo_vel: NDArray,
        halo_radius: float,
        positions_tree: KDTree | None
    ) -> dict[str, NDArray]:
        """
        Overwrites parent method, since no restriction is required.

        Method returns the ``gas_data`` unaltered since for TNG-Cluster,
        the gas data is already loaded only for the cluster.

        :param gas_data: Dictionary of gas data for the TNG-Cluster halo.
        :param halo_id: Dummy parameter.
        :param halo_pos: Dummy parameter.
        :param halo_vel: Dummy parameter.
        :param halo_radius: Dummy parameter.
        :param positions_tree: Dummy parameter.
        :return: ``gas_data``, unaltered (as it is already restricted).
        """
        return gas_data
