"""
Pipeline to plot movement of a few particles with time.
"""
from __future__ import annotations

import dataclasses
import logging
from typing import TYPE_CHECKING

import illustris_python as il
import matplotlib.lines
import matplotlib.pyplot as plt
import numpy as np

from library import units
from library.data_acquisition import bh_daq, tracers_daq
from library.processing import selection
from pipelines.base import Pipeline

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from numpy.typing import NDArray


@dataclasses.dataclass
class FollowParticlesPipeline(Pipeline):
    """
    Pipeline to follow a few selected particles back in time.

    Pipeline chooses from halo 0 a few particles, and follows them
    back in time to trace their x- and y-coordinates, which are then
    plotted and colored by their particle type. For this purpose, it
    uses tracers.
    """

    max_tracers: int = 5  # max number of tracers to follow
    cluster_id: int = 0  # which cluster to use
    particle_positions: NDArray | None = None
    particle_type_flags: NDArray | None = None
    plot_lines: bool = False

    def run(self) -> int:
        """
        Select particles and trace them back in time.

        Steps:

        1. Load particles from halo 0 and select particles
        2. Load tracers for halo 0, select those that are associated with
           the selected particles
        3. Allocate memory for the results
        4. For every snapshot before 99:
           1. Load tracers with ID and parent ID
           2. Identify parent ID of the tracers selected at z = 0
           3. Load particles with their UID and coordinates
           4. Find those particles that have the desired parent ID
           5. Save their coordinates and type in an array
        5. Plot the coordinates with time, colored by parent particle type

        :return: Exit code.
        """
        # Step 0: create directories, allocate memory
        self._create_directories()

        # Step 1: Load particles from halo 0 and select particles
        particle_data = bh_daq.get_most_massive_blackhole(
            self.config.base_path,
            self.config.snap_num,
            halo_id=self.cluster_id,
            fields=["ParticleIDs", "Coordinates"],
        )

        # Step 2: Load tracers for entire snapshot
        tracer_data = tracers_daq.load_tracers(
            self.config.base_path,
            self.config.snap_num,
            cluster_id=self.cluster_id,
        )
        # select only those tracers, that belong to the BH
        tracer_indices = selection.select_if_in(
            tracer_data["ParentID"], particle_data["ParticleIDs"]
        )
        n_tracers = len(tracer_indices)
        logging.info(
            f"Found {n_tracers} tracers in most massive BH of halo "
            f"{self.cluster_id}."
        )
        # select only some of those
        if n_tracers > self.max_tracers:
            logging.info(
                f"Too many tracers found, restricting to only first "
                f"{self.max_tracers}."
            )
            n_tracers = self.max_tracers
            tracer_indices = tracer_indices[:self.max_tracers]
        # select the tracers which we have settled on
        selected_tracers_id = tracer_data["TracerID"][tracer_indices]
        logging.debug(
            f"Selected {n_tracers} with IDs {selected_tracers_id} at array "
            f"positions {tracer_indices}"
        )

        # Step 3: Allocate memory and save first results. We set all to
        # NaN first, so that when we encounter tracers that are all in
        # the same cell, we can just leave the remaining entries as NaN
        # and not worry about them in plotting.
        self.particle_positions = np.empty((100, n_tracers, 2))
        self.particle_positions[:] = np.nan  # initialize to all NaN
        self.particle_type_flags = np.empty((100, n_tracers), dtype=np.uint8)
        self.particle_type_flags[:] = 99  # initialize all to 99
        # set first values (only one, since all tracers are in the same BH
        self.particle_positions[99, 0] = particle_data["Coordinates"][:-1]
        self.particle_type_flags[99, 0] = 5  # type BH

        # Step 4: Step through the snapshots backwards in time
        for snap_num in reversed(range(0, 99, 1)):
            logging.info(
                f"Finding tracers and associated particles in snapshot "
                f"{snap_num}."
            )
            # Step 1: load tracers with ID and parent ID
            cur_tracer_data = tracers_daq.load_tracers(
                self.config.base_path, snap_num, cluster_id=self.cluster_id
            )

            # Step 2: identify parent ID of tracers we are following
            cur_tracer_indices = selection.select_if_in(
                cur_tracer_data["TracerID"], selected_tracers_id
            )
            cur_tracer_ids = cur_tracer_data["TracerID"][cur_tracer_indices]
            # sanity check: does the selection return the same tracer IDs?
            if not np.all(np.equal(cur_tracer_ids, selected_tracers_id)):
                logging.warning(
                    f"Tracer IDs suddenly changed: expected tracer IDs "
                    f"{selected_tracers_id}, but found {cur_tracer_ids} when "
                    f"selecting tracer IDs from tracer indices selected via "
                    f"tracers selected at z = 0."
                )
            cur_parent_ids = cur_tracer_data["ParentID"][cur_tracer_indices]

            # Step 3, 4 & 5: load the particles and identify the selected ones
            self._identify_and_save_particles(cur_parent_ids, snap_num)

        # Step 5: save data to file
        if self.to_file:
            logging.info("Saving traced particle data to file.")
            filename = f"{self.paths['data_file_stem']}_n{self.max_tracers}.npz"
            np.savez(
                self.paths['data_dir'] / filename,
                particle_positions=self.particle_positions,
                particle_type_flags=self.particle_type_flags,
            )

        # Step 6: plot the data
        if self.plot_lines:
            logging.info("Plotting particle positions as line plot.")
            f, a = self._plot_lines()
            ident_flag = f"lines_{n_tracers}_tracers"
        else:
            logging.info("Plotting particle positions as scatter plot.")
            f, a = self._plot_scatter()
            ident_flag = f"scatter_{n_tracers}_tracers"
        self._save_fig(f, ident_flag=ident_flag)

        return 0

    def _identify_and_save_particles(
        self, parent_ids: NDArray, snap_num: int
    ) -> None:
        """
        Identify the particle coordinates and types of the given IDs.

        Method loads, one after another, the particles of the given
        snapshot and compares the given particle IDs to those of the
        loaded particles. When it finds a match, it saves the corresponding
        coordinates and type flag in the allocated array.

        :param parent_ids: An array of the particle UIDs which to look
            for in the particle data.
        :param snap_num: The number of the snapshot to look through.
        :return: None, data is saved in attribute array.
        """
        logging.debug(
            f"Attempting to find {len(parent_ids)} particles with particle "
            f"IDs {parent_ids} in snapshot {snap_num}."
        )

        # lists to temporarily store data before concatenating it
        particle_ids_list = list()
        coordinates_list = list()
        type_flags_list = list()

        # load all coordinates for the particles
        for part_type in [0, 4, 5]:
            # load all particles of this type
            particle_data = self._load_particle_properties(part_type, snap_num)
            if particle_data["count"] == 0:
                # no particles available, so no data to append
                logging.debug(
                    f"Skipping part type {part_type}: no particles found."
                )
                continue
            # place data in corresponding list
            particle_ids_list.append(particle_data["ParticleIDs"])
            coordinates_list.append(particle_data["Coordinates"][:, :-1])
            # generate an array of identical type flag integers
            cur_type_flags = np.empty_like(
                particle_data["ParticleIDs"], dtype=np.uint8
            )
            cur_type_flags[:] = part_type
            type_flags_list.append(cur_type_flags)

        # concatenate all data for all three particle types
        particle_ids = np.concatenate(particle_ids_list, axis=0)
        coordinates = np.concatenate(coordinates_list, axis=0)
        type_flags = np.concatenate(type_flags_list, axis=0)

        # check if the desired particles are available. Use mode
        # `searchsort` to receive one index per ID, regardless of
        # uniqueness of ID:
        indices = selection.select_if_in(
            particle_ids,
            parent_ids,
            mode="searchsort",
            assume_subset=True,
        )
        # verify found indices via IDs, including correct ordering
        np.testing.assert_equal(particle_ids[indices], parent_ids)

        # index particle positions and type flag and safe them
        self.particle_positions[snap_num] = coordinates[indices]
        self.particle_type_flags[snap_num] = type_flags[indices]

        logging.debug(
            f"Found {len(np.unique(indices))} unique particles with IDs "
            f"{parent_ids} with types {self.particle_type_flags[snap_num]}"
        )

    def _load_particle_properties(self, part_type: int,
                                  snap_num: int) -> dict[str, NDArray]:
        """
        Load the particle ID and coordinates of the given particle type.

        :param part_type: Particle type index.
        :param snap_num: Number of snapshot to load from.
        :return: Data dictionary containing fields ``ParticleIDs`` and
            ``Coordinates``.
        """
        fields = ["ParticleIDs", "Coordinates"]
        gas_data = il.snapshot.loadOriginalZoom(
            self.config.base_path,
            snap_num,
            self.cluster_id,
            partType=part_type,
            fields=fields,
        )

        if gas_data["count"] == 0:
            logging.warning(
                f"Loaded data structure is empty, no particles of type "
                f"{part_type} exist in snapshot {snap_num} for this zoom-in "
                f"region."
            )

        # convert units
        gas_data_physical = {}
        for field, data in gas_data.items():
            gas_data_physical[field] = units.UnitConverter.convert(data, field)

        return gas_data_physical

    def _plot_scatter(self) -> tuple[Figure, Axes]:
        """
        Plot the movement of particles over time into the central BH.

        :return: Tuple of figure and axes, with plot drawn on.
        """
        fig, axes = plt.subplots(figsize=(4, 4))
        fig.set_tight_layout(True)
        axes.set_aspect("equal", adjustable="datalim")
        axes.set_xlabel("x [cMpc]")
        axes.set_ylabel("y [cMpc]")

        flag_to_color = {0: "dodgerblue", 4: "gold", 5: "black", 99: "red"}

        for tracer_idx in range(self.particle_type_flags.shape[-1]):
            xs = self.particle_positions[:, tracer_idx, 0] / 1000
            ys = self.particle_positions[:, tracer_idx, 1] / 1000
            ptypes = self.particle_type_flags[:, tracer_idx]
            ptypes[ptypes == np.nan] = 1

            # colors
            colors = [
                flag_to_color[x] if not np.isnan(x) else "red" for x in ptypes
            ]
            snapnum = np.linspace(0, 99, 100, dtype=int)

            cb = axes.scatter(
                xs,
                ys,
                c=snapnum,
                edgecolors=colors,
                linewidths=0.5,
                marker="o",
                s=8,
                cmap="inferno",
                vmin=0,
                vmax=99,
            )

        # legend
        handles = [
            matplotlib.lines.Line2D(
                [], [],
                marker="o",
                color="grey",
                markeredgecolor="dodgerblue",
                ls="",
                label="Gas"
            ),
            matplotlib.lines.Line2D(
                [], [],
                marker="o",
                color="grey",
                markeredgecolor="gold",
                ls="",
                label="Stars"
            ),
            matplotlib.lines.Line2D(
                [], [],
                marker="o",
                color="grey",
                markeredgecolor="black",
                ls="",
                label="Black holes"
            )
        ]
        axes.legend(handles=handles)

        # colorbar
        fig.colorbar(cb, ax=axes, label="Snap num")

        return fig, axes

    def _plot_lines(self) -> tuple[Figure, Axes]:
        """
        Plot the positions of the tracers connected by lines.

        :return: Tuple of figure and axes.
        """
        fig, axes = plt.subplots(figsize=(4, 4))
        fig.set_tight_layout(True)
        axes.set_aspect("equal", adjustable="datalim")
        axes.set_xlabel("x [cMpc]")
        axes.set_ylabel("y [cMpc]")

        # sum of type flags of two points gives color
        flag_sum_to_color = {
            0: "dodgerblue",  # two type 0
            8: "gold",  # two type 4
            10: "black",  # two type 5
            4: "mediumspringgreen",  # one type 0, one type 4
            5: "indigo",  # one type 0, one type 5
            9: "sienna",  # one type 4, one type 5
            99: "red",  # failure: one type 0, one type 99
            103: "red",  # failure: one type 4, one type 99
            104: "red",  # failure: one type 5, one type 99
            198: "red",  # failure: two type 99
        }

        # yes, I know this is horribly slow and ineffective, but this is
        # a test script and I didn't want to spend a day optimizing it...
        for k in range(self.particle_type_flags.shape[-1]):
            for i in range(99):
                pos_now = self.particle_positions[i, k]
                type_now = self.particle_type_flags[i, k]
                pos_next = self.particle_positions[i + 1, k]
                type_next = self.particle_type_flags[i + 1, k]
                try:
                    color = flag_sum_to_color[type_now + type_next]
                except KeyError:
                    color = "red"
                except ValueError:
                    color = "red"
                axes.plot(
                    [pos_now[0], pos_next[0]],
                    [pos_now[1], pos_next[1]],
                    marker=None,
                    linestyle="solid",
                    color=color,
                )

        # add a legend
        handles = [
            matplotlib.lines.Line2D(
                [], [],
                marker="none",
                color="dodgerblue",
                ls="solid",
                label="Gas"
            ),
            matplotlib.lines.Line2D(
                [], [], marker="none", color="gold", ls="solid", label="Stars"
            ),
            matplotlib.lines.Line2D(
                [], [],
                marker="none",
                color="black",
                ls="solid",
                label="Black holes"
            )
        ]
        axes.legend(handles=handles)

        return fig, axes


class FollowParticlesFromFilePipeline(FollowParticlesPipeline):
    """
    Load particle data from file to plot.
    """

    def run(self) -> int:
        """
        Load data and call plotting method.

        :return: Exit code.
        """
        filename = f"{self.paths['data_file_stem']}_n{self.max_tracers}.npz"
        with np.load(self.paths["data_dir"] / filename) as data_file:
            self.particle_positions = data_file["particle_positions"]
            self.particle_type_flags = data_file["particle_type_flags"]

        n_tracers = self.particle_type_flags.shape[-1]
        if self.plot_lines:
            logging.info("Ploting particle positions as lines plot")
            f, _ = self._plot_lines()
            ident_flag = f"lines_{n_tracers}_tracers"
        else:
            logging.info("Plotting particle positions as scatter plot.")
            f, _ = self._plot_scatter()
            ident_flag = f"scatter_{n_tracers}_tracers"
        self._save_fig(f, ident_flag=ident_flag)

        return 0
