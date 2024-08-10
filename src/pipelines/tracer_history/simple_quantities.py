"""
Trace back some simple quantities of the tracer particles.
"""
from __future__ import annotations

import abc
import dataclasses
import logging
from typing import TYPE_CHECKING, ClassVar

import h5py
import matplotlib.pyplot as plt
import numpy as np

from library import constants
from library.data_acquisition import gas_daq, halos_daq
from library.plotting import common
from pipelines import base

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclasses.dataclass
class TraceSimpleQuantitiesBackABC(base.Pipeline, abc.ABC):
    """
    Base class to trace back simple tracer quantities.

    Needs to have its abstract methods implemented.
    """

    quantity_label: str  # y-axis label for quantity
    color: str  # color for faint lines
    make_ridgeline: bool = False

    n_clusters: ClassVar[int] = 352
    n_snaps: ClassVar[int] = 100 - constants.MIN_SNAP

    def __post_init__(self):
        self.tracer_filepath = (
            self.config.data_home / "tracer_history" / "particle_ids"
            / "TNG_Cluster"
            / f"particle_ids_from_snapshot_{self.config.snap_num}.hdf5"
        )

    def run(self) -> int:
        """
        Trace back quantity and plot it.

        :return: Exit code.
        """
        # Step 0: set up directories
        self._create_directories()
        logging.info(f"Tracing back {self.quantity_label} in time.")

        # Step 1: allocate memory for quantities to trace
        quantity_mean = np.zeros((self.n_clusters, self.n_snaps))
        quantity_median = np.zeros_like(quantity_mean)
        # TODO: remove and replace with 2D hist
        quantity_min = np.zeros_like(quantity_mean)
        quantity_max = np.zeros_like(quantity_mean)

        # Step 2: Load cluster primary
        group_primaries = halos_daq.get_halo_properties(
            self.config.base_path,
            self.config.snap_num,
            ["GroupFirstSub"],
            cluster_restrict=True,
        )["GroupFirstSub"]

        # Step 3: Load the gas particle indices file
        tracer_file = h5py.File(self.tracer_filepath, "r")

        # Step 4: Loop through snapshots and zooms to get quantity
        for i, snap_num in enumerate(range(constants.MIN_SNAP, 100)):
            logging.info(f"Processing snapshot {snap_num}.")
            for zoom_id in range(self.n_clusters):
                logging.debug(
                    f"Processing snap {snap_num}, zoom-in {zoom_id}."
                )
                # 4.1: Get particle quantity
                quantity = self._load_quantity(
                    snap_num, zoom_id, group_primaries
                )

                # 4.2: Find gas particle indices
                group = f"ZoomRegion_{zoom_id:03d}"
                indices = tracer_file[f"{group}/particle_indices"][snap_num, :]
                flags = (
                    tracer_file[f"{group}/particle_type_flags"][snap_num, :]
                )
                gas_indices = indices[flags == 0]  # select only gas
                # TODO: make unique to avoid same cell contributing
                #  multiple times

                # 4.3: Mask quantity to only the selected indices
                traced_quantity = quantity[gas_indices]

                # 4.4: Fill allocated arrays
                quantity_mean[zoom_id, i] = np.mean(traced_quantity)
                quantity_median[zoom_id, i] = np.median(traced_quantity)
                # TODO: make 2D histogram here
                quantity_min[zoom_id, i] = np.min(traced_quantity)
                quantity_max[zoom_id, i] = np.max(traced_quantity)

        tracer_file.close()

        # Step 5: save data to file
        if self.to_file:
            # TODO: implement once final version of pipeline exists
            logging.warning("No data saving implemented yet!")

        # Step 6: plot line plots
        self._plot_and_save_lineplots(
            quantity_mean, quantity_median, quantity_min, quantity_max
        )

        # Step 7: plot 2D histograms / ridgeline plots
        if self.make_ridgeline:
            self._plot_and_save_ridgelineplots()
        else:
            self._plot_and_save_2dhistograms()
        return 0

    def _plot_and_save_lineplots(
        self, means: NDArray, medians: NDArray, mins: NDArray, maxs: NDArray
    ) -> None:
        """
        Plot the development of the quantity with time.

        Function plots the mean and median lines for individual clusters
        plus the mean and median over all clusters in a line plot over
        redshift. It saves the plots to file.

        :param means: Array of mean of the quantity for every cluster,
            i.e. an array of shape (100, 352) where every entry is the
            mean of the gas quantity at that snapshot for that cluster.
        :param medians: Array of medians of the quantity for every
            cluster, i.e. an array of shape (100, 352) where every entry
            is the median of the gas quantity at that snapshot for that
            cluster.
        :param mins: Array of minimum of the quantity for every cluster,
            i.e. an array of shape (100, 352) where every entry is the
            min of the gas quantity at that snapshot for that cluster.
        :param maxs: Array of maximums of the quantity for every cluster,
            i.e. an array of shape (100, 352) where every entry is the
            max of the gas quantity at that snapshot for that cluster.
        :return: None
        """
        plot_types = {
            "Mean": means,
            "Median": medians,
            "Minimum": mins,
            "Maximum": maxs,
        }
        for label_prefix, plot_quantity in plot_types.items():
            logging.info(
                f"Creating line plot for {label_prefix} {self.quantity_label}"
            )
            # create figure and configure axes
            fig, axes = plt.subplots(figsize=(4, 4))
            xs = common.make_redshift_plot(axes, start=constants.MIN_SNAP)
            axes.set_ylabel(f"{label_prefix} {self.quantity_label}")
            axes.set_yscale("log")

            # plot mean, median, etc.
            plot_config = {
                "marker": "none",
                "linestyle": "solid",
                "alpha": 0.1,
                "color": self.color,  # TODO: color by cluster mass
            }
            axes.plot(xs, plot_quantity.transpose(), **plot_config)

            # plot mean and median
            m_config = {"marker": "none", "color": "black"}
            mean = np.mean(plot_quantity, axis=0)
            axes.plot(xs, mean, ls="solid", **m_config)
            median = np.median(plot_quantity, axis=0)
            axes.plot(xs, median, ls="dashed", **m_config)

            # save figure
            self._save_fig(
                fig, ident_flag=label_prefix.lower(), subdir="./1d_plots"
            )

    def _plot_and_save_ridgelineplots(self):
        # TODO: implement
        pass

    def _plot_and_save_2dhistograms(self):
        # TODO: implement
        pass

    @abc.abstractmethod
    def _load_quantity(
        self, snap_num: int, zoom_id: int, group_primaries: NDArray
    ) -> NDArray:
        """
        Abstract method to load a cluster quantity.

        Subclasses to this class must implement this method in such a
        way that it returns the array of the desired quantity for all
        gas particles of the given zoom at the given snapshot.

        :param snap_num: The snapshot to query. Must be a number between
            0 and 99.
        :param zoom_id: The zoom-in region ID. Must be a number between
            0 and 351.
        :param group_primaries: Array of IDs of primary subhalo of every
            cluster at snapshot 99. Useful to trace back cluster
            progenitors through time.
        :return: The gas quantity for every gas cell in the zoom-in at
            that snapshot, such that it can be indexed by the indices
            saved by the generation pipeline.
        """
        pass


# -----------------------------------------------------------------------------
# CONCRETE CLASSES:


class TraceDistancePipeline(TraceSimpleQuantitiesBackABC):
    """
    Trace distance of gas particles to cluster with time.
    """

    def _load_quantity(
        self, snap_num: int, zoom_id: int, group_primaries: NDArray
    ) -> NDArray:
        """
        Find the distance of every gas particle to the cluster.

        The distance must be computed to the current position of the
        cluster.

        :param snap_num: Snapshot to find the distances at.
        :param zoom_id: The ID of the zoom-in region.
        :return: Array of the distances of all gas cells to the cluster
            center.
        """
        # Step 1: find the cluster center

        # Step 2: Load gas particles coordinates

        # Step 3: Calculate distances


class TraceTemperaturePipeline(TraceSimpleQuantitiesBackABC):
    """
    Trace temperature of gas particles with time.
    """

    def _load_quantity(
        self, snap_num: int, zoom_id: int, group_primaries: NDArray
    ) -> NDArray:
        """
        Find the temperature of gas particles in the zoom-in.

        This loads the temperature of all gas cells in the zoom-in region
        at the given snapshot and returns it, such that it can be indexed
        with the pre-saved indices.

        :param snap_num: The snap for which to load temperatures.
        :param zoom_id: The ID of the zoom-in region.
        :param group_primaries: Dummy var, not used.
        :return: Array of the temperatures of all gas cells in the zoom-in
            region
        """
        return gas_daq.get_cluster_temperature(
            self.config.base_path,
            snap_num,
            zoom_id=zoom_id,
        )
