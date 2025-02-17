"""
Combine distance and temperature development into one 4D histogram.
"""
from __future__ import annotations

import dataclasses
import logging
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

import h5py
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import yaml

from library import constants
from library.plotting import common
from pipelines import base

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from numpy.typing import NDArray


@dataclasses.dataclass
class PlotStackedHistogramPipeline(base.Pipeline):
    """
    Plot the distance vs. redshift histogram, with temperature as color.

    The histogram encodes both the mass distribution and the mean
    temperature per bin into color information by making the mass per
    bin decide the value of the HSV color, and temperature decide the
    hue of the color.
    """

    individuals: bool = True  # whether to plot a histogram for every cluster

    n_clusters: ClassVar[int] = 352
    n_snaps: ClassVar[int] = 100 - constants.MIN_SNAP
    n_bins: ClassVar[int] = 50  # number of bins

    def __post_init__(self):
        super().__post_init__()

        # update config for plot
        cfg = Path(__file__).parent / "simple_quantities_plot_config.yaml"
        with open(cfg, "r") as cfg_file:
            stream = cfg_file.read()
        try:
            plot_config = yaml.full_load(stream)["DistanceToMP"]["standard"]
            self.hist_range = plot_config["min"], plot_config["max"]
            self.y_label = plot_config["label"]
            self.log = plot_config["log"]
        except KeyError:
            logging.error(
                "Found no plot config for quantity DistanceToMP, or the"
                "config is incomplete. Will set no boundaries for histograms;"
                " 2D plot creation will be skipped."
            )
            raise KeyError("No entry for DistanceToMP in config file.")

    def run(self) -> int:
        """
        Load particle data, and create a 4D histogram.

        :return: Exit code.
        """
        logging.info(
            "Starting pipeline to plot distance histogram, colored by "
            "temperature and tracer mass."
        )

        # Step 0: check archive exists
        if not self.config.cool_gas_history.exists():
            logging.fatal(
                f"Did not find cool gas archive file "
                f"{self.config.cool_gas_history}."
            )
            return 1

        # Step 1: open the archive and load the distance and temperature
        f = h5py.File(self.config.cool_gas_history, "r")
        mhist, thist, dedges, zedges = self._get_hists(f)
        f.close()

        # Step 2: find mean of histograms
        mhist_mean = np.nanmean(mhist, axis=0)
        thist_mean = np.nanmean(thist, axis=0)

        # Step 3: save plot data to file
        # TODO

        # Step 4: plot distribution for individual clusters
        if self.individuals:
            pass  # TODO

        # Step 5: plot the figure for the overall mean
        fig, _ = self._plot(mhist_mean, thist_mean, zedges, dedges)
        self._save_fig(fig)
        logging.info("Succesfully saved histogram to file!")
        return 0

    def _plot(
        self,
        masses: NDArray,
        temperatures: NDArray,
        zedges: NDArray,
        dedges: NDArray,
    ) -> tuple[Figure, Axes]:
        """
        Plot the 4D histogram of the given masses and temperatures.

        :param masses: Histogram weighted by tracer mass.
        :param temperatures: Histogram of mean gas temperature.
        :param zedges: Array of bin edges in redshift.
        :param dedges: Array of bin edges in distance.
        :return: Tuple of figure and axes object with the histogram
            drawn onto them.
        """
        fig, axes = plt.subplots(figsize=(5, 5))
        y_label = self.y_label
        if self.log:
            y_label = y_label.replace("[", r"[$\log_{10}$")
        axes.set_ylabel(y_label)

        # clip value and hue range to avoid white spots
        masses[np.isnan(masses)] = 1e7
        crange = (3, 9, 7, np.log10(np.nanmax(masses)))  # in log scale
        common.plot_4d_histogram(
            axes,
            temperatures.transpose(),
            masses.transpose(),
            zedges,
            dedges,
            hue_scale="log",
            value_scale="log",
            color_range=crange,
            hue_label=r"Temperature [$\log_{10} K$]",
            value_label=r"Tracer mass [$\log_{10} M_\odot$]",
            cbar_labelsize="x-small",
            cbar_linecolor="white",
            nan_color=(0, 0, 0),
        )
        # label x-axis correctly
        common.label_snapshots_with_redshift(
            axes,
            constants.MIN_SNAP,
            99,
            tick_positions_z=np.array([0, 0.1, 0.5, 1, 2, 5]),
            tick_positions_t=np.array([0, 1, 5, 8, 11, 13]),
        )
        return fig, axes

    def _get_hists(
        self, archive_file: h5py.File
    ) -> tuple[NDArray, NDArray, NDArray, NDArray]:
        """
        Create 2D histograms for distance vs. redshift.

        Function loads distance and temperature from the given archive
        file, and creates two histograms of distance weighted by tracer
        mass and temperature of it at every snapshot. The two histograms
        are returned, alongside their common edges.

        :param archive_file: Opened cool gas history archive file.
        :return: Array of histograms of the distribution of the quantity
            at all snapshots analyzed for all clusters. Suitably
            normalized for the given quantity.
        """
        logging.info(
            "Creating histograms for all clusters. This can take a while."
        )

        mass_hist = np.zeros((self.n_clusters, self.n_snaps, self.n_bins))
        temp_hist = np.zeros_like(mass_hist)

        edges = None
        for zoom_id in range(self.n_clusters):
            logging.debug(f"Creating histograms for zoom-in {zoom_id}.")
            group = f"ZoomRegion_{zoom_id:03d}"
            distance = archive_file[group]["DistanceToMP"]
            if self.log:
                distance = np.log10(distance)
                distance = np.clip(distance, *self.hist_range)
            temperature = archive_file[group]["Temperature"]
            for i, snap in enumerate(range(constants.MIN_SNAP, 100, 1)):
                # histogram weighted by tracer mass
                weights = np.ones_like(distance[snap]) * constants.TRACER_MASS
                m_hist, edges, _ = scipy.stats.binned_statistic(
                    distance[snap],
                    weights,
                    statistic="sum",
                    bins=self.n_bins,
                    range=self.hist_range,
                )
                mass_hist[zoom_id, i] = m_hist
                # histogram weighted by temperature
                t_hist, _, _ = scipy.stats.binned_statistic(
                    distance[snap],
                    temperature[snap],
                    statistic="mean",
                    bins=self.n_bins,
                    range=self.hist_range,
                )
                temp_hist[zoom_id, i] = t_hist
        # create edges for the z-axis ("edges" around snapshots)
        z_edges = np.linspace(constants.MIN_SNAP, 100, num=93) - 0.5
        return mass_hist, temp_hist, edges, z_edges
