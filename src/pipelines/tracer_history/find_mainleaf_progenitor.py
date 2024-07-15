"""
Pipeline to plot distribution of earliest progenitor for clusters.
"""
from __future__ import annotations

import dataclasses
import logging
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

import illustris_python as il
import matplotlib.pyplot as plt
import numpy as np

from library.config import config
from library.data_acquisition import halos_daq
from library.processing import selection
from pipelines import base

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from numpy.typing import NDArray


class PlotEarliestSnapNumDistributionPipelineCluster(base.Pipeline):
    """
    Plot the distribution of the earliest snap num of cluster primary.

    Pipeline finds for the primary subhalo of every cluster the earliest
    snapshot to which its main progenitor can be traced and plots the
    distribution of snap numbers as a histogram. This is to estimate
    how far back one can go in analysing clusters and their properties,
    before the merger tree "breaks down" due to no distinct main progenitor
    existing anymore.

    This pipeline works only for TNG-Cluster.
    """

    n_clusters: ClassVar[int] = 352
    y_max: ClassVar[float] = 220
    highest_bin: ClassVar[int] = 8

    def run(self) -> int:
        """
        Plot distribution of earliest snap in merger tree.

        :return: Exit code.
        """
        # Step 0: create directories
        self._create_directories(force=True)

        # Step 1: Load cluster data
        cluster_data = self._load_data()

        # Step 2: Load the merger history for every cluster
        earliest_snap = np.zeros((self.n_clusters, ))
        logging.info("Loading merger trees for clusters.")
        for i, first_sub in enumerate(cluster_data["GroupFirstSub"]):
            merger_tree = il.sublink.loadTree(
                self.config.base_path,
                self.config.snap_num,
                first_sub,
                ["SnapNum"],
                onlyMPB=True,
            )
            earliest_snap[i] = merger_tree[-1]

        # Step 3: save data to file
        if self.to_file:
            filename = f"{self.paths['data_file_stem']}.npy"
            np.save(self.paths["data_dir"] / filename, earliest_snap)
            logging.info(f"Saved data to file: {filename}.")

        # Step 3: plot
        f, _ = self._plot(earliest_snap)
        self._save_fig(f, tight_layout=True)
        logging.info("Successfully plotted snapshot distribution!")
        return 0

    def _load_data(self) -> dict[str, NDArray]:
        """
        Load dictionary of cluster data, including ``GroupFirstSub``.

        :return: Dictionary of cluster data.
        """
        cluster_data = halos_daq.get_halo_properties(
            self.config.base_path,
            self.config.snap_num,
            ["GroupFirstSub"],
            cluster_restrict=True,
        )
        return cluster_data

    def _plot(self, first_snaps: NDArray) -> tuple[Figure, Axes]:
        """
        Plot distribution of snap numbers of earliest main progenitor.

        :param first_snaps: Snap num of the earliest occurrence of the
            main progenitor of every clusters primary subhalo.
        :return: Tuple of figure and axes with plot drawn onto them.
        """
        logging.info("Constructing histogram plot.")
        fig, axes = plt.subplots(figsize=(4, 4))
        axes.set_xlabel("Earliest snapshot")
        axes.set_ylabel("Count")
        axes.set_ylim((0, self.y_max))

        bin_edges = np.arange(0, self.highest_bin + 2, step=1) - 0.5
        hist_config = {
            "histtype": "stepfilled",
            "color": "black",
        }
        hist = axes.hist(first_snaps, bin_edges, **hist_config)
        if not np.sum(hist[0]) == self.n_clusters:
            logging.warning(
                f"Not all clusters are shown in the histogram! Missing "
                f"{int(self.n_clusters - np.sum(hist[0]))} clusters."
            )

        # Add labels to the individual bars
        bin_centers = bin_edges[:-1] + 0.5
        for x, y in zip(bin_centers, hist[0]):
            axes.text(x, y + 8, f"{int(y)}", ha="center")

        return fig, axes


class PlotEarliestSnapNumDistributionPipeline300(
        PlotEarliestSnapNumDistributionPipelineCluster):
    """
    Plot the distribution of the earliest snap num of cluster primary.

    Pipeline finds for the primary subhalo of every cluster the earliest
    snapshot to which its main progenitor can be traced and plots the
    distribution of snap numbers as a histogram. This is to estimate
    how far back one can go in analysing clusters and their properties,
    before the merger tree "breaks down" due to no distinct main progenitor
    existing anymore.

    This pipeline works only for TNG300.
    """

    n_clusters: ClassVar[int] = 280
    y_max: ClassVar[float] = 175
    highest_bin: ClassVar[int] = 10

    def _load_data(self) -> dict[str, NDArray]:
        """
        Load cluster data for TNG300-1.

        :return: Dict of cluster data, including ``GroupFirstSub``.
        """
        halo_data = halos_daq.get_halo_properties(
            self.config.base_path,
            self.config.snap_num,
            ["GroupFirstSub", self.config.mass_field],
        )
        cluster_data = selection.select_clusters(
            halo_data,
            self.config.mass_field,
            expected_number=self.n_clusters,
        )
        return cluster_data


@dataclasses.dataclass
class CombinedEarliestSnapshotDistributionPipeline(base.Pipeline):
    """
    Plot the distribution of earliest snapshot for both sims combined.
    """

    n_clusters: ClassVar[int] = 632
    y_max: ClassVar[float] = 400
    highest_bin: ClassVar[int] = 10

    def __post_init__(self):
        self.tngclstr_basepath = config.get_simulation_base_path("TNG-Cluster")
        self.tng300_basepath = config.get_simulation_base_path("TNG300-1")

    def run(self) -> int:
        """
        Plot distribution for both TNG-Cluster and TNG300-1.

        :return: Exit code.
        """
        logging.info("Plotting combined histogram.")

        # Step 0: create directories
        self._create_directories(force=True)

        # Step 1: load TNG-Cluster data
        filename = f"{self.paths['data_file_stem']}_TNG_Cluster.npy"
        filepath = self.paths['data_dir'] / "TNG_Cluster"
        if not Path(filepath / filename).exists():
            logging.fatal("Missing data for TNG-Cluster. Aborting.")
            return 1
        earliest_snaps_clstr = np.load(filepath / filename)

        # Step 2: load TNG300 data
        filename = f"{self.paths['data_file_stem']}_TNG300_1.npy"
        filepath = self.paths['data_dir'] / "TNG300_1"
        if not Path(filepath / filename).exists():
            logging.fatal("Missing data for TNG300-1. Aborting.")
            return 1
        earliest_snaps_300 = np.load(filepath / filename)

        # Step 3: plot distribution
        f, _ = self._plot(earliest_snaps_clstr, earliest_snaps_300)
        self._save_fig(f, tight_layout=True)
        return 0

    def _plot(self, snaps_tng_cluster: NDArray,
              snaps_tng_300: NDArray) -> tuple[Figure, Axes]:
        """
        Plot the two distributions stacked.

        :param snaps_tng_cluster: List of earliest snaps for TNG-Cluster.
        :param snaps_tng_300: List of earliest snaps for TNG300-1.
        :return: Figure and axes with plot drawn onto them.
        """
        logging.info("Constructing histogram plot.")
        fig, axes = plt.subplots(figsize=(4, 4))
        axes.set_xlabel("Earliest snapshot")
        axes.set_ylabel("Count")
        axes.set_ylim((0, self.y_max))

        # reconstruct histograms
        bin_edges = np.arange(0, self.highest_bin + 2, step=1) - 0.5
        hist_config = {
            "histtype": "stepfilled",
            "color": ["#992160", "navy"],
            "label": ["TNG-Cluster", "TNG300-1"],
            "stacked": True,
        }
        hist, _, _ = axes.hist(
            [snaps_tng_cluster, snaps_tng_300],
            bin_edges,
            **hist_config,
        )

        # add labels of number per bin
        bin_centers = bin_edges[:-1] + 0.5
        for i in range(self.highest_bin + 1):
            # TNG-Cluster
            axes.text(
                bin_centers[i],
                hist[1][i] + 8,
                f"{int(hist[0][i])}",
                color=hist_config["color"][0],
                ha="center",
            )
            # TNG-Cluster
            axes.text(
                bin_centers[i],
                hist[1][i] + 30,
                f"{int(hist[1][i] - hist[0][i])}",
                color=hist_config["color"][1],
                ha="center",
            )

        # add a legend
        axes.legend()

        return fig, axes
