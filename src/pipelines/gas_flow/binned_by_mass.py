"""
Pipeline for velocity distribution plots, binned by cluster mass.
"""
from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Literal

import matplotlib.cm
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np

from library import compute
from library.config import config
from library.data_acquisition import gas_daq, halos_daq
from library.loading import load_velocity_distributions
from library.plotting import plot_gas_velocites
from library.processing import selection
from pipelines.base import DiagnosticsPipeline

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from numpy.typing import NDArray


@dataclass
class MassBinnedVelocityDistributionPipeline(DiagnosticsPipeline):
    """
    Plot the velocity distribution in clusters binned by mass.

    Pipeline creates the velocity distribution histograms for all clusters
    in TNG300-1 and TNG-Cluster by binning them into 0.2 dex mass bins
    and in every nin determining the histogram of velocity distribution
    and plotting it.
    """

    regime: Literal["cool", "warm", "hot"] = "cool"
    log: bool = True

    velocity_bins: ClassVar[int] = 50  # number of velocity bins
    velocity_edges: ClassVar[tuple[float, float]] = (-3000, 3000)  # in km/s
    # edges of mass bins to use (0.2 dex width)
    mass_bin_edges: ClassVar[NDArray] = 10**np.arange(14.0, 15.4, 0.2)
    temperature_regimes: ClassVar[NDArray] = 10**np.array([0, 4.5, 5.5, 50])
    n_clusters: ClassVar[int] = 632  # total number of clusters
    n300: ClassVar[int] = 280  # number of clusters in TNG300-1
    nclstr: ClassVar[int] = 352  # number of clusters in TNG-Cluster

    def __post_init__(self):
        super().__post_init__()
        self.tng300_basepath = config.get_simulation_base_path("TNG300-1")
        self.tngclstr_basepath = config.get_simulation_base_path("TNG-Cluster")
        # translate regime
        if self.regime == "cool":
            self.regime_index = 1
        elif self.regime == "warm":
            self.regime_index = 2
        elif self.regime == "hot":
            self.regime_index = 3
        else:
            logging.fatal(f"Unrecognized regime: {self.regime}.")
            sys.exit(2)

    def run(self) -> int:
        """
        Generate data and create the velocity distribution plot.

        Requires the existence of the pre-calculated velocity files for
        all particles belonging to TNG300-1 halos. These can be created
        with the radial density profile pipeline. Also requires the
        existence of the pre-calculated particle regime indices, which
        can be created with the helper script ``tabulate_cluster_data.py``.

        Steps:

        1. Load halo data for TNG300-1 and TNG-Cluster, place in memory.
        2. Create a mass bin mask for the clusters from their mass.
        3. Allocate memory for the individual histograms.
        4. For every cluster in TNG300-1:
           1. Load radial velocities and particle temperature regime
              from file
           2. Restrict velocities only to cool gas
           3. Calculate the velocity distribution histogram.
           4. Place histogram in the allocated memory.
        5. For every cluster in TNG-Cluster:
           1. Load the particles associated with the cluster.
           2. Calculate the radial velocity for them, discard obsolete
              data.
           3. Find particle temperatures, calculate temperature, mask
              velocities to cool gas only.
           4. Calculate velocity distribution histogram, place into
              allocated array.
        6. Save data to file (histograms and mass bin mask).
        7. Plot the data

        :return:
        """
        # Step 0: create directories, start monitoring
        self._create_directories()

        # Step 1: load halo data (mass)
        halo_masses = np.zeros(self.n_clusters)
        # Load masses for TNG300-1
        halo_data = halos_daq.get_halo_properties(
            self.tng300_basepath,
            self.config.snap_num,
            fields=[self.config.mass_field],
        )
        tng_300_data = selection.select_clusters(
            halo_data, self.config.mass_field, expected_number=self.n300
        )
        halo_masses[:self.n300] = tng_300_data[self.config.mass_field]
        # Load masses for TNG-Cluster
        cluster_fields = [
            self.config.mass_field, self.config.radius_field, "GroupPos"
        ]
        tng_cluster_data = halos_daq.get_halo_properties(
            self.tngclstr_basepath,
            self.config.snap_num,
            fields=cluster_fields,
            cluster_restrict=True,
        )
        halo_masses[self.n300:] = tng_cluster_data[self.config.mass_field]

        # Step 2: create a mass bin mask
        mask = selection.digitize_clusters(halo_masses, self.mass_bin_edges)

        # Step 3: Allocate memory for histograms
        histograms = np.zeros((self.n_clusters, self.velocity_bins))
        edges = np.zeros(self.velocity_bins + 1)

        # Step 4: Loop over TNG300-1 clusters
        for i, halo_id in enumerate(tng_300_data["IDs"]):
            logging.debug(f"Processing halo {halo_id} ({i + 1}/280).")
            # load tabulated data
            regimes = np.load(
                self.config.data_home / "particle_regimes" / "TNG300_1"
                / f"particle_temperature_regimes_halo_{halo_id}.npy"
            )
            velocities = np.load(
                self.config.data_home / "particle_velocities" / "TNG300_1"
                / f"radial_velocity_halo_{halo_id}.npy"
            )
            gas_masses = np.load(
                self.config.data_home / "particle_masses" / "TNG300_1"
                / f"gas_masses_halo_{halo_id}.npy"
            )
            # restrict velocities only to selected temperature regime
            gas_vel_current_regime = velocities[regimes == self.regime_index]
            gas_mass_cur_regime = gas_masses[regimes == self.regime_index]
            # find velocity distribution histogram
            histograms[i], edges = np.histogram(
                gas_vel_current_regime,
                bins=self.velocity_bins,
                range=self.velocity_edges,
                weights=gas_mass_cur_regime,
            )

        # Step 5: Loop over TNG-Cluster clusters
        for i, halo_id in enumerate(tng_cluster_data["IDs"]):
            logging.debug(f"Processing cluster {halo_id} ({i + 1}/352).")
            # load temperatures and gas data required
            temperatures = gas_daq.get_cluster_temperature(
                halo_id, self.tngclstr_basepath, self.config.snap_num
            )
            fields = ["Velocities", "Coordinates", "Masses"]
            gas_data = gas_daq.get_gas_properties(
                self.tngclstr_basepath,
                self.config.snap_num,
                fields=fields,
                cluster=halo_id,
            )
            # mask gas data to temperature regime
            temp_mask = np.digitize(temperatures, self.temperature_regimes)
            masked_data = selection.mask_data_dict(
                gas_data, temp_mask, index=self.regime_index
            )
            del gas_data
            # get the distances of all particles to later restrict to 2R_vir
            distances = np.linalg.norm(
                masked_data["Coordinates"] - tng_cluster_data["GroupPos"][i],
                axis=1
            )
            distances /= tng_cluster_data[self.config.radius_field][i]
            # get radial velocity only within 2R_vir
            radial_vel_in_cluster = compute.get_radial_velocities(
                tng_cluster_data["GroupPos"][i],
                masked_data["Coordinates"][distances <= 2.0],  # limit to 2R
                masked_data["Velocities"][distances <= 2.0],  # limit to 2R
            )
            # calculate gas fractions
            gas_masses_cluster = masked_data["Masses"][distances <= 2.0]
            # calculate histogram
            histograms[self.n300 + i], _ = np.histogram(
                radial_vel_in_cluster,
                bins=self.velocity_bins,
                range=self.velocity_edges,
                weights=gas_masses_cluster,
            )

        # Step 6: Save data to file
        if self.to_file:
            np.savez(
                self.paths["data_dir"] / f"{self.paths['data_file_stem']}.npz",
                histograms=histograms,
                edges=edges,
                halo_masses=halo_masses,
                mass_mask=mask,
            )

        self._plot(histograms, edges, halo_masses, mask)
        return 0

    def _plot(
        self,
        histograms: NDArray,
        edges: NDArray,
        halo_masses: NDArray,
        mass_mask: NDArray
    ) -> None:
        """
        Plot the velocity distribution in 0.2 dex mass bins.

        :param histograms: The array of velocity distribution histograms,
            of shape (632, N) where N is the number of velocity bins.
        :param edges: The array of the edges of the velocity bins, of
            shape (N + 1, ).
        :param halo_masses: The array of the halo masses.
        :param mass_mask: The mask for the masses. Must have length 632
            and contain integers from
        :return: Tuple of figure and axes objects with the plots.
        """
        n_mass_bins = len(self.mass_bin_edges) - 1

        # for every mass bin, plot the data
        for i in range(n_mass_bins):
            fig, axes = self._prepare_figure()
            # mask the histograms to only those in the current mass bin
            mass_bin_idx = i + 1
            current_histograms = selection.mask_quantity(
                histograms, mass_mask, index=mass_bin_idx
            )
            current_masses = selection.mask_quantity(
                halo_masses, mass_mask, index=mass_bin_idx
            )
            current_mean, _ = self._overplot_distribution(
                axes,
                current_histograms,
                edges,
                current_masses,
                i,
                i + 1,
            )
            # add a label describing the mass bin edges
            label = (
                rf"$10^{{{np.log10(self.mass_bin_edges[i]):.1f}}} - "
                rf"10^{{{np.log10(self.mass_bin_edges[i + 1]):.1f}}} M_\odot$"
            )
            axes.text(
                0.97,
                0.97,
                label,
                color="black",
                transform=axes.transAxes,
                horizontalalignment="right",
                verticalalignment="top",
            )
            # add a line at zero
            axes.axvline(0, alpha=1, color="grey", linestyle="solid")
            # add a label for the fraction left and right of zero
            self._add_fraction_labels(axes, current_mean)
            # save figure
            self._save_fig(fig, ident_flag=str(i))

        # plot the total mean and median plus all individuals
        fig, axes = self._prepare_figure()
        total_mean, _ = self._overplot_distribution(
            axes, histograms, edges, halo_masses, 0, -1, alpha=0.01
        )
        axes.text(
            0.97,
            0.97,
            "Total",
            color="black",
            transform=axes.transAxes,
            horizontalalignment="right",
            verticalalignment="top",
        )
        # and a vertical line at zero
        axes.axvline(0, alpha=1, color="grey", linestyle="solid")
        # add label for fractions left and right of zero
        self._add_fraction_labels(axes, total_mean)
        self._save_fig(fig, ident_flag="total")

    def _prepare_figure(self) -> tuple[Figure, Axes]:
        """
        Return a tuple of figure and axes, set up for the current task.

        :return: A figure and an axes object for plotting.
        """
        # create figure and set up axes
        fig, axes = plt.subplots(figsize=(2.5, 2.3))
        fig.set_tight_layout(True)
        axes.set_xlabel("Radial velocity [km/s]")
        axes.set_ylabel(r"Gas mass ($M_\odot$)")
        axes.set_ylim([1e6, 3e11])
        if self.log:
            axes.set_yscale("log")
        return fig, axes

    def _overplot_distribution(
        self,
        axes: Axes,
        histograms: NDArray,
        edges: NDArray,
        masses: NDArray,
        vmin_idx: int,
        vmax_idx: int,
        alpha: float = 0.05,
    ) -> tuple[NDArray, NDArray]:
        """
        Overplot onto the axes all given histograms plus their mean and median.

        :param axes: The axes object onto which to plot the histograms.
        :param histograms: The array of all histograms to plot.
        :param edges: The edges of the histograms.
        :param masses: The masses of the halo belonging to each histogram.
            Must have length equal to the length of the first axis of
            ``histograms``.
        :param vmin_idx: The index of the mass bin edge that demarks the
            lower edge of the current mass bin.
        :param vmax_idx: The index of the mass bin edge that demarks the
            upper edge of the current mass bin.
        :param alpha: The transparency of the individual lines.
        :return: The mean and median histograms respectively.
        """
        # create a colormap for the current mass range
        cmap = matplotlib.cm.get_cmap("plasma")
        norm = matplotlib.colors.Normalize(
            vmin=self.mass_bin_edges[vmin_idx],
            vmax=self.mass_bin_edges[vmax_idx]
        )
        # plot individuals as faint lines
        for j, hist in enumerate(histograms):
            plot_gas_velocites.plot_velocity_distribution(
                axes,
                hist,
                edges,
                color=cmap(norm(masses[j])),
                alpha=alpha,
            )
        # find mean and median, and plot these as prominent lines
        mean = np.nanmean(histograms, axis=0)
        median = np.nanmedian(histograms, axis=0)
        plot_gas_velocites.plot_velocity_distribution(
            axes,
            mean,
            edges,
            color="black",
            label="mean",
        )
        plot_gas_velocites.plot_velocity_distribution(
            axes,
            median,
            edges,
            color="black",
            linestyle="dashed",
            label="median"
        )
        return mean, median

    def _add_fraction_labels(self, axes: Axes, histogram: NDArray) -> None:
        """
        Add labels for the fraction of the histograms above and below zero.

        Method adds two small text boxes around the zero-line which show
        the faction of the given histogram left and right of the zero
        line. For an uneven number of bins, the bin centered on zero is
        not considered.

        The fraction is calculated per halo, and then the mean is written
        to the labels.

        :param axes: The axes object onto which to place the labels.
        :param histogram: The histogram to analyse. Must be centered on
            zero, i.e. there must be equally many bins to either side.
        :return: None
        """
        if self.velocity_bins % 2 == 0:
            middle = int(self.velocity_bins / 2)
            left_frac = np.sum(histogram[:middle])
            right_frac = np.sum(histogram[middle:])
            total = left_frac + right_frac
        else:
            l_border = int(self.velocity_bins // 2)
            r_border = l_border + 1
            left_frac = np.sum(histogram[:l_border])
            right_frac = np.sum(histogram[r_border:])
            total = np.sum(histogram)
        axes.text(
            0.45,
            0.05,
            f"{left_frac / total * 100:.1f}%",
            fontsize=9,
            color="black",
            backgroundcolor=(1, 1, 1, 0.7),
            transform=axes.transAxes,
            horizontalalignment="right",
            verticalalignment="bottom",
        )
        axes.text(
            0.55,
            0.05,
            f"{right_frac / total * 100:.1f}%",
            fontsize=9,
            color="black",
            backgroundcolor=(1, 1, 1, 0.7),
            transform=axes.transAxes,
            horizontalalignment="left",
            verticalalignment="bottom",
        )


class MassBinnedVelocityDistributionFromFilePipeline(
        MassBinnedVelocityDistributionPipeline):
    """
    Load data from file and recreate plots.
    """

    def run(self) -> int:
        """Load data and plot it."""
        self._verify_directories()

        filename = f"{self.paths['data_file_stem']}.npz"
        if not (self.paths["data_dir"] / filename).exists():
            logging.fatal(
                f"Data file {self.paths['data_dir'] / filename} does not "
                f"exist yet. Canceling execution."
            )
            sys.exit(1)
        data = load_velocity_distributions.load_velocity_distributions(
            self.paths["data_dir"] / filename,
            self.velocity_bins,
            self.n_clusters,
        )

        self._plot(*data)
        return 0
