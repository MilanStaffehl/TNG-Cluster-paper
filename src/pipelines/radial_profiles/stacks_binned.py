"""
Pipeline for stacking radial profiles.
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

from library.loading import load_radial_profiles
from library.plotting import common, plot_radial_profiles, pltutil
from library.processing import selection, statistics
from pipelines.base import Pipeline

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from numpy.typing import NDArray


@dataclass
class StackProfilesBinnedPipeline(Pipeline):
    """
    Pipeline to create stacks of radial profiles, binned by halo mass.

    Requires the data files for profiles of TNG300-1 and TNG Cluster to
    already exist, since they will be loaded.
    """

    log: bool
    what: str
    method: Literal["mean", "median"]

    # edges of mass bins to use (0.2 dex width)
    mass_bins: ClassVar[NDArray] = 10**np.arange(14.0, 15.4, 0.2)
    n_clusters: ClassVar[int] = 632  # total number of clusters

    def run(self) -> int:
        """
        Plot radial profiles for different mass bins of the given type.

        This will load the existing profile data from both TNG300-1 and
        TNG Cluster and bin the histograms by halo mass. Then, plots
        will be created, containing the mean or median profile within
        each mass bin, plus a total profile, containing all clusters of
        both simulations (i.e. halos with log M > 14).

        Temperature will be plotted as two 2x4 plots of 2D histograms,
        the first one will show the mean/median temperature profile in
        the 7 mass bins, plus the total across all mass bins; the second
        will show the error. Density will be plotted as one single plot
        with eight lines (including confidence region) which represent
        the seven mass bins (colored) and the total (black), as well as
        eight dotted lines showing the same information but only for the
        cool gas of the clusters.

        Steps:

        1. Allocate space in memory for histogram data.
        2. Load halo data for TNG300-1
        3. Load halo data for TNG Cluster
        4. Create a mass bin mask.
        5. Allocate memory space for mean/median histograms per mass bin
           and a total over all mass bins (8 histograms)
        6. For every mass bin:
           1. Mask histogram data
           2. Calculate mean/median profile with errors
           3. Place data in allocated memory
        7. Create a total mean/median histogram with errors; place it
           into allocated memory.
        8. Save data to file.
        9. Plot the data according to chosen type.

        :return: Exit code.
        """
        # Step 0: verify directories
        self._verify_directories()

        # Step 1 - 3: Load data (depends on what type of data, so this
        # task is split):
        logging.info("Loading halo data from TNG300-1 and TNG Cluster.")
        if self.what == "temperature":
            cluster_data = self._load_temperature_hists()
        elif self.what == "density":
            cluster_data = self._load_density_hists()
        else:
            logging.fatal(f"Unrecognized plot type {self.what}. Aborting.")
            return 1
        # unpack
        cluster_masses = cluster_data["masses"]
        cluster_histograms = cluster_data["histograms"]
        edges = cluster_data["edges"]

        # Step 4: create mass bin mask
        logging.info("Creating a mass bin mask.")
        n_mass_bins = len(self.mass_bins) - 1
        mask = np.digitize(cluster_masses, self.mass_bins)

        # Step 5: allocate memory for mean/median histograms
        # cluster_histograms[0].shape will be one of the following:
        # - (50, 50) for temperature (shape of histogram)
        # - (2, 50) for density (being split into total and only cool gas)
        stacks = np.zeros((n_mass_bins + 1, ) + cluster_histograms[0].shape)
        """First 7 entries: mass bins. Last entry: total."""
        errors = np.zeros(
            (n_mass_bins + 1, 2,) + cluster_histograms[0].shape
        )  # yapf: disable
        """Axes: mass bin, lower/upper error, values"""

        # Step 6: loop over mass bins and create stacks
        logging.info("Start stacking histograms per mass bin.")
        if self.what == "temperature":
            stacking_func = self._stack_temperature_hists
        else:
            stacking_func = self._stack_density_hists
        # loop over mass bins
        for i in range(n_mass_bins):
            # mask histogram data
            hists_in_bin = selection.mask_quantity(
                cluster_histograms, mask, index=(i + 1)
            )
            stacks[i], errors[i] = stacking_func(hists_in_bin)

        # Step 7: create a total mean/median profile
        stacks[-1], errors[-1] = stacking_func(cluster_histograms)

        # Step 8: save data to file
        if self.to_file:
            pass  # not implemented

        # Step 9: plot the data
        logging.info(f"Plotting {self.what} {self.method} stacks.")
        if self.what == "temperature":
            f, _ = self._plot_temperature_stacks(stacks, errors, edges)
        else:
            f, _ = self._plot_density_stacks(stacks, errors, edges)
        logging.info(f"Saving {self.what} {self.method} profile to file.")
        self._save_fig(f, ident_flag=self.method)

        return 0

    def _load_temperature_hists(self) -> dict[str, NDArray]:
        """
        Load temperature histograms, edges and halo masses.

        The returned values are a concatenation of TNG300-1 and TNG
        Cluster data, in that order (meaning the first 280 entries of
        the data arrays of the 'masses' and 'histograms' fields belong
        to TNG300-1, the remaining 352 belong to TNG Cluster).

        :return: Mapping of loaded data. Has as keys 'masses', 'histograms',
            and 'edges', with the values representing the cluster masses,
            the temperature histograms and the edges of the histograms
            for all clusters of TNG300-1 and TNG Cluster.
        """
        # determine shape and edges
        test_path = (
            self.paths["data_dir"] / "TNG300_1" / "temperature_profiles"
        )
        filepath = list(test_path.iterdir())[0]
        with np.load(filepath.resolve()) as test_file:
            shape = test_file["original_histogram"].shape
            edges = np.array(
                [
                    test_file["xedges"][0],
                    test_file["xedges"][-1],
                    test_file["yedges"][0],
                    test_file["yedges"][-1],
                ]
            )

        # allocate memory
        masses = np.zeros(self.n_clusters)
        hists = np.zeros((self.n_clusters, ) + shape)

        # load TNG300-1 data
        load_generator = load_radial_profiles.load_individuals_2d_profile(
            self.paths["data_dir"] / "TNG300_1" / "temperature_profiles",
            shape,  # automatically validates shapes
        )
        n_tng300_clusters = 0
        for i, halo_data in enumerate(load_generator):
            masses[i] = halo_data["halo_mass"]
            hists[i] = halo_data["original_histogram"]
            n_tng300_clusters += 1

        # load TNG Cluster data and verify it
        load_generator = load_radial_profiles.load_individuals_2d_profile(
            self.paths["data_dir"] / "TNG_Cluster" / "temperature_profiles",
            shape,  # automatically validates shapes
        )
        for i, halo_data in enumerate(load_generator):
            if i == 0:
                # verify both simulation data were saved with the same shape
                cledges = np.array(
                    [
                        halo_data["xedges"][0],
                        halo_data["xedges"][-1],
                        halo_data["yedges"][0],
                        halo_data["yedges"][-1],
                    ]
                )
                if not np.allclose(edges, cledges):
                    logging.fatal(
                        f"Temperature histograms for TNG300-1 and TNG-Cluster "
                        f"have different bin edges:\nTNG300: {edges}\n"
                        f"TNG-Cluster: {cledges}"
                    )
                    sys.exit(2)
            masses[i + n_tng300_clusters] = halo_data["halo_mass"]
            hists[i + n_tng300_clusters] = halo_data["original_histogram"]

        # construct and return mapping
        return {"masses": masses, "histograms": hists, "edges": edges}

    def _load_density_hists(self) -> dict[str, NDArray]:
        """
        Load density histograms, edges and halo masses.

        The returned values are a concatenation of TNG300-1 and TNG
        Cluster data, in that order (meaning the first 280 entries of
        the data arrays of the 'masses' and 'histograms' fields belong
        to TNG300-1, the remaining 352 belong to TNG Cluster).

        .. attention::
            The field 'histograms' perhaps unexpectedly has shape (2, N).
            This is due to the fact that the first entry of this array
            is the total histogram for all gas and the second entry is
            the density profile of the cool gas only!

        :return: Mapping of loaded data. Has as keys 'masses', 'histograms',
            and 'edges', with the values representing the cluster masses,
            the density histograms and the edges of the histograms for
            all clusters of TNG300-1 and TNG Cluster.
        """
        # determine shape and edges
        test_path = (self.paths["data_dir"] / "TNG300_1" / "density_profiles")
        filepath = list(test_path.iterdir())[0]
        with np.load(filepath.resolve()) as test_file:
            shape = test_file["total_histogram"].shape
            edges = np.array(test_file["edges"])

        # allocate memory
        masses = np.zeros(self.n_clusters)
        hists = np.zeros((self.n_clusters, 2, ) + shape)  # yapf: disable

        # load TNG300-1 data
        load_generator = load_radial_profiles.load_individuals_1d_profile(
            self.paths["data_dir"] / "TNG300_1" / "density_profiles",
            shape,  # automatically validates shapes
        )
        n_tng300_clusters = 0
        for i, halo_data in enumerate(load_generator):
            masses[i] = halo_data["halo_mass"]
            hists[i][0] = halo_data["total_histogram"]
            hists[i][1] = halo_data["cool_histogram"]
            n_tng300_clusters += 1

        # load TNG Cluster data and verify it
        load_generator = load_radial_profiles.load_individuals_1d_profile(
            self.paths["data_dir"] / "TNG_Cluster" / "density_profiles",
            shape,  # automatically validates shapes
        )
        for i, halo_data in enumerate(load_generator):
            if i == 0:
                if not np.allclose(halo_data["edges"], edges):
                    logging.fatal(
                        f"Density histograms for TNG300-1 and TNG-Cluster "
                        f"have different bin edges:\nTNG300: {edges}\n"
                        f"TNG-Cluster: {halo_data['edges']}"
                    )
                    sys.exit(2)
            masses[i + n_tng300_clusters] = halo_data["halo_mass"]
            hists[i + n_tng300_clusters][0] = halo_data["total_histogram"]
            hists[i + n_tng300_clusters][1] = halo_data["cool_histogram"]

        # construct and return mapping
        return {"masses": masses, "histograms": hists, "edges": edges}

    def _stack_temperature_hists(
        self,
        histograms: NDArray,
    ) -> tuple[NDArray, NDArray]:
        """
        Stack the temperature histograms given according to self.method.

        The method returns as first array the stack, and as second array
        a shape (2, X, Y) array, where X, Y is the shape of the temperature
        histogram array. The first entry is the lower error and the
        second entry is the upper error.

        Note that the stacks are being column-wise normalised such that
        the sum of every column is always unity.

        :param histograms: The array of histograms of shape (X, Y) to
            stack.
        :return: Tuple of the stacked histogram (shape (X, Y)) and the
            error (shape (2, X, Y)).
        """
        stack, low_err, upp_err = statistics.stack_histograms(histograms, self.method)
        # column-normalise the stack
        stack_normalized, _, _ = statistics.column_normalized_hist2d(
            stack, None, None, normalization="density"
        )
        return stack_normalized, np.array([low_err, upp_err])

    def _stack_density_hists(
        self,
        histograms: NDArray,
    ) -> tuple[NDArray, NDArray]:
        """
        Stack the temperature histograms given according to self.method.

        The method expects ``histograms`` to be an array of shape
        (N, 2, X) where N is the number of clusters in the mass bin to
        stack, and X is the number of radial bins. The second axis is
        supposed to split the histograms into a total profile (index 0)
        and a cool-gas-only profile (index 1), i.e. ``histograms[i][0]``
        selects the total density profile of the i-th cluster, while
        ``histograms[i][1]`` selects the density profile for only the
        cool gas of the i-th halo.

        :param histograms: The array of histograms of shape (2, X) to
            stack. Along the first axis, the first entry must be the
            total gas density profile, the second entry must be the
            cool-gas-only density profile.
        :return: Tuple of the stacked histogram (shape (2, Y)) and the
            error (shape (2, 2, X)). For the error, the axes are
            assigned as (total/cool gas, lower/upper error, bins).
        """
        # splice input array [all halos, total/cool only, full histogram]
        total_hists = histograms[:, 0, :]
        cool_gas_hists = histograms[:, 1, :]
        # stack arrays separately
        total_stack, total_lowerr, total_upperr = statistics.stack_histograms(
            total_hists, self.method
        )
        cool_stack, cool_lowerr, cool_upperr = statistics.stack_histograms(
            cool_gas_hists, self.method
        )
        # construct expected return array shape
        stack = np.array([total_stack, cool_stack])
        errors = np.array(
            [[total_lowerr, total_upperr], [cool_lowerr, cool_upperr]]
        )
        return stack, errors

    def _plot_temperature_stacks(
        self, stacks: NDArray, errors: NDArray, edges: NDArray
    ) -> tuple[Figure, Axes]:
        """
        Plot the stacked temperature profiles.

        :param stacks: The array of N + 1 stacked temperature histograms,
            where N is the number of mass bins. Histograms are expected
            to be 2D of shape (X, Y). Last entry of the array (i.e. with
            index N + 1) is expected to be the total stack.
        :param errors: The array of errors on the stacks, of shape
            (N + 1, 2, X, Y).
        :param edges: The edges of the histograms, [xmin, xmax, ymin, ymax].
        :return: The figure and axes with the plot.
        """
        ncols = len(stacks) // 2
        fig, axes = plt.subplots(
            nrows=2,
            ncols=ncols,
            figsize=(ncols * 1.8 + 1.2, 4),
            sharex=True,
            sharey=True,
            gridspec_kw={"hspace": 0, "wspace": 0},
            layout="constrained",
        )
        # fig.set_tight_layout(True)
        flat_axes = axes.flatten()
        # common axes labels
        fig.supxlabel(r"Distance from halo center [$R_{200c}$]")
        fig.supylabel(r"Temperature [$\log K$]")

        if self.log:
            clabel = r"Normalized mean gas fraction ($\log_{10}$)"
            value_range = (-4, np.log10(np.max(stacks)))
            text_pos = (0.1, 3.3)
        else:
            clabel = "Normalized mean gas fraction"
            value_range = (np.min(stacks), np.max(stacks))
            text_pos = (0, 0)

        for i in range(len(stacks)):
            # plot histograms
            with np.errstate(invalid="ignore", divide="ignore"):
                plot_radial_profiles.plot_2d_radial_profile(
                    fig,
                    flat_axes[i],
                    stacks[i],
                    edges,
                    xlabel=None,
                    ylabel=None,
                    cbar_label=clabel,
                    cbar_limits=[-4, None] if self.log else None,
                    scale="log" if self.log else "linear",
                    value_range=value_range,
                    suppress_colorbar=True,
                )
            # running average
            running_average = statistics.get_2d_histogram_running_average(
                stacks[i], edges[-2:]
            )
            plot_radial_profiles.overplot_running_average(
                fig,
                flat_axes[i],
                running_average,
                edges,
                suppress_label=True,
            )
            # label with mass bin
            if i == len(stacks) - 1:
                label = "Total"
            else:
                label = (
                    rf"$10^{{{np.log10(self.mass_bins[i]):.1f}}} - "
                    rf"10^{{{np.log10(self.mass_bins[i + 1]):.1f}}}$"
                )
            flat_axes[i].text(*text_pos, label, color="white")

        # add a colorbar
        norm = matplotlib.colors.Normalize(*value_range)
        fig.colorbar(
            matplotlib.cm.ScalarMappable(norm=norm, cmap="inferno"),
            ax=axes.ravel().tolist(),
            aspect=20,
            pad=0.03,
            label=clabel,
            extend="min" if self.log else "neither",
        )
        return fig, axes

    def _plot_density_stacks(
        self, stacks: NDArray, errors: NDArray, edges: NDArray
    ) -> tuple[Figure, Axes]:
        """
        Plot the stacked density profiles.

        :param stacks: The array of N + 1 stacked density histograms,
            where N is the number of mass bins. Histograms are expected
            to be 2D of shape (2, X). The first axes splits the 1D
            histograms into total stack (first entry) and cool gas only
            stack (second entry). X is the number of radial bins.
        :param errors: The array of errors on the stacks, of shape
            (N + 1, 2, 2, X). This corresponds to the following quantities:
            (bin/total, total/cool-only, lower/upper error, bin).
        :param edges: The edges of the histograms, [xmin, xmax].
        :return: The figure and axes with the plot.
        """
        fig, axes = plt.subplots(figsize=(5, 5))
        axes.set_xlabel(r"Distance from halo center [$R_{200c}$]")
        axes.set_ylabel(r"Mean density in radial shell [$M_\odot / kpc^3$]")
        if self.log:
            axes.set_yscale("log")

        xs = (edges[:-1] + edges[1:]) / 2

        # plot mass bins
        for i in range(len(stacks)):
            if i == len(stacks) - 1:
                color = "black"
                label = "Total"
            else:
                color = pltutil.sample_cmap("jet", len(stacks) - 1, i)
                label = (
                    rf"$10^{{{np.log10(self.mass_bins[i]):.1f}}} - "
                    rf"10^{{{np.log10(self.mass_bins[i + 1]):.1f}}}$"
                )

            # error config
            if self.method == "mean":
                total_errors = errors[i][0]
                cool_errors = errors[i][1]
            elif self.method == "median":
                total_errors = pltutil.get_errorbar_lengths(
                    stacks[i][0], [errors[i][0][0], errors[i][0][1]]
                )
                cool_errors = pltutil.get_errorbar_lengths(
                    stacks[i][1], [errors[i][1][0], errors[i][1][1]]
                )
            else:
                logging.fatal(f"Unrecognised plot method {self.method}.")
                sys.exit(4)

            # total as solid line
            common.plot_curve_with_error_region(
                xs,
                stacks[i][0],
                x_err=None,
                y_err=total_errors,
                axes=axes,
                linestyle="solid",
                color=color,
                label=label,
                suppress_error_line=True,
                suppress_error_region=False,
            )
            # plot cool gas only as dashed line
            common.plot_curve_with_error_region(
                xs,
                stacks[i][1],
                x_err=None,
                y_err=cool_errors,
                axes=axes,
                linestyle="dashed",
                color=color,
                suppress_error_line=True,
                suppress_error_region=False,
            )

        axes.legend(
            loc="lower center",
            bbox_to_anchor=(0.5, 1.1),
            ncol=len(stacks) // 2,
        )
        return fig, axes


@dataclass
class StackDensityProfilesCombinedPipeline(StackProfilesBinnedPipeline):
    """
    Pipeline to create stacks of radial profiles, binned by halo mass.

    Pipeline produces a plot with both mean and median shown in the
    same plot.

    Requires the data files for profiles of TNG300-1 and TNG Cluster to
    already exist, since they will be loaded.
    """

    def run(self) -> int:
        """
        Plot radial density profiles for different mass bins.

        This will load the existing profile data from both TNG300-1 and
        TNG Cluster and bin the histograms by halo mass. Then, a plot
        will be created, containing the mean AND median profile within
        each mass bin, plus a total profile, containing all clusters of
        both simulations (i.e. halos with log M > 14).

       The plot will be a single plot containing the mean lines for the
       total gas density and the cool density, as well as lines for the
       median cool gas density.

        Steps:

        1. Allocate space in memory for histogram data.
        2. Load halo data for TNG300-1
        3. Load halo data for TNG Cluster
        4. Create a mass bin mask.
        5. Allocate memory space for mean and median histograms per mass
           bin and a total over all mass bins (8 histograms)
        6. For every mass bin:
           1. Mask histogram data
           2. Calculate mean and median profile with errors
           3. Place data in allocated memory
        7. Create a total mean/median histogram with errors; place it
           into allocated memory.
        8. Plot the data according to chosen type.

        :return: Exit code.
        """
        # Step 0: verify directories
        self._verify_directories()

        # Step 1 - 3: Load data (depends on what type of data, so this
        # task is split):
        logging.info("Loading halo data from TNG300-1 and TNG Cluster.")
        cluster_data = self._load_density_hists()

        # unpack
        cluster_masses = cluster_data["masses"]
        cluster_histograms = cluster_data["histograms"]
        """Shape: (H, 2, N) for H = number of halos, N = number of bins"""
        edges = cluster_data["edges"]

        # Step 4: create mass bin mask
        logging.info("Creating a mass bin mask.")
        n_mass_bins = len(self.mass_bins) - 1
        mask = np.digitize(cluster_masses, self.mass_bins)

        # Step 5: allocate memory for mean/median histograms
        stacks = np.zeros(
            (
                n_mass_bins + 1,
                3,
            ) + cluster_histograms[0, 0].shape
        )
        """Axes: mass bin, total/mean cool/median cool, histogram bins"""
        errors = np.zeros(
            (n_mass_bins + 1, 3, 2, ) + cluster_histograms[0, 0].shape
        )  # yapf: disable
        """
        Axes: mass bin, total/mean cool/median cool, lower/upper error, values
        """

        # Step 6: loop over mass bins and create stacks
        logging.info("Start stacking histograms per mass bin.")
        # loop over mass bins
        for i in range(n_mass_bins):
            # mask histogram data
            hists_in_bin = selection.mask_quantity(
                cluster_histograms, mask, index=(i + 1)
            )
            stacks[i], errors[i] = self._stack_density_hists(hists_in_bin)

        # Step 7: create a total mean/median profile
        stacks[-1], errors[-1] = self._stack_density_hists(cluster_histograms)

        # Step 8: plot the data
        logging.info("Plotting combined density stacks.")
        f, _ = self._plot_density_stacks(stacks, errors, edges)
        logging.info("Saving combined radial density profile to file.")
        self._save_fig(f, ident_flag="combined")

        return 0

    def _stack_density_hists(
        self,
        histograms: NDArray,
    ) -> tuple[NDArray, NDArray]:
        """
        Stack the temperature histograms given according to self.method.

        The method expects ``histograms`` to be an array of shape
        (N, 2, X) where N is the number of clusters in the mass bin to
        stack, and X is the number of radial bins. The second axis is
        supposed to split the histograms into a total profile (index 0)
        and a cool-gas-only profile (index 1), i.e. ``histograms[i][0]``
        selects the total density profile of the i-th cluster, while
        ``histograms[i][1]`` selects the density profile for only the
        cool gas of the i-th halo.

        :param histograms: The array of histograms of shape (2, X) to
            stack. Along the first axis, the first entry must be the
            total gas density profile, the second entry must be the
            cool-gas-only density profile.
        :return: Tuple of the stacked histogram (shape (2, Y)) and the
            error (shape (2, 2, X)). For the error, the axes are
            assigned as (total/cool gas, lower/upper error, bins).
        """
        # splice input array [all halos, total/cool only, full histogram]
        total_hists = histograms[:, 0, :]
        cool_gas_hists = histograms[:, 1, :]
        # stack arrays separately
        total_stack, total_lowerr, total_upperr = statistics.stack_histograms(
            total_hists, "mean"
        )
        cool_mean, cool_lowstd, cool_uppstd = statistics.stack_histograms(
            cool_gas_hists, "mean"
        )
        cool_median, cool_lowerr, cool_uperr = statistics.stack_histograms(
            cool_gas_hists, "median"
        )
        # construct expected return array shape
        stack = np.array([total_stack, cool_mean, cool_median])
        errors = np.array(
            [
                [total_lowerr, total_upperr],
                [cool_lowstd, cool_uppstd],
                [cool_lowerr, cool_uperr],
            ]
        )
        return stack, errors

    def _plot_density_stacks(
        self, stacks: NDArray, errors: NDArray, edges: NDArray
    ) -> tuple[Figure, Axes]:
        """
        Plot the stacked density profiles.

        :param stacks: The array of N + 1 stacked density histograms,
            where N is the number of mass bins. Histograms are expected
            to be 2D of shape (2, X). The first axes splits the 1D
            histograms into total stack (first entry) and cool gas only
            stack (second entry). X is the number of radial bins.
        :param errors: The array of errors on the stacks, of shape
            (N + 1, 2, 2, X). This corresponds to the following quantities:
            (bin/total, total/cool-only, lower/upper error, bin).
        :param edges: The edges of the histograms, [xmin, xmax].
        :return: The figure and axes with the plot.
        """
        fig, axes = plt.subplots(figsize=(4.5, 4.5))
        axes.set_xlabel(r"Distance from halo center [$R_{200c}$]")
        axes.set_ylabel(r"Mean density in radial shell [$M_\odot / kpc^3$]")
        axes.set_ylim((1e-2, 1e6))
        if self.log:
            axes.set_yscale("log")

        xs = (edges[:-1] + edges[1:]) / 2

        # plot mass bins
        for i in range(len(stacks)):
            if i == len(stacks) - 1:
                color = "black"
                label = "Total"
            else:
                color = pltutil.sample_cmap("jet", len(stacks) - 1, i)
                label = (
                    rf"$10^{{{np.log10(self.mass_bins[i]):.1f}}} - "
                    rf"10^{{{np.log10(self.mass_bins[i + 1]):.1f}}}$"
                )

            # error config
            total_errors = errors[i][0]
            cool_std = errors[i][1]
            cool_errors = pltutil.get_errorbar_lengths(
                stacks[i][2], [errors[i][2][0], errors[i][2][1]]
            )

            # total mean as solid line
            common.plot_curve_with_error_region(
                xs,
                stacks[i][0],
                x_err=None,
                y_err=total_errors,
                axes=axes,
                linestyle="solid",
                color=color,
                label=label,
                suppress_error_line=True,
                suppress_error_region=True,
            )
            # plot cool gas mean only as dashed line
            common.plot_curve_with_error_region(
                xs,
                stacks[i][1],
                x_err=None,
                y_err=cool_std,
                axes=axes,
                linestyle="dashed",
                color=color,
                suppress_error_line=True,
                suppress_error_region=True,
            )
            # plot cool gas median only as dotted line
            common.plot_curve_with_error_region(
                xs,
                stacks[i][2],
                x_err=None,
                y_err=cool_errors,
                axes=axes,
                linestyle="dotted",
                color=color,
                suppress_error_line=True,
                suppress_error_region=True,
            )

        axes.legend(
            loc="upper right",
            # bbox_to_anchor=(0.5, 1.1),
            ncol=2,
            prop={"size": 8},
        )
        return fig, axes
