"""
Pipeline for stacking radial profiles.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Literal

import matplotlib.gridspec
import matplotlib.pyplot as plt
import numpy as np

from library.loading import load_radial_profiles
from library.plotting import common, plot_radial_profiles, pltutil
from library.processing import statistics
from pipelines.base import Pipeline

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from numpy.typing import NDArray


@dataclass
class StackProfilesPipeline(Pipeline):
    """
    Pipeline to create stacks of radial profiles.
    """

    log: bool
    what: str
    method: Literal["mean", "median"]

    divisions: ClassVar[NDArray] = np.array([4.5, 5.5])  # temperature divs

    def run(self) -> int:
        """
        Load the histograms saved to file and stack them, create plots.

        A lot of functionality in this pipeline is hard-coded, which makes it
        error-prone. Use with caution.

        :return: Exit code.
        """
        # Step 0: verify directories and input
        self._create_directories()
        if self.what not in ["temperature", "density"]:
            logging.error(f"Unknown profile type: {self.what}. Aborting.")
            return 1

        # Step 1: find out shape and allocate memory
        files = [f for f in self.paths["data_dir"].iterdir() if f.is_file()]
        n_halos = len(files)
        # test-load first halo
        with np.load(files[0].resolve()) as data_file:
            if self.what == "temperature":
                hist_shape = data_file["original_historgram"].shape
                desired_shape = hist_shape
                edges = [
                    data_file["xedges"][0],
                    data_file["xedges"][-1],
                    data_file["yedges"][0],
                    data_file["yedges"][-1],
                ]
            else:
                # allocate space for total, warm and cool hist
                hist_shape = data_file["total_histogram"].shape
                desired_shape = (3, ) + hist_shape
                edges = data_file["edges"]

        # allocate memory
        histograms = np.zeros((n_halos, ) + desired_shape)

        # Step 2: load the data
        if self.what == "temperature":
            loader = load_radial_profiles.load_individuals_2d_profile
        else:
            loader = load_radial_profiles.load_individuals_1d_profile
        # iterate through directory
        for i, halo_data in enumerate(loader(self.paths["data_dir"], hist_shape)):
            if self.what == "temperature":
                histograms[i] = halo_data["original_histogram"]
            elif self.what == "density":
                histograms[i][0] = halo_data["total_histogram"]
                histograms[i][1] = halo_data["warm_histogram"]
                histograms[i][2] = halo_data["cool_histogram"]

        # Step 3: stack the histograms
        stack = statistics.stack_histograms(histograms, method=self.method)
        if self.what == "temperature":
            # 2D histograms need to be normalized column-wise
            normalized_histograms, _, _ = statistics.column_normalized_hist2d(
                stack[0], None, None, normalization="density"
            )
            # re-assign to variable
            stack = (
                normalized_histograms,
                stack[1].transpose(),
                stack[2].transpose()
            )

        # Step 4: get running average
        if self.what == "temperature":
            running_avg = statistics.get_2d_histogram_running_average(
                stack[0], edges[-2:]
            )

        # Step 5: plot the results
        if self.what == "temperature" and self.method == "mean":
            f, a = self._plot_2d_mean(stack, edges, running_avg)
        elif self.what == "temperature" and self.method == "median":
            f, a = self._plot_2d_median(stack, edges, running_avg)
        elif self.what == "density":
            f, a = self._plot_1d(stack, self.method)
        else:
            logging.error(
                f"Unrecognized combination of method {self.method} and plot "
                f"type {self.what}."
            )
            return 2

        # save plot to file
        if self.no_plots:
            return 0
        name = f"{self.paths['figures_file_stem']}_{self.method}.pdf"
        path = Path(self.paths["figures_dir"])
        if not path.exists():
            logging.info("Creating missing figures directory for stacks.")
            path.mkdir(parents=True)
        f.savefig(path / name, bbox_inches="tight")
        plt.close(f)

    def _plot_2d_mean(
        self,
        stack_data: tuple[NDArray, NDArray, NDArray],
        edges: NDArray,
        running_average: NDArray,
    ) -> tuple[Figure, tuple[Axes, Axes]]:
        """
        Plot the mean histogram plus the standard deviation.

        :param stack_data: The tuple of the mean histogram, and the
            standard deviation.
        :param edges: The edges of the x- and y-axes: [xmin, xmax, ymin, ymax].
        :param running_average: The array of temperature averages in
            every radial column.
        :return: Tuple of figure and axes objects with the plots.
        """
        # create and configure figure and axes
        fig, axes = plt.subplots(ncols=2, figsize=(9, 4), sharey=True)
        fig.set_tight_layout(True)

        # plot the mean/median
        if self.log:
            clabel = r"Normalized mean gas fraction ($\log_{10}$)"
        else:
            clabel = "Normalized mean gas fraction"
        with np.errstate(invalid="ignore", divide="ignore"):
            plot_radial_profiles.plot_2d_radial_profile(
                fig,
                axes[0],
                stack_data[0],
                edges,
                title="Mean temperature profile",
                cbar_label=clabel,
                cbar_limits=[-4, None] if self.log else None,
                scale="log" if self.log else "linear",
                log_msg="all halos above 10^14 M_sol",
            )
            # plot the error
            plot_radial_profiles.plot_2d_radial_profile(
                fig,
                axes[1],
                stack_data[1],
                edges,
                title="Standard deviation",
                colormap="gist_rainbow",
                cbar_label="Standard deviation of gas fraction",
                scale="log" if self.log else "linear",
            )
        # plot the running average
        plot_radial_profiles.overplot_running_average(
            fig,
            axes[0],
            running_average,
            edges,
        )
        # plot temperature division
        if self.log:
            divs = self.divisions
        else:
            divs = 10**self.divisions
        plot_radial_profiles.overplot_temperature_divisions(axes, divs, 0, 2)
        return fig, axes

    def _plot_2d_median(
        self,
        stack_data: tuple[NDArray, NDArray, NDArray],
        edges: NDArray,
        running_average: NDArray,
    ) -> tuple[Figure, tuple[Axes, Axes, Axes]]:
        """
        Plot the median histogram plus the 16th and 84th percentiles.

        :param stack_data: The tuple of the median histogram, and the
            16th and 84th percentiles.
        :param edges: The edges of the x- and y-axes: [xmin, xmax, ymin, ymax].
        :param running_average: The array of average temperature in
            every radial column.
        :return: Tuple of figure and axes objects with the plots.
        """
        fig = plt.figure(layout="constrained", figsize=(7.5, 4))
        gs = matplotlib.gridspec.GridSpec(
            2, 2, figure=fig, width_ratios=[2 / 3, 1 / 3]
        )
        ax1 = fig.add_subplot(gs[:, 0])  # axes for median
        ax2 = fig.add_subplot(gs[0, 1])  # axes for lower percentile
        ax3 = fig.add_subplot(
            gs[1, 1], sharex=ax2
        )  # axes for upper percentile

        # plot the data
        if self.log:
            clabel = r"Normalized median gas fraction ($\log_{10}$)"
        else:
            clabel = "Normalized median gas fraction"
        with np.errstate(invalid="ignore", divide="ignore"):
            plot_radial_profiles.plot_2d_radial_profile(
                fig,
                ax1,
                stack_data[0],
                edges,
                title="Median temperature profile",
                cbar_label=clabel,
                cbar_limits=[-4, None] if self.log else None,
                scale="log" if self.log else "linear",
                log_msg="all halos above 10^14 M_sol",
            )
            plot_radial_profiles.plot_2d_radial_profile(
                fig,
                ax2,
                stack_data[0] - stack_data[1],
                edges,
                xlabel=None,
                cbar_label="Lower error",
                colormap="gist_rainbow",
                labelsize=10,
            )
            plot_radial_profiles.plot_2d_radial_profile(
                fig,
                ax3,
                stack_data[0] - stack_data[2],
                edges,
                cbar_label="Upper error",
                colormap="gist_rainbow",
                labelsize=10,
            )
        # overplot running average
        plot_radial_profiles.overplot_running_average(
            fig, ax1, running_average, edges
        )
        return fig, (ax1, ax2, ax3)

    def _plot_1d(
        self,
        stack_data: tuple[NDArray, NDArray, NDArray],
        type_: Literal["mean", "median"],
    ) -> tuple[Figure, Axes]:
        """
        Plot the mean density histogram.

        :param stack_data: The tuple of the mean histogram, and the
            standard deviation. The three arrays are each of shape
            (3, X) where the first axis splits the histograms into
            total (0), warm (1) and hot (2) histograms, each of shape
            (X, ).
        :param type_: The type of plot, can either be 'mean' or 'median'.
        :return: Tuple of figure and axes objects with the plots.
        """
        logging.info(f"Plotting {type_} density profile.")
        fig, axes = plt.subplots(figsize=(5, 4))
        axes.set_title(f"{type_.capitalize()} density")
        axes.set_xlabel(r"Distance from halo center [$R_{200c}$]")
        axes.set_xlim([0, 2])
        axes.set_ylabel(
            fr"{type_.capitalize()} density in radial shell "
            r"[$M_\odot / kpc^3$]"
        )
        if self.log:
            axes.set_yscale("log")

        # generate x-values in the middle of the radial bins
        xs = np.linspace(0, 2, len(stack_data[0][0]), endpoint=False)
        xs += (1 / len(stack_data[0][0]))

        for i, label in enumerate(["total", "warm", "cool"]):
            # generate an array of errors
            if type_ == "mean":
                errors = np.array([stack_data[1][i], stack_data[2][i]])
            elif type_ == "median":
                errors = pltutil.get_errorbar_lengths(
                    stack_data[0][i], [stack_data[1][i], stack_data[2][i]]
                )

            if label == "total":
                color = "black"
            else:
                color = common.temperature_colors_named[label]

            # plot
            common.plot_curve_with_error_region(
                xs,
                stack_data[0][i],
                None,
                errors,
                axes,
                label=label,
                color=color,
            )
        return fig, axes
