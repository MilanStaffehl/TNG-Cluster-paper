"""
Pipeline to plot radial temperature profiles for individual halos.
"""
# You wanna see programming gore? Well, you've come to the right place...
# I just want to make it known: I know this needs documentation, but we
# don't really have time to write it, so there you go: one Lovecraftian
# horror of a code for you to enjoy and decipher...
from __future__ import annotations

import logging
import sys
import time
import tracemalloc
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Literal, Sequence

import cmasher  # noqa: F401
import matplotlib.cm
import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.lines import Line2D
from numpy.typing import NDArray

from library.config import config
from library.data_acquisition import clusters_daq, gas_daq, halos_daq
from library.loading import load_radial_profiles
from library.plotting import colormaps, common
from library.processing import selection, statistics
from pipelines.base import DiagnosticsPipeline

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


@dataclass
class ClusterCoolGasMassTrendPipeline(DiagnosticsPipeline):
    """
    Pipeline to create plots of cool gas fraction vs halo mass.

    Pipeline creates plot of cool gas fraction vs. cluster mass for all
    clusters (i.e. halos with log M > 14.0) from TNG300-1 and TNG-Cluster.
    They can optionally be colored by a third halo quantity such as SFR
    or black hole mass.

    This pipeline must load all particle data in order to be able to
    plot gas particles that do ot belong to halos as well. There is
    another pipeline that instead infers the cool gas fraction from the
    radial density profiles of the clusters that is far less memory
    intensive when the data for these profiles already exists.
    """

    field: str
    color_scale: Literal["log", "linear"] | None
    deviation_scale: Literal["log", "linear"] | None
    gas_domain: Literal["halo", "central"]
    forbid_recalculation: bool
    force_recalculation: bool

    n_clusters: ClassVar[int] = 632
    n300: ClassVar[int] = 280  # number of clusters in TNG300-1
    nclstr: ClassVar[int] = 352  # number of clusters in TNG-Cluster

    def __post_init__(self):
        super().__post_init__()
        # file path for gas fraction data ("base data") and particle IDs
        if self.gas_domain == "halo":
            self.base_filename = "mass_trends_clusters_base_data.npz"
            id_subdir = "particle_ids"
        else:
            self.base_filename = "mass_trends_clusters_core_base_data.npz"
            id_subdir = "particle_ids_core"
        self.part_id_dir = self.config.data_home / id_subdir / "TNG300_1"

        # sim base paths
        self.tng300_basepath = config.get_simulation_base_path("TNG300-1")
        self.tngclstr_basepath = config.get_simulation_base_path("TNG-Cluster")

        # config file
        cur_dir = Path(__file__).parent
        with open(cur_dir / "plot_config.yaml", "r") as config_file:
            stream = config_file.read()
        self.plot_config: dict[str, Any] = yaml.full_load(stream)

        # set color scale if not explicitly stated as the field default
        if self.color_scale is None:
            self.color_scale = self.plot_config[self.field]["default-scale"]
            logging.debug(
                f"Set color scale to {self.color_scale} automatically."
            )
        if self.deviation_scale is None:
            scale = self.plot_config[self.field]["dev-config"]["default-scale"]
            self.deviation_scale = scale
            logging.debug(
                f"Set deviation plot color scale to {self.deviation_scale} "
                f"automatically."
            )

    def run(self) -> int:
        """
        Create plots of cool gas fraction vs. halo mass for all clusters.

        Steps:

        1. Get the halo masses and gas fractions either from file or,
           if not created yet, by following these following steps:
            1. Allocate memory space for the halo gas fraction, masses.
            2. Load halo data for TNG300-1, restrict to clusters.
            3. Get gas cell temperatures for TNG300-1.
            4. Get gas cell masses for TNG300-1.
            5. For every halo in TNG300-1:
               1. Select cells within 2R_vir using particle ID filed, mask
                  masses and temperatures.
               2. Sum their masses.
               3. Mask masses to cool gas only, sum cool gas mass.
               4. Calculate cool gas fraction.
               5. Clean-up.
            6. Load halo data for TNG-Cluster, restrict to zoom-ins.
            7. For every cluster in TNG-Cluster:
               1. Load gas cell data for the cluster (positions, mass, all
                  data required for temperature)
               2. Calculate distances, restrict temperature and masses to
                  only cells within 2R_vir
               3. Sum masses of gas cells.
               4. Mask masses to cool gas only, sum cool gas mass.
               5. Calculate cool gas fraction, place into allocated array.
               6. Clean-up.
        3. Acquire third quantity for all clusters.
        4. Plot scatter plot of the data; save it to file.

        :return: Exit code.
        """
        # Step 0: create directories, start memory monitoring, timing
        self._create_directories()
        if self.gas_domain == "central":
            logging.info("Plotting mass trends for core only.")

        # Step 1: acquire base halo data
        base_file = self.paths["data_dir"] / self.base_filename
        if base_file.exists() and not self.force_recalculation:
            logging.info("Found base data on file, loading it from there.")
            with np.load(base_file) as base_data:
                halo_masses = base_data["halo_masses"]
                cool_gas_fracs = base_data["cool_gas_fracs"]
        elif self.forbid_recalculation:
            halo_masses, cool_gas_fracs = self._load_base_data()
        else:
            halo_masses, cool_gas_fracs = self._get_base_data()

        # Step 2: acquire data to color the points with
        try:
            field_name = self.plot_config[self.field]["daq-field-name"]
        except KeyError:
            logging.fatal(
                f"Unsupported field name {self.field}. Config file contains "
                f"no such field or has no field name for the data acquisition "
                f"function available."
            )
            raise
        try:
            color_quantity = clusters_daq.get_cluster_property(
                field_name, self.config.snap_num, self.config.mass_field
            )
        except clusters_daq.UnsupportedFieldError as exc:
            logging.error(
                f"Unknown or unsupported field name {self.field}. Tried to "
                f"load field with name {field_name}, but encountered an "
                f"exception:\n\n{exc}"
            )
            raise
        else:
            # since there is color data, save it to file
            if self.to_file:
                logging.info(f"Writing color data {self.field} to file.")
                filename = f"{self.paths['data_file_stem']}.npy"
                filepath = self.paths["data_dir"] / "color_data"
                np.save(filepath / filename, color_quantity)

        # Step 3: plot data, save figure
        kwargs = self._get_plot_kwargs()
        if self.color_scale == "log":
            plot_color = self._nanlog10(color_quantity)
        else:
            plot_color = np.copy(color_quantity)
        f, _ = self._plot(halo_masses, cool_gas_fracs, plot_color, kwargs)
        self._save_fig(f)
        logging.info(
            f"Successfully plotted mass trend colored by {self.field}."
        )

        # Step 4: plot not the quantity, but its trend at fixed mass
        deviation_color = statistics.find_deviation_from_median_per_bin(
            color_quantity,
            np.log10(halo_masses),
            min_mass=14.0,
            max_mass=15.4,
            num_bins=7,
        )
        if self.deviation_scale == "log":
            # avoid NaNs (okay, since this array is used only for plotting)
            deviation_color = self._nanlog10(deviation_color, "deviation")
        dev_kwargs = self._get_plot_kwargs_for_median_diff()
        f, a = self._plot(
            halo_masses,
            cool_gas_fracs,
            deviation_color,
            dev_kwargs,
        )

        # Step 5: overplot statistical quantities
        corcoef = statistics.pearson_corrcoeff_per_bin(
            cool_gas_fracs,
            color_quantity,
            np.log10(halo_masses),
            min_mass=14.0,
            max_mass=15.4,
            num_bins=7,
        )
        deltas = statistics.two_side_difference(
            cool_gas_fracs,
            color_quantity,
            np.log10(halo_masses),
            min_mass=14.0,
            max_mass=15.4,
            num_bins=7,
        )
        # overplot Pearson correlation coefficients as text
        self._add_statistics_labels(a, corcoef, deltas)

        # Step 6: save median deviation figure
        self._save_fig(f, ident_flag="median_dev")
        logging.info(
            f"Successfully plotted mass trend median deviation of field "
            f"{self.field}."
        )

        return 0

    def _get_base_data(self) -> tuple[NDArray, NDArray]:
        """
        Generate the base data for the plot.

        Function requires the particle IDs of particles associated with
        all clusters in TNG300-1 to already exist. These can be generated
        with the :mod:`~pipelines.radial_profiles.individuals` pipelines.

        :return: The tuple of arrays of length 632 of the halo masses
            and their corresponding cool gas fraction.
        """
        logging.info("Loading base data directly from simulation data.")
        tracemalloc.start()
        begin = time.time()

        # Step 1: allocate memory for plot data
        halo_masses = np.zeros(self.n_clusters)
        cool_gas_fracs = np.zeros(self.n_clusters)

        # Step 2: load halo data for TNG300-1, restrict to clusters
        logging.info("Loading halo data for TNG300-1.")
        fields = [self.config.mass_field, self.config.radius_field]
        halo_data = halos_daq.get_halo_properties(
            self.tng300_basepath,
            self.config.snap_num,
            fields=fields,
        )
        cluster_data = selection.select_clusters(
            halo_data, self.config.mass_field, expected_number=self.n300
        )
        # assign halo masses and third quantity
        halo_masses[:self.n300] = cluster_data[self.config.mass_field]
        # clean-up
        del halo_data
        timepoint = self._diagnostics(
            begin, "loading TNG300 halo data", unit="MB"
        )

        # Step 3: Get gas cell temperatures for TNG300-1
        temperatures = gas_daq.get_gas_temperatures(
            self.tng300_basepath, self.config.snap_num
        )
        timepoint = self._diagnostics(
            timepoint, "calculating gas cell temperatures"
        )

        # Step 4: Get gas cell masses for TNG300-1
        masses = gas_daq.get_gas_properties(
            self.tng300_basepath,
            self.config.snap_num,
            ["Masses"],
        )["Masses"]
        timepoint = self._diagnostics(timepoint, "loading gas cell masses")

        # Step 5: Loop through clusters in TNG300-1
        logging.info("Begin processing TNG300-1 halos to get gas fraction.")
        log_level = logging.getLogger("root").level
        core = "_core" if (self.gas_domain == "central") else ""
        for i, halo_id in enumerate(cluster_data["IDs"]):
            if log_level <= 15:
                print(f"Processing halo {halo_id} ({i + 1} / 280).", end="\r")
            # Step 5.1: Select cells within 2R_vir, mask masses & temps
            neighbors = np.load(
                self.part_id_dir / f"particles_halo_{halo_id}{core}.npy"
            )
            cur_masses = masses[neighbors]
            cur_temps = temperatures[neighbors]

            # Step 5.2: Sum masses to get total gas mass
            total_mass = np.sum(cur_masses)

            # Step 5.3: Mask masses to cool gas only, sum cool gas mass
            mask = np.digitize(cur_temps, [0, 10**4.5, np.inf])
            cool_gas_masses = selection.mask_quantity(
                cur_masses, mask, index=1, compress=True
            )
            cool_mass = np.sum(cool_gas_masses)

            # Step 5.4: Calculate and save cool gas fraction
            cool_gas_fracs[i] = cool_mass / total_mass

            # Step 5.5: Clean-up
            del neighbors, cur_masses, cur_temps, mask, cool_gas_masses
        timepoint = self._diagnostics(
            timepoint, "processing all TNG300 clusters"
        )

        # Step 6: Load halo data from TNG-Cluster
        logging.info("Loading halo data for TNG-Cluster.")
        fields.append("GroupPos")
        cluster_data = halos_daq.get_halo_properties(
            self.tngclstr_basepath,
            self.config.snap_num,
            fields=fields,
            cluster_restrict=True,
        )
        halo_masses[self.n300:] = cluster_data[self.config.mass_field]
        timepoint = self._diagnostics(
            timepoint, "loading TNG-Cluster halo data"
        )

        # Step 7: Loop through clusters
        logging.info("Begin processing TNG-Cluster halos to get gas fraction.")
        for i, halo_id in enumerate(cluster_data["IDs"]):
            if log_level <= 15:
                print(f"Processing halo {halo_id} ({i + 1}/352).", end="\r")
            # Step 7.1: Load gas cell data for temperature
            cluster_temperatures = gas_daq.get_cluster_temperature(
                halo_id,
                self.tngclstr_basepath,
                self.config.snap_num,
            )

            # Step 7.2: Load gas cell positions, calculate distance
            gas_data = gas_daq.get_gas_properties(
                self.tngclstr_basepath,
                self.config.snap_num,
                fields=["Coordinates", "Masses"],
                cluster=halo_id,
            )
            gas_distances = np.linalg.norm(
                gas_data["Coordinates"] - cluster_data["GroupPos"][i], axis=1
            ) / cluster_data[self.config.radius_field][i]

            # Step 7.3: Restrict masses and temperatures
            limit = 0.05 if (self.gas_domain == "central") else 2.0
            cur_masses = gas_data["Masses"][gas_distances <= limit]
            cur_temps = cluster_temperatures[gas_distances <= limit]
            del cluster_temperatures, gas_distances  # clean-up

            # Step 7.4: Sum mass to get total mass
            total_mass = np.sum(cur_masses)

            # Step 7.5: Mask masses to only cool gas
            mask = np.digitize(cur_temps, [0, 10**4.5, np.inf])
            cool_gas_masses = selection.mask_quantity(
                cur_masses, mask, index=1, compress=True
            )
            cool_mass = np.sum(cool_gas_masses)

            # Step 7.6: Calculate gas fraction; place in array
            cool_gas_fracs[self.n300 + i] = cool_mass / total_mass

            # Step 7.7: Clean-up
            del gas_data, cool_gas_masses

        self._diagnostics(timepoint, "processing all TNG-Cluster halos")

        # Step 8: Save data to file and plot it
        if self.to_file:
            np.savez(
                self.paths["data_dir"] / self.base_filename,
                halo_masses=halo_masses,
                cool_gas_fracs=cool_gas_fracs,
            )
        self._timeit(begin, "entire base data acquisition")
        return halo_masses, cool_gas_fracs

    def _load_base_data(self):
        """
        Load the halo masses and cool gas fraction from the density profiles.

        .. attention:: The order of the halos is not preserved with this
            method. Halos may appear in the array in arbitrary order, not
            in the order of their IDs. They are however still split by
            simulation (i.e. first 280 halos are guaranteed to be from
            TNG300, the last 352 are from TNG-Cluster)

        :return: The tuple of arrays of length 632 of the halo masses
            and their corresponding cool gas fraction.
        """
        logging.info("Loading base data via radial density profiles.")
        halo_masses = np.zeros(self.n_clusters)
        cool_gas_fracs = np.zeros(self.n_clusters)

        data_path = self.config.data_home / "radial_profiles" / "individuals"
        subdir_name = "density_profiles"
        if self.gas_domain == "central":
            subdir_name += "_core"
            limit = 0.05
        else:
            limit = 2.0

        edges = np.linspace(0, limit, num=51, endpoint=True)
        shell_volumes = 4 / 3 * np.pi * (edges[1:]**3 - edges[:-1]**3)

        # load data for TNG300-1
        logging.info("Loading data for TNG300-1.")
        halo_loader = load_radial_profiles.load_individuals_1d_profile(
            data_path / "TNG300_1" / subdir_name, (50, )
        )
        idx = 0
        for halo_data in halo_loader:
            halo_masses[idx] = halo_data["halo_mass"]
            cool_profile = halo_data["cool_inflow"] + halo_data["cool_outflow"]
            cool_gas = np.sum(cool_profile * shell_volumes)
            total_profile = halo_data["total_inflow"] + halo_data[
                "total_outflow"]
            total_gas = np.sum(total_profile * shell_volumes)
            cool_gas_fracs[idx] = cool_gas / total_gas
            idx += 1

        # load data for TNG-Cluster
        logging.info("Loading data for TNG-Cluster.")
        halo_loader = load_radial_profiles.load_individuals_1d_profile(
            data_path / "TNG_Cluster" / subdir_name, (50, )
        )
        for halo_data in halo_loader:
            halo_masses[idx] = halo_data["halo_mass"]
            cool_profile = halo_data["cool_inflow"] + halo_data["cool_outflow"]
            cool_gas = np.sum(cool_profile * shell_volumes)
            total_profile = halo_data["total_inflow"] + halo_data[
                "total_outflow"]
            total_gas = np.sum(total_profile * shell_volumes)
            cool_gas_fracs[idx] = cool_gas / total_gas
            idx += 1

        if self.to_file:
            np.savez(
                self.paths["data_dir"] / self.base_filename,
                halo_masses=halo_masses,
                cool_gas_fracs=cool_gas_fracs,
            )
            logging.info("Wrote base data to file.")

        return halo_masses, cool_gas_fracs

    def _plot(
        self,
        halo_masses: NDArray,
        gas_fraction: NDArray,
        colored_quantity: NDArray | None,
        additional_kwargs: dict[str, Any],
    ) -> tuple[Figure, Axes]:
        """
        Plot the cool gas fraction vs. halo mass plot.

        :param halo_masses: Array of shape (632, ) containing the halo
            masses in units of M_sol.
        :param gas_fraction: Array of shape (632, ) containing the gas
            fraction of cool gas in halos.
        :param colored_quantity: Array of shape (632, ) containing the
            quantity that shall be used for coloring the data points.
            Can be in arbitrary units. If no color is to be used, set
            this to None. There will be nor transformation applied to
            this data, i.e. if you wish to plot it in log scale, you
            must pass it already in log scale.
        :param additional_kwargs: A dictionary containing additional
            keywords for the :func:`plot_scatterplot` function.
        :return: Figure and axes objects as tuple.
        """
        logging.info("Plotting cool gas fraction mass trend for clusters.")
        fig, axes = plt.subplots(figsize=(5, 4))
        axes.set_xlabel(r"Halo mass $M_{200c}$ [$\log M_\odot$]")
        axes.set_xticks(np.linspace(14.0, 15.4, num=8))
        if self.gas_domain == "central":
            axes.set_ylabel(r"Cool gas fraction within $0.05R_{200c}$")
        else:
            axes.set_ylabel(r"Cool gas fraction within $2R_{200c}$")
        axes.set_yscale("log")

        logging.debug(f"Smallest gas frac value: {np.min(gas_fraction):e}")
        # make zero-values visible; scatter them a little
        rng = np.random.default_rng(42)
        n_zeros = len(gas_fraction) - np.count_nonzero(gas_fraction)
        logging.debug(f"Number of zero entries: {n_zeros}")
        randnums = np.power(5, rng.random(n_zeros))
        gas_fraction[gas_fraction == 0] = 1e-7 * randnums

        logging.info(f"Coloring scatter points by {self.field}.")
        # determine min and max of colorbar values
        cbar_min, cbar_max = None, None
        if "cbar_range" in additional_kwargs.keys():
            cbar_min, cbar_max = additional_kwargs.pop("cbar_range")
        if cbar_min is None:
            cbar_min = np.nanmin(colored_quantity)
        if cbar_max is None:
            cbar_max = np.nanmax(colored_quantity)
        # extract color label or set a default
        try:
            label = additional_kwargs.pop("cbar_label")
        except KeyError:
            logging.warning("Received no color bar label!")
            label = self.field
        # plot TNG300 data
        common.plot_scatterplot(
            fig,
            axes,
            np.log10(halo_masses[:self.n300]),
            gas_fraction[:self.n300],
            colored_quantity[:self.n300],
            legend_label="TNG300-1",
            marker_style="o",
            cbar_label=label,
            cbar_range=[cbar_min, cbar_max],
            **additional_kwargs,
        )
        # plot TNG-Cluster data
        common.plot_scatterplot(
            fig,
            axes,
            np.log10(halo_masses[self.n300:]),
            gas_fraction[self.n300:],
            colored_quantity[self.n300:],
            legend_label="TNG-Cluster",
            marker_style="D",
            suppress_colorbar=True,
            cbar_range=[cbar_min, cbar_max],
            **additional_kwargs
        )
        # set handles manually to avoid coloring them
        tng300_handle = Line2D(
            [],
            [],
            color="black",
            marker="o",
            ls="",
            markersize=3,
            label="TNG300-1",
        )
        tngclstr_handle = Line2D(
            [],
            [],
            color="black",
            marker="D",
            ls="",
            markersize=3,
            label="TNG-Cluster",
        )
        axes.legend(handles=[tng300_handle, tngclstr_handle])
        return fig, axes

    @staticmethod
    def _add_statistics_labels(
        axes: Axes, pearson_cc: NDArray, deltas: NDArray
    ) -> Axes:
        """
        Add labels with the statistical quantities to the plot.

        :param axes: The axes onto which to draw the labels.
        :param pearson_cc: The array of Pearson correlation coefficients.
        :param deltas: Difference between mean color above and below
            median gas fraction per mass bin.
        :return: The Axes object, for convenience. Axes is altered in
            place.
        """
        if len(pearson_cc) != len(deltas):
            logging.error(
                "Pearson coefficients and Deltas have different lengths, "
                "cannot add them to the plot."
            )
            return axes

        n_bins = len(pearson_cc)
        bin_width = 0.9 / n_bins
        for i in range(n_bins):
            axes.text(
                0.05 + i * bin_width + bin_width / 2,
                0.05 + (i % 2) * 0.1,
                f"R = {pearson_cc[i]:.2f}\n"
                rf"$\Delta$ = {deltas[i]:.2f}",
                fontsize=8,
                color="black",
                backgroundcolor=(1, 1, 1, 0.6),
                transform=axes.transAxes,
                horizontalalignment="center",
                verticalalignment="bottom",
                zorder=10,
            )
        return axes

    def _get_plot_kwargs(self) -> dict[str, Any]:
        """
        Return keyword parameters for the plotting function.

        For the given field, return a dictionary of keyword arguments
        for the plotting function.

        :return: Dict of keyword argument.
        """
        cur_cfg = self.plot_config[self.field]
        norm_type = cur_cfg["cbar-config"]["norm"]
        if norm_type == "default":
            kwargs = self._get_default_norm_plot_kwargs()
        elif norm_type == "twoslope":
            kwargs = self._get_twoslope_norm_plot_kwargs()
        else:
            raise NotImplementedError(f"Norm not implemented: {norm_type}")
        kwargs.update({"cbar_label": rf"{cur_cfg['label'][self.color_scale]}"})
        return kwargs

    def _get_default_norm_plot_kwargs(self):
        """
        Return the plot kwargs for a normal colorbar norm.

        :return: Dict of keyword arguments for plotting.
        """
        cbar_cfg = self.plot_config[self.field]["cbar-config"]
        kwargs = {"cmap": cbar_cfg["cmap"]}
        # Add cbar limits if they are specified
        limits = [None, None]
        if "vmin" in cbar_cfg.keys():
            if self.color_scale in cbar_cfg["vmin"].keys():
                limits[0] = float(cbar_cfg["vmin"][self.color_scale])
        if "vmax" in cbar_cfg.keys():
            if self.color_scale in cbar_cfg["vmax"].keys():
                limits[1] = float(cbar_cfg["vmax"][self.color_scale])
        kwargs.update({"cbar_range": limits})
        # add config for open cbar caps if specified
        if "caps" in cbar_cfg.keys():
            kwargs.update({"cbar_caps": cbar_cfg["caps"]})
        return kwargs

    def _get_twoslope_norm_plot_kwargs(self):
        """
        Return the plot kwargs for a two-slope colorbar norm.

        :return: Dict of keyword arguments for plotting.
        """
        cbar_cfg = self.plot_config[self.field]["cbar-config"]
        # extract colormaps
        if not isinstance(cbar_cfg["cmap"], Sequence):
            logging.error(
                "Two-slope norms require multiple colormaps, but only one "
                "was given. Using fallback cmaps."
            )
            lower_cmap = "winter"
            upper_cmap = "autumn"
        else:
            lower_cmap = cbar_cfg["cmap"][0]
            upper_cmap = cbar_cfg["cmap"][1]

        # extract limits
        try:
            # limits differ by scale
            vmin = float(cbar_cfg["vmin"][self.color_scale])
            vmax = float(cbar_cfg["vmax"][self.color_scale])
            vcenter = float(cbar_cfg["vcenter"][self.color_scale])
        except KeyError as exc:
            logging.fatal(
                f"At least one norm parameter vmin, vmax, vcenter was not "
                f"specified, but required:\n {exc}"
            )
            raise

        cmap, norm = colormaps.two_slope_cmap(
            lower_cmap, upper_cmap, vmin=vmin, vmax=vmax, vcenter=vcenter,
        )
        kwargs = {"cmap": cmap, "norm": norm}
        if "caps" in cbar_cfg.keys():
            kwargs.update({"cbar_caps": cbar_cfg["caps"]})

        return kwargs

    def _get_plot_kwargs_for_median_diff(self) -> dict[str, Any]:
        """
        Return keyword parameters for plotting median deviation.

        For the given field, return a dictionary of keyword arguments
        for the plotting function. This method returns the kwargs for
        plotting the deviation of a quantity from a mass bin median.

        :param field: Name of the field.
        :return: Dict of keyword argument.
        """
        dev_cfg = self.plot_config[self.field]["dev-config"]
        descr = self.plot_config[self.field]["label"]["dev"]
        if self.deviation_scale == "log":
            label = (
                rf"Deviation from median [$\log_{{10}}({descr}/"
                rf"\tilde{{{descr}}})$]"
            )
            limits = dev_cfg["limits-log"]
            norm_config = {"vmin": limits[0], "vcenter": 0, "vmax": limits[1]}
        else:
            label = rf"Deviation from median [${descr}/\tilde{{{descr}}}$]"
            limits = dev_cfg["limits-linear"]
            norm_config = {"vmin": limits[0], "vcenter": 1, "vmax": limits[1]}

        # create a custom colormap
        full_range_channel = np.linspace(0, 1, 256)
        full_cmap = np.zeros((512, 3))
        full_cmap[:256, 2] = np.flip(full_range_channel)  # blue channel
        full_cmap[256:, 0] = full_range_channel  # red channel
        deviation_map = matplotlib.colors.LinearSegmentedColormap.from_list(
            "deviation_map", full_cmap
        )
        norm = matplotlib.colors.TwoSlopeNorm(**norm_config)

        config_dict = {
            "cbar_label": label,
            "norm": norm,
            "cmap": deviation_map,
            "cbar_caps": "both",
        }
        return config_dict

    @staticmethod
    def _nanlog10(x: NDArray, what: str = "color"):
        """
        Logarithm avoiding NaN values by setting zeros to small value.

        Method replaces all zero entries in ``x`` with 0.1 times the
        smallest non-zero value of ``x``, then returns the log10 of the
        resulting array. This avoids NaNs, but makes the result inaccurate.
        Useful for plotting, when zero-values may be treated as small
        non-zero values. Avoid for real physics.

        :param x: The quantity to log.
        :param what: Description of the quantity being logged, for the
            debug log message emitted. Defaults to 'color'.
        :return: The log10 of the array, with all zero-entries replaced
            by a small value before taking the log.
        """
        quantity = np.copy(x)
        # set zeros to a small value to avoid NaNs
        min_val = np.nanmin(quantity[quantity != 0])
        logging.debug(f"Smallest non-zero {what} value: {min_val}")
        quantity[quantity == 0] = 0.1 * min_val
        # log the values
        return np.log10(quantity)


class ClusterCoolGasFromFilePipeline(ClusterCoolGasMassTrendPipeline):
    """
    Pipeline to plot the gas fraction vs. mass trend from file.

    Plots a single quantity as a single panel.
    """

    def __post_init__(self):
        super().__post_init__()

    def run(self) -> int:
        """Load data from file and plot it."""
        self._verify_directories()
        if self.field is None:
            logging.fatal("Require color field to plot.")
            sys.exit(10)

        # load base data
        logging.info("Loading base data from file.")
        base_file = self.paths["data_dir"] / self.base_filename
        if not base_file.exists():
            logging.fatal(f"Base data file {base_file} does not exist!")
            return 1

        with np.load(base_file) as base_data:
            halo_masses = base_data["halo_masses"]
            cool_gas_fracs = base_data["cool_gas_fracs"]

        # make zero-values visible; scatter them a little
        rng = np.random.default_rng(42)
        n_zeros = len(cool_gas_fracs) - np.count_nonzero(cool_gas_fracs)
        randnums = np.power(5, rng.random(n_zeros))
        cool_gas_fracs[cool_gas_fracs == 0] = 1e-7 * randnums

        # load color data
        filename = f"{self.paths['data_file_stem']}.npy"
        filepath = self.paths["data_dir"] / "color_data"
        color_data = np.load(filepath / filename)
        color_kwargs = self._get_plot_kwargs()

        # plot data
        if self.color_scale == "log":
            # log the values, avoiding NaNs
            plot_color = self._nanlog10(color_data)
        else:
            plot_color = np.copy(color_data)
        f, _ = self._plot(halo_masses, cool_gas_fracs, plot_color, color_kwargs)
        self._save_fig(f)
        logging.info(
            f"Successfully plotted mass trend colored by {self.field}."
        )

        # plot deviation
        deviation_color = statistics.find_deviation_from_median_per_bin(
            color_data,
            np.log10(halo_masses),
            min_mass=14.0,
            max_mass=15.4,
            num_bins=7,
        )
        # get statistical quantities
        corcoef = statistics.pearson_corrcoeff_per_bin(
            cool_gas_fracs,
            color_data,
            np.log10(halo_masses),
            min_mass=14.0,
            max_mass=15.4,
            num_bins=7,
        )
        deltas = statistics.two_side_difference(
            cool_gas_fracs,
            color_data,
            np.log10(halo_masses),
            min_mass=14.0,
            max_mass=15.4,
            num_bins=7,
        )
        if self.deviation_scale == "log":
            deviation_color = self._nanlog10(deviation_color, "deviation")
        dev_kwargs = self._get_plot_kwargs_for_median_diff()
        f, a = self._plot(
            halo_masses,
            cool_gas_fracs,
            deviation_color,
            dev_kwargs,
        )
        self._add_statistics_labels(a, corcoef, deltas)
        self._save_fig(f, ident_flag="median_dev")
        logging.info(
            f"Successfully plotted mass trend median deviation of field "
            f"{self.field}."
        )
