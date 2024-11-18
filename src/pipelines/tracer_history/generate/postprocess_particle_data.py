"""
Pipelines to process existing particle data into derived quantities.
"""
from __future__ import annotations

import dataclasses
import logging
from typing import TYPE_CHECKING, ClassVar

import h5py
import numpy as np

from library import constants
from library.data_acquisition import halos_daq, sublink_daq
from pipelines import base

if TYPE_CHECKING:
    from numpy.typing import NDArray


class CrossingUtilMixin:
    """
    Mixin class, providing common methods to find crossing times.
    """

    @staticmethod
    def _first_and_last_zero_crossing(
        differences: NDArray,
        treat_zeroes: bool = False,
    ) -> tuple[NDArray, NDArray]:
        """
        Given an array of numbers, return first and last zero-crossing.

        Method takes an array of shape (S, N) and finds, along the first
        axis, the first and last crossing over zero .This is used to
        find the first and last crossing of particles over the virial
        radius of a cluster: given an array of the difference between
        the distance of the particle and the virial radius of shape
        (S, N), where S is the number of snapshots and N is the number
        of particles, the function finds all crossing from positive to
        negative (i.e. all moments where a particle crossed the virial
        radius from the outside to the inside) and returns an array of
        shape (N, ) containing the index of the first time this happens
        and one of shape (N, ) of the last time where it happens (which
        may be the same as the first crossing).

        .. warning:: If the differences are ever exactly 0, the corresponding
            index is not returned! A warning will be logged then.

        .. note:: The indices are such that they point to the last positive
            entry. The next entry will be negative.

        It can theoretically happen, that the difference is exactly zero
        in some cases. This is not easily taken into account by the
        algorithm. If this is however something that needs to be taken
        into account, set ``treat_zeroes`` to True, so that the algorithm
        will also treat differences from positive values to exactly zero
        and subsequently negative values as a crossing. If set to False,
        these pseudo-crossings are ignored and only a warning is logged
        when there are zero-entries in ``differences``.

        :param differences: An array of shape (S, N) containing the
            difference ``part_distance - virial_radius``.
        :param treat_zeroes: #whether to treat values exactly zero in
            ``differences`` as a crossing, provided the value before is
            positive and the value after is negative.
        :return: The array of indices of the first change from positive
            to negative in ``differences`` along the S-axis of shape (N, )
            and one array of the indices of the last change from positive
            to negative. Note that the index points to the last entry of
            ``differences`` that is positive before it changes sign, i.e.
            it points at the entry where the particle is still outside
            the threshold.
        """
        contains_zeros = (np.count_nonzero(differences) != differences.size)
        if not treat_zeroes:
            if contains_zeros:
                logging.warning(
                    "Encountered difference with values exactly zero! This "
                    "means some crossing indices will not be correct!"
                )

        # find all array positions where the sign changes towards negative
        crossings_indices = (np.diff(np.sign(differences), axis=0) == -2)
        first_crossing_idx = np.argmax(crossings_indices, axis=0)
        index_from_back = np.argmax(np.flip(crossings_indices, axis=0), axis=0)
        last_crossing_idx = crossings_indices.shape[0] - index_from_back - 1

        # take into account particles that never cross
        no_changes_mask = ~np.any(crossings_indices, axis=0)
        first_crossing_idx[no_changes_mask] = -1
        last_crossing_idx[no_changes_mask] = -1

        # find exact zero-crossings, check if they qualify for first or last
        if treat_zeroes and contains_zeros:
            pseudo_crossings = (np.diff(np.sign(differences), axis=0) == -1)
            # check whether this is a crossing to zero, rather than
            # a crossing FROM zero to a negative number:
            for snap_num, pidx in zip(*np.nonzero(pseudo_crossings)):
                # check value at next snapshot
                if differences[snap_num + 1, pidx] != 0:
                    # crossing is from zero to negative, remove:
                    pseudo_crossings[snap_num, pidx] = 0
            first_zero_idx = np.argmax(pseudo_crossings, axis=0)
            idx_from_back = np.argmax(
                np.flip(pseudo_crossings, axis=0), axis=0
            )
            last_zero_idx = crossings_indices.shape[0] - idx_from_back - 1

            mask = np.any(pseudo_crossings, axis=0)
            # replace first crossing with earlier pseudo-crossings; set invalid
            # entries (-1) to a high value so that np.maximum ignores them:
            where_invalid = (first_crossing_idx == -1) & mask
            first_crossing_idx[where_invalid] = 99
            first_crossing_idx[mask] = np.minimum(
                first_crossing_idx[mask], first_zero_idx[mask]
            )
            # replace last crossings with later pseudo-crossings
            last_crossing_idx[mask] = np.maximum(
                last_crossing_idx[mask], last_zero_idx[mask]
            )

        return first_crossing_idx, last_crossing_idx

    @staticmethod
    def _interpolate_crossing_redshift(
        redshifts: NDArray,
        differences: NDArray,
        crossing_indices: NDArray,
    ) -> NDArray:
        """
        Find redshift of crossing, given the snap indices.

        Given the array of differences between the particle distance and
        the virial radius of shape (S, N), and a list of redshifts for
        all S snaps, the method interpolates the redshift of all crossings
        over the virial radius, specified by the given indices.

        Returns an estimate for the redshift of every crossing.

        ``differences`` can be computed as follows:

        .. code:: Python

            vr_broadcast = np.broadcast_to(virial_radii[:, None], (S, N))
            differences = distances - multiplier * vr_broadcast

        Where we assume that ``virial_radii`` is the S radii of the
        cluster MP at the S different snapshots, and ``distances`` is the
        shape (S, N) array of distances of particles from the main
        progenitor of the primary subhalo of the cluster, as saved in the
        archive under the field name ``DistanceToMP``.

        :param redshifts: Array of redshifts associated with the S
            snapshots analyzed. Shape (S, ).
        :param differences: The array of the difference between the
            particle distance to the cluster center at every snapshot
            and the virial radius of the cluster at that snapshot. Shape
            (S, N).
        :param crossing_indices: The Array of indices of shape (N, )
            giving for every particle the index of the snapshot where
            the particle crosses the virial radius for the first time,
            i.e. the index ``i`` for which ``differences[i] > 0`` but
            ``differences[i + 1] < 0``.
        :return: The redshifts of the crossings pointed to by
            ``crossing_indices``, interpolated between snapshots.
        """
        z_1 = redshifts[crossing_indices]
        z_2 = redshifts[crossing_indices + 1]
        n = differences.shape[1]
        d_1 = np.array([differences[crossing_indices[i], i] for i in range(n)])
        d_2 = np.array(
            [differences[crossing_indices[i] + 1, i] for i in range(n)]
        )
        z_interp = z_2 - d_2 * (z_2 - z_1) / (d_2 - d_1)
        # set crossing time to NaN for particles that never crossed
        never_crossed_mask = crossing_indices == -1
        if np.any(never_crossed_mask):
            n = np.count_nonzero(never_crossed_mask)
            logging.warning(
                f"Encountered {n} particles that never crossed while "
                f"interpolating redshifts! Positions: "
                f"{np.argwhere(never_crossed_mask)[:, 0]}."
            )
        z_interp[never_crossed_mask] = np.nan
        return z_interp


@dataclasses.dataclass
class TimeOfCrossingPipeline(CrossingUtilMixin, base.Pipeline):
    """
    Find for every particle the time of first and last crossing.
    """

    distance_multiplier: int = 2  # multiple of R_200
    zoom_in: int | None = None

    n_clusters: ClassVar[int] = 352
    n_snaps: ClassVar[int] = 100 - constants.MIN_SNAP

    def run(self) -> int:
        """
        Find time of first and last crossing for every particle.

        :return: Exit code.
        """
        # Step 1: open the archive
        archive_file = h5py.File(self.config.cool_gas_history, "r+")

        # Step 2: load primary subhalo IDs
        group_primaries = halos_daq.get_halo_properties(
            self.config.base_path,
            self.config.snap_num,
            ["GroupFirstSub"],
            cluster_restrict=True,
        )["GroupFirstSub"]

        # Step 2: loop over the zoom-ins
        if self.zoom_in is None:
            logging.info(
                "Finding time of first and last crossing for all particles "
                "in all zoom-ins."
            )
            for zoom_id in range(self.n_clusters):
                self._find_crossing_times(
                    zoom_id, group_primaries[zoom_id], archive_file
                )
        else:
            logging.info(
                f"Finding the time of first and last crossing for all "
                f"particles of zoom-in {self.zoom_in}."
            )
            self._find_crossing_times(
                self.zoom_in, group_primaries[self.zoom_in], archive_file
            )

        archive_file.close()
        logging.info("Done! Successfully found and archived crossing times!")
        return 0

    def _find_crossing_times(
        self, zoom_id: int, primary_subhalo_id: int, archive_file: h5py.File
    ) -> None:
        """
        Find and archive the time of first and last crossing.

        Method loads the virial radius of the given zoom-in FoF and
        uses it to find the time at which each traced particle has
        crossed its virial radius for the first and last time *into**
        the cluster (i.e. crossing from inside the cluster to the
        outside are not counted) and saves the time to the archive file.
        The time is interpolated in between snapshots to reach an
        accurate redshift estimate for the time of crossing.

        If a distance multiplier is set in the init, the time of crossing
        is found with respect to that multiple of the virial radius.

        :param zoom_id: The ID of the zoom-in region for which to find
            the time of crossing.
        :param primary_subhalo_id: ID of the primary subhalo of the
            cluster of the selected zoom-in at redshift zero (snap 99).
            Required to load main progenitor branch virial radii.
        :param archive_file: The archive file containing the cool gas
            tracer history. Must contain the distances of every particle
            to the main progenitor of the cluster at redshift zero,
            given by the field ``DistanceToMP``.
        :return: None, resulting times are archived to the archive file.
        """
        logging.info(f"Finding crossing times for zoom-in {zoom_id}.")
        # Step 0: allocate memory for result
        time_of_first_crossing = np.empty(100)
        time_of_first_crossing[:] = np.nan
        time_of_last_crossing = np.empty(100)
        time_of_last_crossing[:] = np.nan

        # Step 1: load distances from archive
        ds = f"ZoomRegion_{zoom_id:03d}/DistanceToMP"
        distances = archive_file[ds][constants.MIN_SNAP:]

        # Step 2: load virial radius from MPB
        virial_radii = self._get_main_progenitor_radii(
            primary_subhalo_id, zoom_id
        )

        # Step 3: find crossings
        shape = (self.n_snaps, distances.shape[1])
        vr_broadcast = np.broadcast_to(virial_radii[:, None], shape)
        d = distances - self.distance_multiplier * vr_broadcast
        first_crossing, last_crossing = self._first_and_last_zero_crossing(d)
        # Note: the indices returned here are NOT snapshot numbers, but
        # start at MIN_SNAP as 0, i.e. 0 points to MIN_SNAP, not snapshot
        # zero!

        # Step 4: interpolate time of crossing
        redshifts = constants.REDSHIFTS[constants.MIN_SNAP:]
        first_crossing_z = self._interpolate_crossing_redshift(
            redshifts, d, first_crossing
        )
        last_crossing_z = self._interpolate_crossing_redshift(
            redshifts, d, last_crossing
        )

        # Step 5: save crossing times to archive
        logging.info(f"Archiving crossing times for zoom-in {zoom_id}.")
        multiplier = int(self.distance_multiplier)
        sfx = "" if multiplier == 2 else f"{multiplier:d}Rvir"
        # convert crossing indices to actual snapshot numbers
        first_crossing_snap = first_crossing + constants.MIN_SNAP
        first_crossing_snap[first_crossing == -1] = -1
        last_crossing_snap = last_crossing + constants.MIN_SNAP
        last_crossing_snap[last_crossing == -1] = -1

        field_mapping = {
            f"FirstCrossingRedshift{sfx}": first_crossing_z,
            f"LastCrossingRedshift{sfx}": last_crossing_z,
            f"FirstCrossingSnapshot{sfx}": first_crossing_snap,
            f"LastCrossingSnapshot{sfx}": last_crossing_snap,
        }
        for field, value in field_mapping.items():
            grp = f"ZoomRegion_{zoom_id:03d}"
            if field not in archive_file[grp].keys():
                logging.debug(f"Creating missing dataset {field}.")
                archive_file[grp].create_dataset(
                    field, value.shape, value.dtype, data=value
                )
            else:
                archive_file[grp][field][:] = value

    def _get_main_progenitor_radii(
        self, primary_subhalo_id: int, zoom_id: int
    ) -> NDArray:
        """
        Load the virial radius of the primary subhalo along the MPB.

        Function loads the virial radius of the main progenitor branch
        for the given subhalo and returns its radius at all snapshots,
        starting from ``constants.MIN_SNAP``.

        :param primary_subhalo_id: ID of the primary at redshift zero.
        :param zoom_id: ID of the zoom-in region.
        :return: Array of viral radii of the primary along its main
            progenitor branch.
        """
        logging.debug(f"Loading virial radii for zoom-in {zoom_id}.")
        mpb_data = sublink_daq.get_mpb_properties(
            self.config.base_path,
            self.config.snap_num,
            primary_subhalo_id,
            fields=[self.config.radius_field],
            start_snap=constants.MIN_SNAP,
            log_warning=True,
        )
        return mpb_data[self.config.radius_field]


@dataclasses.dataclass
class TimeOfCoolingPipeline(CrossingUtilMixin, base.Pipeline):
    """
    Find for every particle the time of first and last cooling.
    """

    zoom_in: int | None = None

    n_clusters: ClassVar[int] = 352
    n_snaps: ClassVar[int] = 100 - constants.MIN_SNAP
    threshold: ClassVar[float] = 10**4.5  # in Kelvin

    def run(self) -> int:
        """
        Find time of first and last cooling for every particle.

        :return: Exit code.
        """
        # Step 1: open the archive
        archive_file = h5py.File(self.config.cool_gas_history, "r+")

        # Step 2: loop over the zoom-ins
        if self.zoom_in is None:
            logging.info(
                "Finding time of first and last cooling for all particles "
                "in all zoom-ins."
            )
            for zoom_id in range(self.n_clusters):
                self._find_cooling_times(zoom_id, archive_file)
        else:
            logging.info(
                f"Finding the time of first and last cooling for all "
                f"particles of zoom-in {self.zoom_in}."
            )
            self._find_cooling_times(self.zoom_in, archive_file)

        archive_file.close()
        logging.info("Done! Successfully found and archived cooling times!")
        return 0

    def _find_cooling_times(
        self, zoom_id: int, archive_file: h5py.File
    ) -> None:
        """
        Find and archive the time of first and last cooling.

        Method finds the time at which each traced particle has cooled
        from a higher temperature to below the threshold temperature
        (i.e. crossing from a temperature log T > 4.5 to below the
        threshold temperature of log T = 4.5) and saves the time to the
        archive file. The time is interpolated in between snapshots to
        reach an accurate redshift estimate for the time of cooling.

        :param zoom_id: The ID of the zoom-in region for which to find
            the time of cooling.
        :param archive_file: The archive file containing the cool gas
            tracer history. Must contain the temperature of every
            particle, given by the field ``Temperature``.
        :return: None, resulting times are archived to the archive file.
        """
        logging.info(f"Finding cooling times for zoom-in {zoom_id}.")
        # Step 0: allocate memory for result
        time_of_first_cooling = np.empty(100)
        time_of_first_cooling[:] = np.nan
        time_of_last_cooling = np.empty(100)
        time_of_last_cooling[:] = np.nan

        # Step 1: load distances from archive
        ds = f"ZoomRegion_{zoom_id:03d}/Temperature"
        temperatures = archive_file[ds][constants.MIN_SNAP:]

        # Step 2: find crossings
        diff = temperatures - self.threshold
        # We treat particles in stars as hot, i.e. as having a positive
        # difference, so we set this here manually:
        diff[np.isnan(diff)] = 1
        # there are zeros in the diff, so we treat them separately
        first_cooling, last_cooling = self._first_and_last_zero_crossing(
            diff, treat_zeroes=True
        )
        # Note: the indices returned here are NOT snapshot numbers, but
        # start at MIN_SNAP as 0, i.e. 0 points to MIN_SNAP, not snapshot
        # zero!

        # Step 3: interpolate time of crossing
        redshifts = constants.REDSHIFTS[constants.MIN_SNAP:]
        first_crossing_z = self._interpolate_crossing_redshift(
            redshifts, diff, first_cooling
        )
        last_crossing_z = self._interpolate_crossing_redshift(
            redshifts, diff, last_cooling
        )

        # Step 4: save crossing times to archive
        logging.info(f"Archiving cooling times for zoom-in {zoom_id}.")
        # convert from index to snapshot number
        first_cooling_snap = first_cooling + constants.MIN_SNAP
        first_cooling_snap[first_cooling == -1] = -1
        last_cooling_snap = last_cooling + constants.MIN_SNAP
        last_cooling_snap[last_cooling == -1] = -1

        field_mapping = {
            "FirstCoolingRedshift": first_crossing_z,
            "LastCoolingRedshift": last_crossing_z,
            "FirstCoolingSnapshot": first_cooling_snap,
            "LastCoolingSnapshot": last_cooling_snap,
        }
        for field, value in field_mapping.items():
            grp = f"ZoomRegion_{zoom_id:03d}"
            if field not in archive_file[grp].keys():
                logging.debug(f"Creating missing dataset {field}.")
                archive_file[grp].create_dataset(
                    field, value.shape, value.dtype, data=value
                )
            else:
                archive_file[grp][field][:] = value


@dataclasses.dataclass
class ParentCategoryPipeline(base.Pipeline):
    """
    Pipeline to sort particles into categories, based on their parent.

    The following categories exist:
    0 - unbound, no parent ("unbound")
    1 - bound to halo that is not the primary halo ("other halo")
    2 - bound to the primary halo, but not any subhalo ("inner fuzz")
    3 - bound to the primary halo and its primary subhalo ("central galaxy")
    4 - bound to the primary halo and any other subhalo ("satellite")
    255 - faulty entry, cannot assign category (caused by missing entry
          for the corresponding snap in the SUBLINK merger tree)
    """

    zoom_in: int | None = None

    n_clusters: ClassVar[int] = 352

    def run(self) -> int:
        """
        Assign every particle a parent category and archive it.

        :return: Exit code.
        """
        logging.info("Starting pipeline to archive parent category.")

        # Step 1: load primaries at redshift zero
        primaries = halos_daq.get_halo_properties(
            self.config.base_path,
            self.config.snap_num,
            fields=["GroupFirstSub"],
            cluster_restrict=True,
        )["GroupFirstSub"]

        # Step 2: open archive
        archive_file = h5py.File(self.config.cool_gas_history, "r+")

        # Step 3: loop over zoom-ins or process single zoom-in
        if self.zoom_in is None:
            logging.info("Starting loop over zoom-ins for parent category.")
            for zoom_in in range(self.n_clusters):
                self._archive_parent_category(zoom_in, primaries, archive_file)
        else:
            logging.info(
                f"Finding parent categories for zoom-in {self.zoom_in} only."
            )
            self._archive_parent_category(
                self.zoom_in, primaries, archive_file
            )

        logging.info("Done! Archived parent category for all zoom-ins!")
        return 0

    def _archive_parent_category(
        self, zoom_in: int, primaries: NDArray, archive_file: h5py.File
    ) -> None:
        """
        Find and archive the parent categories for a single zoom-in.

        The following categories exist:
        0 - unbound, no parent ("unbound")
        1 - bound to halo that is not the primary halo ("other halo")
        2 - bound to the primary halo, but not any subhalo ("inner fuzz")
        3 - bound to the primary halo and its primary subhalo ("central galaxy")
        4 - bound to the primary halo and any other subhalo ("satellite")
        255 - faulty entry, cannot assign category (caused by missing entry
              for the corresponding snap in the SUBLINK merger tree)

        :param zoom_in: The ID/index of the zoom-in.
        :param primaries: The array of primary subhalo IDs at redshift
            zero for all zoom-ins. ``primaries[zoom_in]`` must therefore
            be the primary subhalo of the current cluster at redshift
            zero.
        :param archive_file: The opened archive file in "r+" mode.
        :return: None, data is archived to the opened archive file.
        """
        logging.debug(f"Finding parent categories for zoom-in {zoom_in}.")
        # Step 0: load MPB info
        mpb = sublink_daq.get_mpb_properties(
            self.config.base_path,
            self.config.snap_num,
            primaries[zoom_in],
            start_snap=constants.MIN_SNAP,
            fields=["SubfindID", "SubhaloGrNr"],
            interpolate=False,  # cannot interpolate IDs
        )
        # find snapshots without data
        skip_snaps = mpb["SnapNum"][mpb["SubfindID"] == -1]

        # Step 1: load the parent IDs
        grp = f"ZoomRegion_{zoom_in:03d}"
        parent_halos = archive_file[grp]["ParentHaloIndex"][()]
        parent_subhalos = archive_file[grp]["ParentSubhaloIndex"][()]

        # Step 2: allocate memory for the parent category
        categories = np.zeros_like(parent_halos, dtype=np.uint8)
        categories[:constants.MIN_SNAP, :] = 255  # invalid before min snap

        # Step 3: assign 255 to all snaps that are missing sublink data
        if skip_snaps.size > 0:
            logging.warning(
                f"Zoom-in {zoom_in}: cannot determine primary subhalo ID "
                f"for snapshots {', '.join(skip_snaps.astype(str))} due to "
                f"snaps missing from SUBLINK. All particles in these snaps "
                f"will be assigned category 255 (\"faulty category\")."
            )
        categories[skip_snaps, :] = 255

        # Step 4: assign category 1 to all in any halo
        not_unbound = parent_halos != -1
        not_invalid = categories != 255
        in_any_halo = np.logical_and(not_unbound, not_invalid)
        categories[in_any_halo] = 1

        # Step 5: assign category 2 to all in current cluster
        for snap_num in range(constants.MIN_SNAP, 100):
            if snap_num in skip_snaps:
                continue
            idx = snap_num - constants.MIN_SNAP
            current_halo_id = mpb["SubhaloGrNr"][idx]
            in_current_halo = parent_halos[snap_num] == current_halo_id
            categories[snap_num][in_current_halo] = 2

        # Step 6: assign category 3 to all in primary subhalo, 4 to all
        # others in subhalos (satellites)
        for snap_num in range(constants.MIN_SNAP, 100):
            if snap_num in skip_snaps:
                continue
            idx = snap_num - constants.MIN_SNAP
            cur_primary_id = mpb["SubfindID"][idx]
            # assign category 3 (in primary)
            in_primary = parent_subhalos[snap_num] == cur_primary_id
            categories[snap_num][in_primary] = 3
            # assign category 4 (satellite)
            not_unbound = parent_subhalos[snap_num] != -1
            # all possible remaining particles are now flagged with 2,
            # since all in the primary have been assigned 3 already:
            in_current_cluster = categories[snap_num] == 2
            in_satellite = np.logical_and(not_unbound, in_current_cluster)
            categories[snap_num][in_satellite] = 4

        # Step 7: archive categories
        grp = f"ZoomRegion_{zoom_in:03d}"
        if "ParentCategory" not in archive_file[grp].keys():
            logging.debug(
                f"Creating missing dataset ParentCategory for zoom-in "
                f"{zoom_in}."
            )
            archive_file[grp].create_dataset(
                "ParentCategory",
                categories.shape,
                categories.dtype,
                data=categories,
            )
        else:
            archive_file[grp]["ParentCategory"][:] = categories
        logging.debug(
            f"Finished archiving parent category of zoom-in {zoom_in}."
        )
