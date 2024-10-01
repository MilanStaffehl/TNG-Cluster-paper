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


@dataclasses.dataclass
class TimeOfCrossingPipeline(base.Pipeline):
    """
    Find for every particle the time of first and last crossing.
    """

    distance_multiplier: int = 2  # multiple of R_200c
    zoom_id: int | None = None

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
        if self.zoom_id is None:
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
                f"particles of zoom-in {self.zoom_id}."
            )
            self._find_crossing_times(
                self.zoom_id, group_primaries[self.zoom_id], archive_file
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
        field_mapping = {
            "FirstCrossingRedshift": first_crossing_z,
            "LastCrossingRedshift": last_crossing_z,
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

    @staticmethod
    def _first_and_last_zero_crossing(
        differences: NDArray,
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

        :param differences: An array of shape (S, N) containing the
            difference ``part_distance - virial_radius``.
        :return: The array of indices of the first change from positive
            to negative in ``distances`` along the S-axis of shape (N, )
            and one array of the indices of the last change from positive
            to negative. Note that the index points to the last entry of
            ``differences`` that is positive before it changes sign.
        """
        if np.count_nonzero(differences) != differences.size:
            logging.warning(
                "Encountered difference with values exactly zero! This means "
                "some crossing indices will not be correct!"
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
        z_interp[never_crossed_mask] = np.nan
        return z_interp
