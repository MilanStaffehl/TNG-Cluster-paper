"""
Tests for loading tracer data from simulations.
"""
from pathlib import Path

import numpy as np
import pytest

from library.config import config
from library.data_acquisition import tracers_daq

base_path = config.get_simulation_base_path("TNG-Cluster")


@pytest.mark.skipif(
    not Path(base_path).exists(),
    reason="Can only be executed if simulation data exists."
)
def test_load_traces_tracer_ids(subtests):
    """Tracer IDs must be the same in every snapshot"""
    for halo_id in [0, 252455, 476245]:
        with subtests.test(msg=f"Cluster {halo_id}"):
            # load data for three snapshots
            tracer_data_99 = tracers_daq.load_tracers(
                base_path, 99, ["TracerID"], cluster_id=halo_id
            )
            tracer_data_98 = tracers_daq.load_tracers(
                base_path, 98, ["TracerID"], cluster_id=halo_id
            )
            tracer_data_97 = tracers_daq.load_tracers(
                base_path, 97, ["TracerID"], cluster_id=halo_id
            )

            # check that only the requested field was loaded
            assert "ParentID" not in tracer_data_99.keys()
            assert "ParentID" not in tracer_data_98.keys()
            assert "ParentID" not in tracer_data_97.keys()

            # extract actual data arrays
            tracer_ids_99 = tracer_data_99["TracerID"]
            tracer_ids_98 = tracer_data_98["TracerID"]
            tracer_ids_97 = tracer_data_97["TracerID"]

            # assert the arrays all have the same length
            assert tracer_ids_99.shape == tracer_ids_98.shape
            assert tracer_ids_99.shape == tracer_ids_98.shape
            assert tracer_ids_98.shape == tracer_ids_97.shape

            # assert all tracer IDs are the same (no tracers lost or added)
            np.testing.assert_equal(tracer_ids_99, tracer_ids_98)
            np.testing.assert_equal(tracer_ids_99, tracer_ids_98)
            np.testing.assert_equal(tracer_ids_98, tracer_ids_97)


@pytest.mark.skipif(
    not Path(base_path).exists(),
    reason="Can only be executed if simulation data exists."
)
def test_load_tracers_parent_ids(subtests):
    """Parent IDs should not be equal throughout snaps, but have same length"""
    for halo_id in [0, 252455, 476245]:
        with subtests.test(msg=f"Cluster {halo_id}"):
            # load data for three snapshots
            parent_data_99 = tracers_daq.load_tracers(
                base_path, 99, ["ParentID"], cluster_id=halo_id
            )
            parent_data_98 = tracers_daq.load_tracers(
                base_path, 98, ["ParentID"], cluster_id=halo_id
            )
            parent_data_97 = tracers_daq.load_tracers(
                base_path, 97, ["ParentID"], cluster_id=halo_id
            )

            # check only requested field was loaded
            assert "TracerID" not in parent_data_99.keys()
            assert "TracerID" not in parent_data_98.keys()
            assert "TracerID" not in parent_data_97.keys()

            # extract actual data arrays
            parent_ids_99 = parent_data_99["ParentID"]
            parent_ids_98 = parent_data_98["ParentID"]
            parent_ids_97 = parent_data_97["ParentID"]

            # assert all arrays have the same shape
            assert parent_ids_99.shape == parent_ids_98.shape
            assert parent_ids_99.shape == parent_ids_98.shape
            assert parent_ids_98.shape == parent_ids_97.shape

            # assert arrays do NOT have same content (tracers moved)
            assert np.any(np.not_equal(parent_ids_99, parent_ids_98))
            assert np.any(np.not_equal(parent_ids_99, parent_ids_98))
            assert np.any(np.not_equal(parent_ids_98, parent_ids_97))
