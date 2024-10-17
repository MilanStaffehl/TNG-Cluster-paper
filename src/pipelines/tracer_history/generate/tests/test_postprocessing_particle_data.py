"""
Tests for the postprocessing pipelines.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy
import numpy as np
import pytest

from library import constants
from library.config import config
from pipelines.tracer_history.generate import postprocess_particle_data

if TYPE_CHECKING:
    from pytest_mock import MockFixture


def test_first_and_last_zero_crossing(caplog) -> None:
    """Test method to find crossings."""
    caplog.set_level(logging.WARNING)
    # yapf: disable
    mock_dist = np.array([
        [8, 7, 6, 5, 3, 1, -1, -2, -3],  # simple case
        [4, 3, -1, -2, 1, 3, 1, -1, -2],  # two crossings
        [1, -1, 1, -1, 1, -1, 1, -1, 1],  # multiple crossings
        [1, 2, 3, 4, 5, 6, 7, 8, 9],  # no crossings
        [-4, -3, -2, -1, 1, 2, 3, 4, 5],  # wrong direction
        [4, 3, 2, 1, 0, -1, -2, -3, -4],  # crossing with 0
    ]).transpose()
    # yapf: enable

    klass = postprocess_particle_data.TimeOfCrossingPipeline
    output = klass._first_and_last_zero_crossing(mock_dist)
    expected_first = np.array([5, 1, 0, -1, -1, -1])
    expected_last = np.array([5, 6, 6, -1, -1, -1])
    np.testing.assert_array_equal(expected_first, output[0])
    np.testing.assert_array_equal(expected_last, output[1])

    # check warning was logged
    assert len(caplog.messages) > 0
    expected_msg = (
        "Encountered difference with values exactly zero! This means "
        "some crossing indices will not be correct!"
    )
    assert expected_msg == caplog.messages[0]


def test_first_and_last_zero_crossing_real_data(caplog) -> None:
    """Test method to find crossings."""
    caplog.set_level(logging.WARNING)
    mock_radii = np.linspace(45, 65, num=92)
    mock_dist = np.zeros((92, 10))  # 92 snaps, 10 particles
    for i in range(10):
        # create distances for i-th particle
        mock_dist[:, i] = np.flip(np.linspace(i, i + 100, num=92))

    # we perform the exact same operation as the pipeline to get to the
    # difference between mock_dist and mock_radii:
    vr_broadcast = np.broadcast_to(mock_radii[:, None], (92, 10))
    diff = mock_dist - vr_broadcast

    # expected result
    expected = np.array([41, 42, 43, 43, 44, 45, 46, 47, 47, 48])

    klass = postprocess_particle_data.TimeOfCrossingPipeline
    output = klass._first_and_last_zero_crossing(diff)
    np.testing.assert_array_equal(expected, output[0])
    np.testing.assert_array_equal(expected, output[1])

    # check no warning was logged
    assert len(caplog.messages) == 0


def test_interpolate_crossing_redshift() -> None:
    """Test function to interpolate redshifts"""
    redshifts = np.array([0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5])
    differences = np.array(
        [
            [3, 2, 1, -1, -2, -3, -4, -5, -6],
            [3, 2, 2, 2, 2, 1, -1, -1, -1],
            [3, 2, 4, 5, 6, 7, 5, 3, 2],  # never crosses
            [5, 6, 1, 2, -3, -4, 2, 0, -1],
            [5, 5, 3, 4, 6, -8, -2, -1, 2],
        ]
    ).transpose()  # yapf: ignore
    indices = np.array([2, 5, -1, 3, 4])  # match actual transitions

    # create expected redshifts by manually interpolating one-by-one
    expected_redshifts = np.zeros_like(indices, dtype=np.float32)
    for i, index in enumerate(indices):
        if i == 2:
            expected_redshifts[i] = np.nan
            continue
        xs = np.flip(differences[index:index + 2, i])
        ys = np.flip(redshifts[index:index + 2])
        expected_redshifts[i] = np.interp(0, xs, ys)

    klass = postprocess_particle_data.TimeOfCrossingPipeline
    output = klass._interpolate_crossing_redshift(
        redshifts, differences, indices
    )
    np.testing.assert_allclose(output, expected_redshifts, rtol=1e-3)


@pytest.fixture
def parent_category_pipeline():
    """Build a mock pipeline for testing."""
    cfg = config.Config(
        "TNG-Cluster",
        "base/path/of/tng/cluster",
        99,
        "Group_M_Crit200",
        "Group_R_Crit200",
        Path("data/"),
        Path("figures/"),
        None,
    )
    paths = {
        "data_dir": Path("data/dir"),
        "figures_dir": Path("figures/dir"),
        "data_file_stem": "data_file_stem",
        "figures_file_stem": "figures_file_stem",
    }
    pipe = postprocess_particle_data.ParentCategoryPipeline(
        config=cfg,
        paths=paths,
        processes=1,
        to_file=False,
        no_plots=True,
        fig_ext="pdf",
        zoom_in=0,
    )
    yield pipe


@pytest.fixture
def mock_archive_file():
    """Create a mock archive file"""
    mock_halo_ids = np.array([-1, -1, -1, 300, 100, 100, 100, 200])
    mock_halo_ids = np.broadcast_to(mock_halo_ids[None, :], (100, 8))
    mock_subhalo_ids = np.array([-1, -1, -1, -1, -1, 10, 20, 25])
    mock_subhalo_ids = np.broadcast_to(mock_subhalo_ids[None, :], (100, 8))
    mock_file = {
        "ZoomRegion_000":
            {
                "ParentHaloIndex": mock_halo_ids,
                "ParentSubhaloIndex": mock_subhalo_ids,
                "ParentCategory": np.zeros_like(mock_halo_ids),
            }
    }
    yield mock_file


@pytest.fixture
def patch_sublink_daq(mocker: MockFixture):
    """Patch sublink DAQ to return mock data"""
    mock_subfind_id = np.ones(92) * 10
    mock_subhalo_grnr = np.ones(100) * 100
    # add a few "problems" to the subfind data (we shift index by the
    # value of MIN_SNAP)
    ms = constants.MIN_SNAP
    mock_subfind_id[50 - ms] = -1
    mock_subhalo_grnr[50 - ms] = -1
    mock_subfind_id[67 - ms] = -1
    mock_subhalo_grnr[67 - ms] = -1
    mock_return_data = {
        "SubfindID": mock_subfind_id,
        "SubhaloGrNr": mock_subhalo_grnr,
        "SnapNum": np.arange(8, 100, step=1),
    }
    m = mocker.patch("library.data_acquisition.sublink_daq.get_mpb_properties")
    m.return_value = mock_return_data


def test_parent_category_assignment_method(
    parent_category_pipeline, mock_archive_file, patch_sublink_daq, caplog
) -> None:
    """Test the method that assign parent category to particles"""
    # set caplog to level WARNING
    caplog.set_level(logging.WARNING)

    pipe = parent_category_pipeline
    mock_file = mock_archive_file
    pipe._archive_parent_category(
        zoom_in=0, primaries=[234], archive_file=mock_file
    )
    expected_categories = np.zeros((100, 8), dtype=np.uint8)
    # everything before MIN_SNAP is invalid
    expected_categories[:8, :] = 255
    # expected categories are the same everywhere else...
    expected_categories[8:, :] = np.array([0, 0, 0, 1, 2, 3, 4, 1])
    # ...except for the two "broken" snaps:
    expected_categories[50, :] = 255
    expected_categories[67, :] = 255

    # check generated output
    numpy.testing.assert_array_equal(
        expected_categories, mock_file["ZoomRegion_000"]["ParentCategory"]
    )

    # check logged warnings about missing snaps
    assert len(caplog.records) == 1
    expected_msg = (
        "Zoom-in 0: cannot determine primary subhalo ID for snapshots 50, "
        "67 due to snaps missing from SUBLINK. All particles in these snaps "
        "will be assigned category 255 (\"faulty category\")."
    )
    assert expected_msg in caplog.text
