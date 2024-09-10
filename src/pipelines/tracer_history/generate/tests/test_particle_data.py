"""Selected tests for particle data generation pipeline."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Iterator, TypeAlias

import numpy as np
import pytest

from library.config import config
from pipelines.tracer_history.generate import particle_data

if TYPE_CHECKING:
    from unittest.mock import Mock

    from pytest_mock import MockerFixture
    from pytest_subtests import SubTests

PipelineClass: TypeAlias = particle_data.TraceSimpleQuantitiesBackABC


@pytest.fixture
def distance_to_parent_pipeline() -> Iterator[PipelineClass]:
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
    pipe = particle_data.TraceDistanceToParentHaloPipeline(
        config=cfg,
        paths=paths,
        processes=1,
        to_file=False,
        no_plots=True,
        fig_ext="pdf",
        unlink=False,
        force_overwrite=False,
        zoom_id=None,
        archive_single=False,
    )
    yield pipe


@pytest.fixture
def patch_loader(mocker: MockerFixture) -> Iterator[Mock]:
    """Patch the loading function for particle data."""
    # yapf: disable
    mock_halo_positions = np.array(
        [
            [0, 0, 0],
            [3, 6, 8],
            [10, 9, 0],
            [-5, 8, 9],
            [-1, -2, -5],
            [1, -14, -5],
            [2, 8, -1],
            [1, 9, 6],
            [9, 4, 5],
            [-9, -12, -6],
        ],
        dtype=np.float32,
    )  # shape (10, 3)
    # yapf: enable
    m = mocker.patch("library.data_acquisition.halos_daq.get_halo_properties")
    m.return_value = {"Coordinates": mock_halo_positions, "count": 10}
    yield m


def test_distance_to_parent_halo_pipeline_process_to_quantity_method(
    distance_to_parent_pipeline: PipelineClass,
    patch_loader: Mock,
    subtests: SubTests,
    mocker: MockerFixture,
) -> None:
    """Test the method to process particle positions into distances."""
    # set up pipeline
    pipeline = distance_to_parent_pipeline

    # set up mock data
    mock_parent_indices = np.empty((3, 6), dtype=np.int32)
    mock_parent_indices[:] = -1
    mock_parent_indices[1] = np.array([0, -1, 3, 5, 9, -1])
    mock_hdf5_file = {"ZoomRegion_089/ParentHaloIndex": mock_parent_indices}
    mock_particle_positions = np.array(
        [
            [1, -2, 3],
            [0, 2, 4],
            [-9, 8, 10],
            [0, 0, 0],
            [2, -4, -8],
            [-1, -1, -1],
        ],
        dtype=np.float32,
    )  # yapf: disable

    with subtests.test(msg="No box boundaries"):
        # call method with mock data
        output = pipeline._process_into_quantity(
            89,
            1,
            mock_particle_positions,
            -1,
            mock_hdf5_file  # type: ignore
        )

        # check output
        expected = np.array(
            [
                np.sqrt(14),
                np.nan,
                np.sqrt(17),
                np.sqrt(222),
                np.sqrt(189),
                np.nan
            ]
        )
        np.testing.assert_allclose(output, expected, rtol=1e-6)

    # test the same thing again but make the box smaller
    with subtests.test(msg="Box boundaries limited to 10.0"):
        # mock box size
        mock_dict = {"TNG-Cluster": 10.0}
        mocker.patch("library.constants.BOX_SIZES", mock_dict)

        # call method with mock data
        output = pipeline._process_into_quantity(
            89,
            1,
            mock_particle_positions,
            -1,
            mock_hdf5_file  # type: ignore
        )

        # check output
        expected = np.array(
            [np.sqrt(14), np.nan, np.sqrt(17), np.sqrt(42), 3.0, np.nan]
        )
        np.testing.assert_allclose(output, expected, rtol=1e-6)
