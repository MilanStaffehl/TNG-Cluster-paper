"""
Sanity checks for the data generation pipeline for tracers.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Iterator, TypeAlias

import numpy as np
import pytest

from library.config import config
from pipelines.tracer_history.generate import tracer_data

if TYPE_CHECKING:
    from unittest.mock import Mock

    from numpy.typing import NDArray
    from pytest_mock import MockerFixture

# types
FinderPipeline: TypeAlias = tracer_data.FindTracedParticleIDsInSnapshot

# patch paths
PARTICLE_IDS_DAQ = "library.data_acquisition.particle_daq.get_particle_ids"
TRACER_IDS_DAQ = "library.data_acquisition.tracers_daq.load_tracers"


@pytest.fixture
def finder_pipeline() -> Iterator[FinderPipeline]:
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
    pipe = tracer_data.FindTracedParticleIDsInSnapshot(
        config=cfg,
        paths=paths,
        processes=1,
        to_file=False,
        no_plots=True,
        fig_ext="pdf",
        snap_num=98,
    )
    yield pipe


@pytest.fixture
def patch_particle_daq(mocker: MockerFixture) -> Iterator[Mock]:
    """
    Patch the particle DAQ function to load particleIDs.

    Specifically, this method patches the loading function for the
    particle IDs of gas, star and BH particles to return a set of
    integers from 0 to 9, 10 to 19, and 20 to 21 respectively when
    called successively. Yields the resulting mock object.
    """
    # create mock data
    gas_ids = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]) + 12
    star_ids = np.array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19]) + 15
    bh_ids = np.array([20, 21, 22, 23, 24, 25, 26, 27, 28, 29]) + 17

    # patch corresponding calls
    mock_particle_daq = mocker.patch(
        PARTICLE_IDS_DAQ, side_effect=[gas_ids, star_ids, bh_ids]
    )

    yield mock_particle_daq


@pytest.fixture
def patch_match_particle_ids(mocker: MockerFixture) -> Iterator[Mock]:
    """Patch the pipeline method that generates particle indices."""
    # create mock data
    mock_indices = np.array([4, 5, 6, 8, 11, 15, 17, 19, 21, 26, 28, 29])
    mock_flags = np.array([0, 0, 0, 0, 4, 4, 4, 4, 5, 5, 5, 5])
    mock_lens = np.array([4, 4, 4], dtype=np.uint64)

    mock_method = mocker.patch.object(
        tracer_data.FindTracedParticleIDsInSnapshot,
        "_match_particle_ids_to_particles",
    )
    mock_method.return_value = (mock_indices, mock_flags, mock_lens)

    yield mock_method


@pytest.fixture
def patch_numpy_load(mocker: MockerFixture) -> Iterator[Mock]:
    """Patch the ``numpy.load`` function and return the mock"""
    mock_load = mocker.patch("numpy.load")
    yield mock_load


@pytest.fixture
def patch_numpy_savez(mocker: MockerFixture) -> Iterator[Mock]:
    """Patch the ``numpy.savez`` function and return the mock"""
    mock_savez = mocker.patch("numpy.savez")
    yield mock_savez


SEARCH_IDS = (
    np.array([14, 15, 16, 18, 20, 27, 29, 33, 34, 37, 39, 44]),  # sorted
    np.array([14, 27, 40, 17, 31, 32, 45, 21, 26, 25, 41, 21]),  # unsorted
)
EXPECTED_INDICES = (
    np.array([2, 3, 4, 6, 8, 12, 14, 18, 19, 20, 22, 27]),  # sorted
    np.array([2, 12, 23, 5, 16, 17, 28, 9, 11, 10, 24, 9]),  # unsorted
)
EXPECTED_TYPE_FLAGS = (
    np.array([0, 0, 0, 0, 0, 4, 4, 4, 4, 5, 5, 5]),  # for sorted IDs
    np.array([0, 4, 5, 0, 4, 4, 5, 0, 4, 4, 5, 0]),  # for unsorted IDs
)
PARAM_LIST = zip(SEARCH_IDS, EXPECTED_TYPE_FLAGS, EXPECTED_INDICES)


@pytest.mark.parametrize(
    "search_ids, expected_type_flags, expected_indices", PARAM_LIST
)
def test_match_particle_ids_to_particles(
    search_ids: NDArray,
    expected_type_flags: NDArray,
    expected_indices: NDArray,
    finder_pipeline: FinderPipeline,
    patch_particle_daq: Mock,
) -> None:
    """Test that the method selects the desired indices"""
    # call method with mock data
    output = finder_pipeline._match_particle_ids_to_particles(search_ids, 0)

    # verify output
    np.testing.assert_equal(expected_indices, output[0])
    np.testing.assert_equal(expected_type_flags, output[1])
    np.testing.assert_equal(np.array([10, 10, 10], dtype=np.uint64), output[2])

    # assert mock calls
    sim_path = "base/path/of/tng/cluster"
    mock_calls = patch_particle_daq.call_args_list
    assert mock_calls is not None
    assert mock_calls[0].args == (sim_path, 98)
    assert mock_calls[0].kwargs == {"part_type": 0, "zoom_id": 0}
    assert mock_calls[1].args == (sim_path, 98)
    assert mock_calls[1].kwargs == {"part_type": 4, "zoom_id": 0}
    assert mock_calls[2].args == (sim_path, 98)
    assert mock_calls[2].kwargs == {"part_type": 5, "zoom_id": 0}


def test_generate_particle_indices(
    mocker: MockerFixture,
    finder_pipeline: FinderPipeline,
    patch_match_particle_ids: Mock,
    patch_numpy_load: Mock,
    patch_numpy_savez: Mock,
) -> None:
    """Test that the correct data is saved/loaded."""
    # mock tracer DAQ
    mock_tracer_ids = {
        "TracerID": np.arange(100, 135, step=1),
        "ParentID": np.arange(0, 35, step=1),
    }
    mock_tracer_daq = mocker.patch(
        TRACER_IDS_DAQ, return_value=mock_tracer_ids
    )

    # prepare mock for loading tracer IDs from file
    tracer_ids_from_file = np.array(
        [103, 106, 108, 109, 112, 114, 115, 119, 122, 125, 127, 129]
    )
    patch_numpy_load.return_value.__enter__.return_value = {
        "tracer_ids": tracer_ids_from_file
    }

    # call method
    finder_pipeline._generate_particle_indices(0, 0)

    # Verify mock calls of other method.
    # THIS IS MOST IMPORTANT! It is this assertion that verifies that the
    # tracer IDs from the snapshot 99 are correctly translated to their
    # parent IDs and passed on to the method!
    match_call_args = patch_match_particle_ids.call_args[0]
    np.testing.assert_equal(
        np.array([3, 6, 8, 9, 12, 14, 15, 19, 22, 25, 27, 29]),
        match_call_args[0],
    )
    assert match_call_args[1] == 0

    # verify that the load and save mocks were called correctly
    filepath = Path(
        "data/dir/cool_gas_tracer_ids_99/tracer_ids_snapshot99_cluster_0.npz"
    )
    patch_numpy_load.assert_called_with(filepath)
    filepath = Path(
        "data/dir/particle_ids/snapshot_98/particle_ids_zoom_region_0.npz"
    )
    save_call_args = patch_numpy_savez.call_args
    assert save_call_args.args == (filepath, )
    np.testing.assert_equal(
        np.array([4, 5, 6, 8, 11, 15, 17, 19, 21, 26, 28, 29]),
        save_call_args.kwargs["particle_indices"]
    )
    np.testing.assert_equal(
        np.array([0, 0, 0, 0, 4, 4, 4, 4, 5, 5, 5, 5]),
        save_call_args.kwargs["particle_type"]
    )
    np.testing.assert_equal(
        np.array([4, 4, 4], dtype=np.uint64),
        save_call_args.kwargs["total_part_len"],
    )

    # verify that the tracer DAQ was called
    mock_tracer_daq.assert_called_with(
        "base/path/of/tng/cluster", 98, zoom_id=0
    )
