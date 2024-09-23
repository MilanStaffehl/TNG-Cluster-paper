"""
Tests for the sublink DAQ module.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Iterator

import numpy as np
import pytest

from library import constants
from library.data_acquisition import sublink_daq

if TYPE_CHECKING:
    from unittest.mock import Mock

    from numpy.typing import NDArray
    from pytest import LogCaptureFixture


@pytest.fixture
def patch_il_sublink(mocker) -> Iterator[Mock]:
    """Patch il.sublink.loadTree"""
    mock_coords = np.array([
        [2.3, 1.1, 4.5],
        [2.8, 1.0, 4.0],
        [2.3, 1.8, 3.8],
        [1.8, 1.4, 3.5],
        [2.1, 1.6, 3.6],
        [1.5, 1.9, 3.0],
        [1.0, 2.1, 2.8],
        [0.2, 2.8, 2.0],
        [4.8, 3.4, 1.4],
        [4.7, 3.6, 1.2],
    ])  # yapf: disable
    mock_data = {
        "SnapNum":
            np.array([4, 5, 6, 8, 7, 9, 10, 12, 13, 14]),
        "Coordinates":
            mock_coords,
        "Group_R_Crit200":
            np.array([0.2, 0.3, 0.4, 0.6, 0.5, 0.7, 0.8, 1.0, 1.1, 1.2]),
        "SubhaloLen":
            np.array([100, 110, 120, 140, 130, 150, 200, 220, 230, 240]),
    }
    mock_load = mocker.patch("illustris_python.sublink.loadTree")
    mock_load.return_value = mock_data
    yield mock_load


@pytest.fixture
def expected_results() -> Iterator[dict[str, NDArray]]:
    """Return results for ``patch_il_sublink``."""
    expected_coords = np.array([
        [2.3, 1.1, 4.5],
        [2.8, 1.0, 4.0],
        [2.3, 1.8, 3.8],
        [2.1, 1.6, 3.6],
        [1.8, 1.4, 3.5],
        [1.5, 1.9, 3.0],
        [1.0, 2.1, 2.8],
        [0.6, 2.45, 2.4],
        [0.2, 2.8, 2.0],
        [4.8, 3.4, 1.4],
        [4.7, 3.6, 1.2],
    ]) / constants.HUBBLE  # yapf: disable
    expected_radii = np.array(
        [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]
    ) / constants.HUBBLE  # yapf: disable
    expected_data = {
        "SnapNum": np.array([4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]),
        "Coordinates": expected_coords,
        "Group_R_Crit200": expected_radii,
        "SubhaloLen": np.array(
            [100, 110, 120, 130, 140, 150, 200, 210, 220, 230, 240]
        )
    }  # yapf: disable
    yield expected_data


def test_get_mpb_properties(
    patch_il_sublink: Mock,
    expected_results: dict[str, NDArray],
    caplog: LogCaptureFixture
) -> None:
    """Test function to load MPB properties"""
    mock_load = patch_il_sublink

    fields = ["Coordinates", "Group_R_Crit200", "SubhaloLen"]
    output = sublink_daq.get_mpb_properties(
        "path/to/simulation",
        14,
        0,
        fields=fields,
    )

    assert isinstance(output, dict)
    for field in expected_results.keys():
        assert field in output.keys()
        np.testing.assert_allclose(output[field], expected_results[field])

    fields = ["Coordinates", "Group_R_Crit200", "SubhaloLen", "SnapNum"]
    mock_load.assert_called_with(
        "path/to/simulation", 14, 0, fields=fields, onlyMPB=True
    )
    assert len(caplog.messages) == 0


def test_get_mpb_properties_with_start_snap(
    patch_il_sublink: Mock,
    expected_results: dict[str, NDArray],
    caplog: LogCaptureFixture
) -> None:
    """Test function to load MPB properties with a starting snap"""
    mock_load = patch_il_sublink

    fields = ["Coordinates", "Group_R_Crit200", "SubhaloLen"]
    output = sublink_daq.get_mpb_properties(
        "path/to/simulation", 14, 0, fields=fields, start_snap=6
    )

    assert isinstance(output, dict)
    for field in expected_results.keys():
        assert field in output.keys()
        np.testing.assert_allclose(output[field], expected_results[field][2:])

    fields = ["Coordinates", "Group_R_Crit200", "SubhaloLen", "SnapNum"]
    mock_load.assert_called_with(
        "path/to/simulation", 14, 0, fields=fields, onlyMPB=True
    )
    assert len(caplog.messages) == 0


def test_get_mpb_properties_warning(
    patch_il_sublink: Mock,
    expected_results: dict[str, NDArray],
    caplog: LogCaptureFixture
) -> None:
    """Test the warning that is emitted when interpolating"""
    fields = ["Coordinates", "Group_R_Crit200", "SubhaloLen"]
    sublink_daq.get_mpb_properties(
        "path/to/simulation", 14, 0, fields=fields, log_warning=True
    )

    # verify warning
    expected_warning = (
        "Interpolated missing main branch progenitor properties for subhalo "
        "of ID 0 (defined at snapshot 14) at snapshots 11."
    )
    print(caplog.messages)
    assert expected_warning in caplog.messages
