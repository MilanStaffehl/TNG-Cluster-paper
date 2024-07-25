"""
Tests for common plot library.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Iterator

import numpy as np
import pytest

from library import constants
from library.plotting import common

if TYPE_CHECKING:
    from unittest.mock import Mock


@pytest.fixture
def mock_axes(mocker) -> Iterator[Mock]:
    """Return a mock matplotlib Axes object"""
    mock_axes = mocker.Mock()
    mock_axes.set_xlabel = mocker.Mock()
    mock_axes.set_xticks = mocker.Mock()
    yield mock_axes


def test_make_redshift_plot(mock_axes: Mock) -> None:
    """Test that returned array of redshifts is correct"""
    output = common.make_redshift_plot(mock_axes)
    # check output
    redshifts_plus_offset = np.array(constants.REDSHIFTS) + 0.01
    np.testing.assert_allclose(np.log10(redshifts_plus_offset), output)
    # check mock calls
    mock_axes.set_xlabel.assert_called_with("Redshift")


def test_make_redshift_plot_limited_range(mock_axes: Mock) -> None:
    """Test that returned array of redshifts is correct"""
    output = common.make_redshift_plot(mock_axes, start=8, stop=90)
    # check output
    assert len(output) == 83
    redshifts = np.array(constants.REDSHIFTS)[8:91]
    np.testing.assert_allclose(np.log10(redshifts), output, rtol=1e-5)
    # check mock calls
    mock_axes.set_xlabel.assert_called_with("Redshift")
