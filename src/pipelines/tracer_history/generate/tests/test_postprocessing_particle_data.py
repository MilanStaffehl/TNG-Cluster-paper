"""
Tests for the postprocessing pipelines.
"""
import logging

import numpy as np

from pipelines.tracer_history.generate import postprocess_particle_data


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
