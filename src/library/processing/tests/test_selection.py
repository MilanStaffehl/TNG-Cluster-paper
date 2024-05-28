"""Tests for the selection module"""
import logging

import numpy as np
from pytest_subtests import SubTests

from library.processing import selection


def test_digitize_clusters() -> None:
    """Test that overflowing masses are assigned correct indices"""
    test_masses = 10**np.array([14.1, 14.3, 14.7, 15.0, 15.3999, 15.4001])
    expected = np.array([1, 2, 4, 6, 7, 7])
    output = selection.digitize_clusters(test_masses)
    np.testing.assert_array_equal(expected, output)


def test_digitize_clusters_custom_bins() -> None:
    """Test the function for custom bin edges"""
    test_masses = np.array([1, 4, 3, 0, 5, 600, -1])
    test_bins = np.array([0, 2, 4])
    expected = np.array([1, 2, 2, 1, 2, 2, 0])
    output = selection.digitize_clusters(test_masses, test_bins)
    np.testing.assert_array_equal(expected, output)


def test_select_if_in_s_is_subset_of_a(subtests: SubTests) -> None:
    """Test for case: a unique, s is subset of a"""
    a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
    s = np.array([2, 4, 6])
    expected = np.array([1, 3, 5])  # indices of a where s is found

    for mode in ["iterate", "searchsorted"]:
        with subtests.test(msg=f"mode {mode}", mode=mode):
            output = selection.select_if_in(a, s, mode=mode)
            np.testing.assert_equal(output, expected)


def test_select_if_in_s_is_not_subset_of_a() -> None:
    """Test for case: a unique, s is not subset of a"""
    a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
    s = np.array([2, 4, 0])  # zero not in a

    # for mode iterate this should return a correct result
    expected = np.array([1, 3])
    output = selection.select_if_in(a, s, mode="iterate")
    np.testing.assert_equal(output, expected)

    # in mode searchsorted, execution passes with wrong result
    expected = np.array([1, 3, 0])  # last entry is wrong
    output = selection.select_if_in(a, s, mode="searchsorted")
    np.testing.assert_equal(output, expected)


def test_select_if_in_a_is_not_unique_s_is_subset() -> None:
    """Test for case: a is not unique, s is subset of a"""
    a = np.array([1, 1, 1, 2, 3, 3, 4, 5, 5, 5, 6, 7, 8, 9, 9])
    s = np.array([1, 2, 5])

    # for mode iterate the correct result should be returned
    expected = np.array([0, 1, 2, 3, 7, 8, 9])
    output = selection.select_if_in(a, s, mode="iterate")
    np.testing.assert_equal(output, expected)

    # in mode searchsorted, the function falls short and returns only
    # the first index of every duplicated value
    expected = np.array([0, 3, 7])  # duplicate entries are missing
    output = selection.select_if_in(a, s, mode="searchsorted")
    np.testing.assert_equal(output, expected)


def test_select_if_in_a_is_not_unique_s_is_not_subset() -> None:
    """Test for case: a is not unique, a is not subset of a"""
    a = np.array([1, 1, 1, 2, 3, 3, 4, 5, 5, 5, 6, 7, 8, 9, 9])
    s = np.array([1, 2, 5, 0])

    # in mode iterate this will give the correct result
    expected = np.array([0, 1, 2, 3, 7, 8, 9])
    output = selection.select_if_in(a, s, mode="iterate")
    np.testing.assert_equal(output, expected)

    # in mode searchsorted, entries are missing and there is a wrong
    # entry present due to the zero being sorted in as well
    expected = np.array([0, 3, 7, 0])
    output = selection.select_if_in(a, s, mode="searchsorted")
    np.testing.assert_equal(output, expected)


def test_select_if_in_with_unsorted_input() -> None:
    """Test that unsorted input produces correct results"""
    a = np.array([0, 6, 8, 2, 3, 1, 4, 7, 9])
    s = np.array([1, 8, 0])

    # in mode iterate, the indices are sorted
    expected = np.array([0, 2, 5])
    output = selection.select_if_in(a, s, mode="iterate")
    np.testing.assert_equal(output, expected)

    # in mode searchsorted, the indices are in order of occurrence in s
    expected = np.array([5, 2, 0])
    output = selection.select_if_in(a, s, mode="searchsorted")
    np.testing.assert_equal(output, expected)


def test_select_if_in_unknown_mode(caplog) -> None:
    """Test behavior when given a wrong mode name"""
    a = np.array([1, 2, 3, 4, 5])
    s = np.array([2, 4])
    expected = np.array([np.nan])
    with caplog.at_level(logging.ERROR):
        output = selection.select_if_in(a, s, mode="notamode")

    # verify results
    np.testing.assert_equal(output, expected)
    msg = "Unsupported mode notamode for `selection.select_if_in`."
    assert msg in caplog.text


def test_select_if_in_mode_detect_s_subset_a_unique(caplog) -> None:
    """Test mode detect for: a unique, s subset of a"""
    a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
    s = np.array([1, 3, 7])
    expected = np.array([0, 2, 6])

    with caplog.at_level(logging.DEBUG):
        output = selection.select_if_in(a, s, mode="detect")
    np.testing.assert_equal(output, expected)
    msg = (
        "`select_if_in1: mode `detect` found `a` has only unique "
        "entries; `s` is a subset of `a`. Set mode to `searchsorted`."
    )
    assert msg in caplog.text


def test_select_if_inf_mode_detect_s_subset_a_not_unique(caplog) -> None:
    """Test mode detect for: a not unique, s subset of a"""
    a = np.array([1, 1, 1, 2, 3, 3, 4, 5, 5, 5, 6, 7, 8, 8, 9])
    s = np.array([1, 2, 5])
    expected = np.array([0, 1, 2, 3, 7, 8, 9])

    with caplog.at_level(logging.DEBUG):
        output = selection.select_if_in(a, s, mode="detect")
    np.testing.assert_equal(output, expected)
    msg = (
        "`select_if_in`: mode `detect` found duplicate entries in "
        "array `a`. Set mode to `iterate`."
    )
    assert msg in caplog.text


def test_select_if_in_mode_detect_s_not_subset_a_unique(caplog) -> None:
    """Test mode detect for: a unique, s not subset of a"""
    a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
    s = np.array([1, 2, 5, 0])
    expected = np.array([0, 1, 4])

    with caplog.at_level(logging.DEBUG):
        output = selection.select_if_in(a, s, mode="detect")
    np.testing.assert_equal(output, expected)
    msg = (
        "`select_if_in`: mode `detect` found `s` to not be a subset "
        "of `a`. Set mode to `iterate`."
    )
    assert msg in caplog.text


def test_select_if_in_mode_detect_s_not_subset_a_not_unique(caplog) -> None:
    """Test mode detect for:  not unique, s not subset of a"""
    a = np.array([1, 1, 1, 2, 3, 3, 4, 4, 5, 6, 7, 8, 9])
    s = np.array([1, 2, 5, 0])
    expected = np.array([0, 1, 2, 3, 8])

    with caplog.at_level(logging.DEBUG):
        output = selection.select_if_in(a, s, mode="detect")
    np.testing.assert_equal(output, expected)
    msg = (
        "`select_if_in`: mode `detect` found duplicate entries in "
        "array `a`. Set mode to `iterate`."
    )
    assert msg in caplog.text
    # assert that no unnecessary checks were performed
    msg = (
        "`select_if_in`: mode `detect` found `s` to not be a subset "
        "of `a`. Set mode to `iterate`."
    )
    assert msg not in caplog.text


def test_select_if_in_s_not_unique() -> None:
    """Test the behavior when s contains duplicates"""
    a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
    s = np.array([1, 3, 5, 1])

    # in mode iterate, the duplicate element is not duplicated
    expected = np.array([0, 2, 4])
    output = selection.select_if_in(a, s, mode="iterate")
    np.testing.assert_equal(output, expected)

    # in mode searchsorted, the duplicate element causes a duplicate idx
    expected = np.array([0, 2, 4, 0])
    output = selection.select_if_in(a, s, mode="searchsorted")
    np.testing.assert_equal(output, expected)


def test_select_if_in_s_and_a_not_unique() -> None:
    """Test the behavior when both s and a contain duplicates"""
    a = np.array([1, 1, 1, 2, 3, 4, 5, 5, 6, 7, 7, 7, 8, 9, 9])
    s = np.array([1, 3, 5, 1])

    # in mode iterate, the duplicate element is not duplicated
    expected = np.array([0, 1, 2, 4, 6, 7])
    output = selection.select_if_in(a, s, mode="iterate")
    np.testing.assert_equal(output, expected)

    # in mode searchsorted, the duplicate element causes a duplicate idx
    expected = np.array([0, 4, 6, 0])  # completely wrong
    output = selection.select_if_in(a, s, mode="searchsorted")
    np.testing.assert_equal(output, expected)
