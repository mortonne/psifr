"""Test transition masker output."""

import pytest
import numpy as np

from psifr import maskers


@pytest.fixture()
def list_data():
    """Create list data with item and category information."""
    list_data = {
        'pool_position': [1, 2, 3, 4, 5, 6, 7, 8],
        'pool_category': [1, 1, 1, 1, 2, 2, 2, 2],
        'output_position': [1, 3, 4, 8, 5, 4, 7, 6],
        'output_category': [1, 1, 1, 2, 2, 1, 2, 2],
    }
    return list_data


def test_output(list_data):
    """Test serial position by output position."""
    masker = maskers.outputs_masker(
        list_data['pool_position'],
        list_data['output_position'],
        list_data['pool_position'],
        list_data['output_position'],
    )
    steps = [[z, x, y.tolist()] for x, y, z in masker]
    expected = [
        [1, 1, [1, 2, 3, 4, 5, 6, 7, 8]],
        [2, 3, [2, 3, 4, 5, 6, 7, 8]],
        [3, 4, [2, 4, 5, 6, 7, 8]],
        [4, 8, [2, 5, 6, 7, 8]],
        [5, 5, [2, 5, 6, 7]],
        [6, 7, [2, 6, 7]],
        [7, 6, [2, 6]],
    ]
    assert steps == expected


def test_position(list_data):
    """Test serial position output."""
    masker = maskers.transitions_masker(
        list_data['pool_position'],
        list_data['output_position'],
        list_data['pool_position'],
        list_data['output_position'],
    )
    steps = [[p, x, y, z.tolist()] for p, x, y, z in masker]
    expected = [
        [1, 1, 3, [2, 3, 4, 5, 6, 7, 8]],
        [2, 3, 4, [2, 4, 5, 6, 7, 8]],
        [3, 4, 8, [2, 5, 6, 7, 8]],
        [4, 8, 5, [2, 5, 6, 7]],
        [7, 7, 6, [2, 6]],
    ]
    assert steps == expected


def test_category(list_data):
    """Test category output."""
    masker = maskers.transitions_masker(
        list_data['pool_position'],
        list_data['output_position'],
        list_data['pool_category'],
        list_data['output_category'],
    )
    steps = [[p, x, y, z.tolist()] for p, x, y, z in masker]
    expected = [
        [1, 1, 1, [1, 1, 1, 2, 2, 2, 2]],
        [2, 1, 1, [1, 1, 2, 2, 2, 2]],
        [3, 1, 2, [1, 2, 2, 2, 2]],
        [4, 2, 2, [1, 2, 2, 2]],
        [7, 2, 2, [1, 2]],
    ]
    assert steps == expected


def test_position_cond_category(list_data):
    """Test position output for within-category transitions."""
    masker = maskers.transitions_masker(
        list_data['pool_position'],
        list_data['output_position'],
        list_data['pool_position'],
        list_data['output_position'],
        list_data['pool_category'],
        list_data['output_category'],
        lambda x, y: x == y,
    )
    steps = [[p, x, y, z.tolist()] for p, x, y, z in masker]
    expected = [
        [1, 1, 3, [2, 3, 4]],
        [2, 3, 4, [2, 4]],
        [4, 8, 5, [5, 6, 7]],
        [7, 7, 6, [6]],
    ]
    assert steps == expected


def test_category_cond_position(list_data):
    """Test category output for short-lag transitions."""
    masker = maskers.transitions_masker(
        list_data['pool_position'],
        list_data['output_position'],
        list_data['pool_category'],
        list_data['output_category'],
        list_data['pool_position'],
        list_data['output_position'],
        lambda x, y: np.abs(x - y) < 4,
    )
    steps = [[p, x, y, z.tolist()] for p, x, y, z in masker]
    expected = [
        [1, 1, 1, [1, 1, 1]],
        [2, 1, 1, [1, 1, 2, 2]],
        [4, 2, 2, [2, 2, 2]],
        [7, 2, 2, [2]],
    ]
    assert steps == expected


def test_sequences_position(list_data):
    """Test position masked within a window."""
    masker = maskers.sequences_masker(
        2,
        list_data['pool_position'],
        list_data['output_position'],
        list_data['pool_position'],
        list_data['output_position'],
    )
    steps = [tuple([p, x, y, [a.tolist() for a in z]]) for p, x, y, z in masker]
    output = [[1, 2], [2, 3], [3, 4]]
    prev = [[1, 3], [3, 4], [4, 8]]
    curr = [[3, 4], [4, 8], [8, 5]]
    poss = [
        [[2, 3, 4, 5, 6, 7, 8], [2, 4, 5, 6, 7, 8]],
        [[2, 4, 5, 6, 7, 8], [2, 5, 6, 7, 8]],
        [[2, 5, 6, 7, 8], [2, 5, 6, 7]],
    ]
    assert steps == list(zip(output, prev, curr, poss))


@pytest.fixture()
def window_data():
    pool = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    outputs = [16, 15, 9, 1, 4, 8, 13, 14, 4, 7, 5]
    list_length = 16
    return pool, outputs, list_length


def test_windows_position(window_data):
    """Test study position within a window."""
    pool, outputs, list_length = window_data
    window_lags = [-1, 0, 1]

    # outputs: [16, 15, 9, 1, 4, 8, 13, 14, 4, 7, 5]
    # included: [(15, 9), (9, 1), (4, 8), (8, 13), (7, 5)]
    # excluded: [(16, 15), (1, 4), (13, 14), (14, 4), (4, 7)]
    output = [2, 3, 5, 6, 10]
    prev = [[14, 15, 16], [8, 9, 10], [3, 4, 5], [7, 8, 9], [6, 7, 8]]
    curr = [9, 1, 8, 13, 5]
    poss = [
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
        [1, 2, 3, 4, 5, 6, 7, 11, 12, 13, 14],
        [2, 6, 7, 8, 10, 11, 12, 13, 14],
        [2, 3, 5, 6, 10, 11, 12, 13, 14],
        [2, 3, 5, 10, 11, 12],
    ]

    # test the yielded values at each included output position
    masker = maskers.windows_masker(
        list_length, window_lags, pool, outputs, pool, outputs
    )
    for i, (a_output, a_prev, a_curr, a_poss) in enumerate(masker):
        assert a_output == output[i]
        assert a_prev.tolist() == prev[i]
        assert a_curr == curr[i]
        assert a_poss.tolist() == poss[i]


def test_windows_position_exclude_prev(window_data):
    """Test excluding previous recalls within the window."""
    pool, outputs, list_length = window_data
    window_lags = [-1, 0, 1]

    # outputs: [16, 15, 9, 1, 4, 8, 13, 14, 4, 7, 5]
    # included: [(9, 1), (4, 8), (8, 13), (7, 5)]
    # excluded: [(16, 15), (15, 9), (1, 4), (13, 14), (14, 4), (4, 7)]
    # excluded due to previous transition: [(15, 9), (14, 4)]
    output = [3, 5, 6, 10]
    prev = [[8, 9, 10], [3, 4, 5], [7, 8, 9], [6, 7, 8]]
    curr = [1, 8, 13, 5]
    poss = [
        [1, 2, 3, 4, 5, 6, 7, 11, 12, 13, 14],
        [2, 6, 7, 8, 10, 11, 12, 13, 14],
        [2, 3, 5, 6, 10, 11, 12, 13, 14],
        [2, 3, 5, 10, 11, 12],
    ]
    # test the yielded values at each included output position
    masker = maskers.windows_masker(
        list_length, window_lags, pool, outputs, pool, outputs, exclude_prev_window=True
    )
    for i, (a_output, a_prev, a_curr, a_poss) in enumerate(masker):
        assert a_output == output[i]
        assert a_prev.tolist() == prev[i]
        assert a_curr == curr[i]
        assert a_poss.tolist() == poss[i]


def test_windows_position_cond_category(window_data):
    """Test study position within a window conditional on category."""
    pool, outputs, list_length = window_data
    pool_category = [1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2]
    outputs_category = [2, 2, 2, 1, 1, 1, 2, 2, 1, 1, 1]
    window_lags = [-1, 0, 1]
    test = lambda x, y: x == y

    # outputs: [16, 15, 9, 1, 4, 8, 13, 14, 4, 7, 5]
    # included: [(15, 9), (4, 8), (7, 5)]
    # excluded: [(16, 15), (9, 1), (1, 4), (8, 13), (13, 14), (14, 4), (4, 7)]
    output = [2, 5, 10]
    prev = [[14, 15, 16], [3, 4, 5], [6, 7, 8]]
    curr = [9, 8, 5]
    poss = [[9, 10, 11, 12, 13], [2, 6, 7, 8], [2, 3, 5]]

    # test the yielded values at each included output position
    masker = maskers.windows_masker(
        list_length,
        window_lags,
        pool,
        outputs,
        pool,
        outputs,
        pool_category,
        outputs_category,
        test,
    )
    for i, (a_output, a_prev, a_curr, a_poss) in enumerate(masker):
        assert a_output == output[i]
        assert a_prev.tolist() == prev[i]
        assert a_curr == curr[i]
        assert a_poss.tolist() == poss[i]
