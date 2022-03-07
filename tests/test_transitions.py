"""Test transition masker output."""

import pytest
import numpy as np
from psifr import transitions


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


def test_position(list_data):
    """Test serial position output."""
    masker = transitions.transitions_masker(
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
    masker = transitions.transitions_masker(
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
    masker = transitions.transitions_masker(
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
        [7, 7, 6, [6]]
    ]
    assert steps == expected


def test_category_cond_position(list_data):
    """Test category output for short-lag transitions."""
    masker = transitions.transitions_masker(
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
        [7, 2, 2, [2]]
    ]
    assert steps == expected


def test_sequences_position(list_data):
    """Test position masked within a window."""
    masker = transitions.sequences_masker(
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
        [[2, 5, 6, 7, 8], [2, 5, 6, 7]]
    ]
    assert steps == list(zip(output, prev, curr, poss))
