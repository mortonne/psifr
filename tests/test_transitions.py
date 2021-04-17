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
    steps = [[x, y, z.tolist()] for x, y, z in masker]
    expected = [
        [1, 3, [2, 3, 4, 5, 6, 7, 8]],
        [3, 4, [2, 4, 5, 6, 7, 8]],
        [4, 8, [2, 5, 6, 7, 8]],
        [8, 5, [2, 5, 6, 7]],
        [7, 6, [2, 6]],
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
    steps = [[x, y, z.tolist()] for x, y, z in masker]
    expected = [
        [1, 1, [1, 1, 1, 2, 2, 2, 2]],
        [1, 1, [1, 1, 2, 2, 2, 2]],
        [1, 2, [1, 2, 2, 2, 2]],
        [2, 2, [1, 2, 2, 2]],
        [2, 2, [1, 2]],
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
    steps = [[x, y, z.tolist()] for x, y, z in masker]
    expected = [
        [1, 3, [2, 3, 4]],
        [3, 4, [2, 4]],
        [8, 5, [5, 6, 7]],
        [7, 6, [6]]
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
    steps = [[x, y, z.tolist()] for x, y, z in masker]
    expected = [
        [1, 1, [1, 1, 1]],
        [1, 1, [1, 1, 2, 2]],
        [2, 2, [2, 2, 2]],
        [2, 2, [2]]
    ]
    assert steps == expected
