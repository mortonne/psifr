"""Test rank distance."""

import numpy as np
import pytest
from psifr import transitions


@pytest.fixture()
def list_data():
    """Create list data with item and category information."""
    data = {
        'pool_items': [[0, 1, 2, 3, 4, 5, 6, 7]],
        'recall_items': [[3, 2, 1, 7, 0, 6, 5]],
        'pool_test': [[1, 1, 2, 2, 2, 2, 1, 1]],
        'recall_test': [[2, 2, 1, 1, 1, 1, 2]],
    }
    return data


@pytest.fixture()
def distances():
    """Create distance matrix."""
    mat = np.array(
        [
            [0, 1, 1, 1, 2, 2, 2, 2],
            [1, 0, 1, 1, 2, 2, 2, 2],
            [1, 1, 0, 1, 2, 2, 2, 2],
            [1, 1, 1, 0, 2, 2, 2, 2],
            [2, 2, 2, 2, 0, 3, 3, 3],
            [2, 2, 2, 2, 3, 0, 3, 3],
            [2, 2, 2, 2, 3, 3, 0, 3],
            [2, 2, 2, 2, 3, 3, 3, 0],
        ]
    )
    return mat


def test_rank_distance(list_data, distances):
    """Test rank distance measure."""
    ranks = transitions.rank_distance(
        distances,
        list_data['pool_items'],
        list_data['recall_items'],
        list_data['pool_items'],
        list_data['recall_items'],
    )
    # recalls: [3, 2, 1, 7, 0, 6, 5]
    # actual: [1, 1, 2, 2, 2, 3]
    # possible: [
    #     [1, 1, 1, 2, 2, 2, 2],
    #     [1, 1, 2, 2, 2, 2],
    #     [1, 2, 2, 2, 2],
    #     [2, 3, 3, 3],
    #     [2, 2, 2],
    #     [3, 3],
    # ]
    # actual_rank: [2.0, 1.5, 3.5, 1.0, 2.0, 1.5]
    # possible_rank: [
    #     [2.0, 2.0, 2.0, 5.5, 5.5, 5.5, 5.5],
    #     [1.5, 1.5, 4.5, 4.5, 4.5, 4.5],
    #     [1.0, 3.5, 3.5, 3.5, 3.5],
    #     [1.0, 3.0, 3.0, 3.0],
    #     [2.0, 2.0, 2.0],
    #     [1.5, 1.5],
    # ]
    expected = np.array(
        [
            1 - (1 / 6),
            1 - (0.5 / 5),
            1 - (2.5 / 4),
            1 - (0 / 3),
            1 - (1 / 2),
            1 - (0.5 / 1),
        ]
    )
    np.testing.assert_allclose(ranks, expected)


def test_rank_distance_within(list_data, distances):
    """Test rank distance for within-category transitions."""
    ranks = transitions.rank_distance(
        distances,
        list_data['pool_items'],
        list_data['recall_items'],
        list_data['pool_items'],
        list_data['recall_items'],
        pool_test=list_data['pool_test'],
        recall_test=list_data['recall_test'],
        test=lambda x, y: x == y,
    )
    expected = np.array([1 - (0 / 2), 1 - (1.5 / 2), 1 - (0 / 1), np.nan])
    np.testing.assert_allclose(ranks, expected)


def test_rank_distance_across(list_data, distances):
    """Test rank distance for across-category transitions."""
    ranks = transitions.rank_distance(
        distances,
        list_data['pool_items'],
        list_data['recall_items'],
        list_data['pool_items'],
        list_data['recall_items'],
        pool_test=list_data['pool_test'],
        recall_test=list_data['recall_test'],
        test=lambda x, y: x != y,
    )
    expected = np.array([1 - (0.5 / 3), 1 - (0.5 / 1)])
    np.testing.assert_allclose(ranks, expected)
