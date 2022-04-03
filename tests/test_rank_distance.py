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


def test_rank_distance_shifted(list_data, distances):
    """Test rank distance based on shifted recalls."""
    list_data['recall_items'] = [[4, 3, 1, 7, 3, 0, 6, 2]]
    distances[3, 1] = 4
    distances[1, 3] = 4
    # sequences: [[4, 3, 1], [3, 1, 7], [0, 6, 5]]
    # -1
    # actual: [4, 2, 2]
    # possible: [
    #     [1, 1, 2, 2, 2, 4],
    #     [1, 1, 2, 2, 2],
    #     [2, 3],
    # ]
    # rank: [0.0, 0.25, 1.0]
    # -2
    # actual: [2, 2, 1]
    # possible: [
    #     [2, 2, 2, 3, 3, 3],
    #     [1, 1, 2, 2, 2],
    #     [1, 2],
    # ]
    # rank: [0.8, 0.25, 1.0]
    ranks = transitions.rank_distance_shifted(
        distances,
        2,
        list_data['pool_items'],
        list_data['recall_items'],
        list_data['pool_items'],
        list_data['recall_items'],
    )
    expected = np.array([[0.8, 0.0], [0.25, 0.25], [1.0, 1.0]])
    np.testing.assert_allclose(ranks, expected)


def test_rank_distance_window():
    """Test rank distance relative to items in a window."""
    distances = np.array(
        [
            [0, 3, 1, 1, 2, 2, 2, 2],
            [3, 0, 1, 1, 2, 2, 2, 2],
            [1, 1, 0, 4, 2, 2, 2, 2],
            [1, 1, 4, 0, 2, 2, 2, 2],
            [2, 2, 2, 2, 0, 5, 3, 3],
            [2, 2, 2, 2, 5, 0, 3, 3],
            [2, 2, 2, 2, 3, 3, 0, 6],
            [2, 2, 2, 2, 3, 3, 6, 0],
        ]
    )
    pool = [[1, 2, 3, 4, 5, 6, 7, 8]]
    outputs = [[4, 2, 1, 7, 5, 3, 4]]
    pool_index = [[4, 5, 6, 7, 0, 1, 2, 3]]
    outputs_index = [[3, 1, 0, 6, 4, 2, 3]]
    list_length = 8
    window_lags = [-1, 0, 1]

    # included: [(4, 2), (7, 5), (5, 3)]
    # prev: [[3, 4, 5], [6, 7, 8], [4, 5, 6]]
    # curr: [2, 5, 3]
    # poss: [[1, 2, 6, 7, 8], [3, 5], [3, 8]]

    # prev: [[6, 7, 0], [1, 2, 3], [7, 0, 1]]
    # curr: [5, 0, 6]
    # poss: [[4, 5, 1, 2, 3], [6, 0], [6, 3]]

    # -1
    # actual: [3, 3, 6]
    # possible: [[3, 3, 2, 2, 2], [2, 3], [6, 2]]
    # rank: [0.125, 0.0, 0.0]

    # 0
    # actual: [3, 1, 2]
    # possible: [[3, 3, 2, 2, 2], [2, 1], [2, 1]]
    # rank: [0.125, 1.0, 0.0]

    # +1
    # actual: [2, 1, 2]
    # possible: [[2, 2, 3, 1, 1], [2, 1], [2, 1]]
    # rank: [0.375, 1.0, 0.0]
    ranks = transitions.rank_distance_window(
        distances,
        list_length,
        window_lags,
        pool,
        outputs,
        pool_index,
        outputs_index
    )
    expected = np.array([[0.125, 0.125, 0.375], [0, 1, 1], [0, 0, 0]])
    np.testing.assert_allclose(ranks, expected)
