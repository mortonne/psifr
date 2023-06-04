"""Test lag clustering statistics."""

import numpy as np
import pytest

from psifr import stats


@pytest.fixture()
def data():
    """Create list data with position and category."""
    test_list = {
        'list_length': 8,
        'n_block': 4,
        'pool_position': [1, 2, 3, 4, 5, 6, 7, 8],
        'pool_category': [1, 1, 2, 2, 1, 1, 2, 2],
        'pool_block': [1, 1, 2, 2, 3, 3, 4, 4],
        'output_position': [1, 3, 4, 8, 5, 4, 7, 6],
        'output_category': [1, 2, 2, 2, 1, 2, 2, 1],
        'output_block': [1, 2, 2, 4, 3, 2, 4, 3],
    }
    return test_list


@pytest.fixture()
def list_data():
    """Create list data with item and category information."""
    data = {
        'pool_items': [[1, 2, 3, 4, 5, 6]],
        'recall_items': [[6, 2, 1, 5, 4]],
        'pool_test': [[1, 1, 1, 2, 2, 2]],
        'recall_test': [[2, 1, 1, 2, 2]],
    }
    return data


def test_lag_count(data):
    """Test transition counts by serial position lag."""
    actual, possible = stats.count_lags(
        data['list_length'], [data['pool_position']], [data['output_position']]
    )
    np.testing.assert_array_equal(
        actual.to_numpy(), np.array([0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0])
    )
    np.testing.assert_array_equal(
        possible.to_numpy(), np.array([0, 1, 1, 0, 1, 2, 3, 0, 3, 3, 3, 3, 2, 1, 1])
    )


def test_lag_count_category(data):
    """Test transition counts by lag for within-category transitions."""
    # within category
    actual, possible = stats.count_lags(
        data['list_length'],
        [data['pool_position']],
        [data['output_position']],
        pool_test=[data['pool_category']],
        recall_test=[data['output_category']],
        test=lambda x, y: x == y,
    )
    np.testing.assert_array_equal(
        actual.to_numpy(), np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0])
    )
    np.testing.assert_array_equal(
        possible.to_numpy(), np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 2, 1, 0, 0])
    )

    # across category
    actual, possible = stats.count_lags(
        data['list_length'],
        [data['pool_position']],
        [data['output_position']],
        pool_test=[data['pool_category']],
        recall_test=[data['output_category']],
        test=lambda x, y: x != y,
    )
    np.testing.assert_array_equal(
        actual.to_numpy(), np.array([0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0])
    )
    np.testing.assert_array_equal(
        possible.to_numpy(), np.array([0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1])
    )


def test_lag_count_block(data):
    """Test transition counts by block lag."""
    actual, possible = stats.count_lags(
        data['n_block'],
        [data['pool_block']],
        [data['output_block']],
        count_unique=False,
    )
    np.testing.assert_array_equal(actual.to_numpy(), np.array([0, 0, 2, 1, 1, 1, 0]))
    np.testing.assert_array_equal(possible.to_numpy(), np.array([2, 0, 5, 3, 6, 6, 2]))


def test_lag_count_block_unique(data):
    """Test transition counts by unique block lag."""
    actual, possible = stats.count_lags(
        data['n_block'], [data['pool_block']], [data['output_block']], count_unique=True
    )
    np.testing.assert_array_equal(actual.to_numpy(), np.array([0, 0, 2, 1, 1, 1, 0]))
    np.testing.assert_array_equal(possible.to_numpy(), np.array([2, 0, 4, 3, 3, 3, 1]))


def test_compound_lag_count():
    """Test transition lag count conditional on prior lag."""
    list_length = 4
    pool_position = [[1, 2, 3, 4]]
    output_position = [[4, 1, 2, 3]]
    # -3: +1 (+1, +2)
    # +1: +1 (+1)
    # -3:-3, -3:-2, -3:-1, -3:0, -3:+1, -3:+2, -3:+3, ...
    expected_actual = np.hstack(
        (
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
        )
    )
    expected_possible = np.hstack(
        (
            [0, 0, 0, 0, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
        )
    )
    actual, possible = stats.count_lags_compound(
        list_length, pool_position, output_position
    )
    np.testing.assert_array_equal(actual, expected_actual)
    np.testing.assert_array_equal(possible, expected_possible)


def test_percentile_rank():
    """Test calculation of percentile rank."""
    possible = [1, 2, 3, 4]
    rank = []
    for actual in possible:
        rank.append(stats.percentile_rank(actual, possible))
    np.testing.assert_array_equal(np.array(rank), np.array([0, 1 / 3, 2 / 3, 1]))


def test_rank_lag(list_data):
    """Test temporal lag rank."""
    ranks = stats.rank_lags(
        list_data['pool_items'],
        list_data['recall_items'],
        list_data['pool_items'],
        list_data['recall_items'],
    )
    expected = np.array([1 / 4, 5 / 6, 0, 1])
    np.testing.assert_allclose(ranks, expected)


def test_rank_lag_short(list_data):
    """Test temporal lag rank without label inputs."""
    ranks = stats.rank_lags(
        list_data['pool_items'],
        list_data['recall_items'],
    )
    expected = np.array([1 / 4, 5 / 6, 0, 1])
    np.testing.assert_allclose(ranks, expected)


def test_rank_lag_within(list_data):
    """Test temporal lag rank for within-category transitions."""
    ranks = stats.rank_lags(
        list_data['pool_items'],
        list_data['recall_items'],
        pool_test=list_data['pool_test'],
        recall_test=list_data['recall_test'],
        test=lambda x, y: x == y,
    )
    expected = np.array([0.5, np.nan])
    np.testing.assert_array_equal(ranks, expected)


def test_rank_lag_across(list_data):
    """Test temporal lag rank for across-category transitions."""
    ranks = stats.rank_lags(
        list_data['pool_items'],
        list_data['recall_items'],
        pool_test=list_data['pool_test'],
        recall_test=list_data['recall_test'],
        test=lambda x, y: x != y,
    )
    expected = np.array([0.5, 0])
    np.testing.assert_array_equal(ranks, expected)
