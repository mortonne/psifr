"""Test rank measures of temporal clustering."""

import numpy as np
import pytest
from psifr import transitions


def test_percentile_rank():
    """Test calculation of percentile rank."""
    possible = [1, 2, 3, 4]
    rank = []
    for actual in possible:
        rank.append(transitions.percentile_rank(actual, possible))
    np.testing.assert_array_equal(np.array(rank), np.array([0, 1 / 3, 2 / 3, 1]))


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


def test_rank_lags(list_data):
    """Test temporal lag rank."""
    ranks = transitions.rank_lags(
        list_data['pool_items'],
        list_data['recall_items'],
        list_data['pool_items'],
        list_data['recall_items'],
    )
    expected = np.array([1 / 4, 5 / 6, 0, 1])
    np.testing.assert_allclose(ranks, expected)


def test_rank_lags_short(list_data):
    """Test temporal lag rank without label inputs."""
    ranks = transitions.rank_lags(
        list_data['pool_items'],
        list_data['recall_items'],
    )
    expected = np.array([1 / 4, 5 / 6, 0, 1])
    np.testing.assert_allclose(ranks, expected)


def test_rank_lags_within(list_data):
    """Test temporal lag rank for within-category transitions."""
    ranks = transitions.rank_lags(
        list_data['pool_items'],
        list_data['recall_items'],
        pool_test=list_data['pool_test'],
        recall_test=list_data['recall_test'],
        test=lambda x, y: x == y,
    )
    expected = np.array([0.5, np.nan])
    np.testing.assert_array_equal(ranks, expected)


def test_rank_lags_across(list_data):
    """Test temporal lag rank for across-category transitions."""
    ranks = transitions.rank_lags(
        list_data['pool_items'],
        list_data['recall_items'],
        pool_test=list_data['pool_test'],
        recall_test=list_data['recall_test'],
        test=lambda x, y: x != y,
    )
    expected = np.array([0.5, 0])
    np.testing.assert_array_equal(ranks, expected)
