"""Test counting transition item pairs."""

import pytest
import numpy as np

from psifr import stats


@pytest.fixture()
def data():
    """Create list data with item numbers."""
    list_data = {
        'n_item': 8,
        'pool_items': [list(range(0, 4)), list(range(4, 8))],
        'recall_items': [[0, 2, 3, 0], [7, 4, 5, 6]],
    }
    return list_data


def test_pair_count(data):
    """Test transition counts by item pairs."""
    actual, possible = stats.count_pairs(
        data['n_item'], data['pool_items'], data['recall_items']
    )
    actual_expected = np.array(
        [
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
        ]
    )
    possib_expected = np.array(
        [
            [0, 1, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 1, 0],
        ]
    )
    np.testing.assert_array_equal(actual, actual_expected)
    np.testing.assert_array_equal(possible, possib_expected)
