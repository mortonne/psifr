"""Test counting recalls by output."""

import numpy as np
import pytest

from psifr import stats


@pytest.fixture()
def list_data():
    """Create lists with position and category."""
    data = {
        'list_length': 4,
        'pool_items': [[1, 2, 3, 4], [1, 2, 3, 4]],
        'recall_items': [[4, 3], [4, 1, 2, 3]],
        'pool_test': [[1, 1, 2, 2], [1, 1, 2, 2]],
        'recall_test': [[2, 2], [2, 1, 1, 2]],
    }
    return data


def test_count_output(list_data):
    """Test actual and possible serial positions by output position."""
    actual, possible = stats.count_outputs(
        list_data['list_length'],
        list_data['pool_items'],
        list_data['recall_items'],
        list_data['pool_items'],
        list_data['recall_items'],
    )
    expected_actual = np.array(
        [
            [0, 0, 0, 2],
            [1, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
        ]
    )
    expected_possible = np.array(
        [
            [2, 2, 2, 2],
            [2, 2, 2, 0],
            [0, 1, 1, 0],
            [0, 0, 1, 0],
        ]
    )
    np.testing.assert_array_equal(actual, expected_actual)
    np.testing.assert_array_equal(possible, expected_possible)
