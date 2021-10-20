"""Test counting transition distance bins."""

import pytest
import numpy as np
from psifr import transitions


@pytest.fixture()
def data():
    """Create position data for a list."""
    list_data = {
        'pool_position': [0, 1, 2, 3, 4, 5, 6, 7],
        'recall_position': [3, 2, 1, 7, 0, 6, 5],
    }
    return list_data


@pytest.fixture()
def distance():
    """Create a distance matrix."""
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


def test_distance_count(data, distance):
    """Test distance bin count."""
    edges = [0.5, 1.5, 2.5, 3.5]
    actual, possible = transitions.count_distance(
        distance,
        edges,
        [data['pool_position']],
        [data['recall_position']],
        [data['pool_position']],
        [data['recall_position']],
        count_unique=False,
    )
    # recalls: [3, 2, 1, 7, 0, 6, 5]
    # actual: [1, 1, 2, 2, 2, 3]
    # possible: [
    #     [1, 1, 1, 2, 2, 2, 2],
    #     [1, 1, 2, 2, 2, 2],
    #     [1, 2, 2, 2, 2],
    #     [2, 3, 3, 3],
    #     [2, 2, 2],
    #     [3, 3]
    # ]
    expected_actual = np.array([2, 3, 1])
    expected_possible = np.array([6, 16, 5])
    np.testing.assert_array_equal(actual.to_numpy(), expected_actual)
    np.testing.assert_array_equal(possible.to_numpy(), expected_possible)


def test_distance_count_unique(data, distance):
    """Test distance bin count by transition."""
    edges = [0.5, 2.5, 3.5]
    inputs = [
        distance,
        edges,
        [data['pool_position']],
        [data['recall_position']],
        [data['pool_position']],
        [data['recall_position']],
    ]

    # first check these bins with count_unique=False
    actual, possible = transitions.count_distance(*inputs, count_unique=False)
    np.testing.assert_array_equal(actual.to_numpy(), np.array([5, 1]))
    np.testing.assert_array_equal(possible.to_numpy(), np.array([22, 5]))

    # now test with only one bin increment per transition
    actual, possible = transitions.count_distance(*inputs, count_unique=True)
    np.testing.assert_array_equal(actual.to_numpy(), np.array([5, 1]))
    np.testing.assert_array_equal(possible.to_numpy(), np.array([5, 2]))
