import pytest
import numpy as np
from psifr import transitions


@pytest.fixture()
def data():
    test_list = {
        'list_length': 8,
        'pool_position': [1, 2, 3, 4, 5, 6, 7, 8],
        'pool_category': [1, 1, 2, 2, 1, 1, 2, 2],
        'output_position': [1, 3, 4, 8, 5, 4, 7, 6],
        'output_category': [1, 2, 2, 2, 1, 2, 2, 1],
    }
    return test_list


def test_lag_count(data):
    actual, possible = transitions.count_lags(
        data['list_length'], [data['pool_position']], [data['output_position']]
    )
    np.testing.assert_array_equal(
        actual.to_numpy(), np.array([0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0])
    )
    np.testing.assert_array_equal(
        possible.to_numpy(), np.array([0, 1, 1, 0, 1, 2, 3, 0, 3, 3, 3, 3, 2, 1, 1])
    )


def test_lag_count_category(data):
    # within category
    actual, possible = transitions.count_lags(
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
    actual, possible = transitions.count_lags(
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
