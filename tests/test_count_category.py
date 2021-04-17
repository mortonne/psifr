"""Test counting category transitions."""

import pytest
from psifr import transitions


@pytest.fixture()
def data():
    """Create list data with position and category."""
    test_list = {
        'list_length': 8,
        'pool_position': [1, 2, 3, 4, 5, 6, 7, 8],
        'pool_category': [1, 1, 1, 1, 2, 2, 2, 2],
        'output_position': [1, 3, 4, 8, 5, 4, 2, 7, 6],
        'output_category': [1, 1, 1, 2, 2, 1, 1, 2, 2],
    }
    return test_list


def test_category_count(data):
    """Test within category actual and possible counts."""
    # six transitions, but only five have a valid within-category
    # transition that is possible
    actual, possible = transitions.count_category(
        [data['pool_position']],
        [data['output_position']],
        [data['pool_category']],
        [data['output_category']],
    )
    assert actual == 4
    assert possible == 5
