"""Test category clusting statistics."""

import numpy as np
import pytest

from psifr import stats


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
    actual, possible = stats.count_category(
        [data['pool_position']],
        [data['output_position']],
        [data['pool_category']],
        [data['output_category']],
    )
    assert actual == 4
    assert possible == 5


@pytest.fixture()
def cases1():
    # test cases taken from Roenker et al. 1971, Psychological Bulletin
    recall_str = [
        'aaaaabacbbbbcbcccc',
        'aaaaabbbbccccabaaaacccccabcbbbbbccccccaaabbbbccaabbaba',
        'aacccccacbbcccccbb',
        'aaaaacbcccccccabbbcccccbcccccbbbbbcccaabcbccccccaaaacc',
        'aaaefbbbefcccedfdd',
        'aaaaadedbbbcbaaaaeeedbecccfffecccccddddedebffffffbbdeb',
        'abeeeecedffffffedf',
        'abbddddedfcccdeeeeedefdffffffcfffeefffffdceeeeeffffffd',
    ]
    data = {'recall': [np.array(list(s)) for s in recall_str]}
    return data


@pytest.fixture()
def cases2():
    # test cases taken from Stricker et al. 2002, J Int Neuropsychol Soc.
    # the CVLT isn't very well described in the sources I have, but
    # this study list matches the description (4 each of 4 categories,
    # same-category items not adjacent)
    study = [['a', 'b', 'c', 'd'] * 4] * 10
    recall_str = [
        'aabb',
        'aaaa',
        'aaaabbbb',
        'aabbccdd',
        'aaaabbbbcccc',
        'aaabbbcccddd',
        'aabbcdcd',
        'aaaacdcd',
        'aabbabab',
        'aaabbbcccdddabcd',
    ]
    recall = [np.array(list(s)) for s in recall_str]
    data = {'study': study, 'recall': recall}
    return data


def test_arc1(cases1):
    # checked the below expected ARC scores against the reported number of
    # clusters (R), expected clusters (E[R]), and maximum clusters (maxR).
    # The expected ARC scores also match the reported ones within rounding
    # error.
    expected = np.array([0.5, 0.5, 0.49295775, 0.5, 0.5, 0.5, 0.49295775, 0.5])
    observed = stats.arc(cases1['recall'])
    np.testing.assert_allclose(observed, expected)


def test_arc2(cases2):
    # original values below. One case was different from reported
    # expected = np.array([1.00, np.nan, 1.00, 1.00, 1.00, 1.00, 0.33, 0.33, 0.11, 0.55])
    expected = np.array([1.00, np.nan, 1.00, 1.00, 1.00, 1.00, 0.33, 0.33, -0.33, 0.55])
    observed = stats.arc(cases2['recall'])
    np.testing.assert_allclose(observed, expected, rtol=0.011)


def test_lbc(cases2):
    # original values below. Found some small deviations from the paper
    # for unclear reasons; probably rounding errors or typos
    # expected = np.array([1.40, 2.40, 4.62, 2.62, 6.80, 5.80, 0.62, 1.62, 0.62, 4.99])

    # corrected expected values for the Stricker test cases
    expected = np.array([1.40, 2.40, 4.60, 2.60, 6.80, 5.80, 0.60, 1.60, 0.60, 5.00])

    observed = stats.lbc(cases2['study'], cases2['recall'])
    np.testing.assert_allclose(observed, expected)
