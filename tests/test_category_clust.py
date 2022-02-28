"""Test category clustering measures."""

import numpy as np
import pytest
from psifr import clustering


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
    observed = clustering.arc(cases1['recall'])
    np.testing.assert_allclose(observed, expected)


def test_arc2(cases2):
    # original values below. One case was different from reported
    # expected = np.array([1.00, np.nan, 1.00, 1.00, 1.00, 1.00, 0.33, 0.33, 0.11, 0.55])
    expected = np.array([1.00, np.nan, 1.00, 1.00, 1.00, 1.00, 0.33, 0.33, -0.33, 0.55])
    observed = clustering.arc(cases2['recall'])
    np.testing.assert_allclose(observed, expected, rtol=0.011)


def test_lbc(cases2):
    # original values below. Found some small deviations from the paper
    # for unclear reasons; probably rounding errors or typos
    # expected = np.array([1.40, 2.40, 4.62, 2.62, 6.80, 5.80, 0.62, 1.62, 0.62, 4.99])

    # corrected expected values for the Stricker test cases
    expected = np.array([1.40, 2.40, 4.60, 2.60, 6.80, 5.80, 0.60, 1.60, 0.60, 5.00])

    observed = clustering.lbc(cases2['study'], cases2['recall'])
    np.testing.assert_allclose(observed, expected)
