"""Test high-level operations in the fr module."""

import numpy as np
import pandas as pd
import pytest
from psifr import fr
from psifr import transitions


@pytest.fixture()
def raw():
    raw = pd.DataFrame(
        {
            'subject': [
                1, 1, 1, 1, 1, 1,
                1, 1, 1, 1, 1, 1
            ],
            'list': [
                1, 1, 1, 1, 1, 1,
                2, 2, 2, 2, 2, 2
            ],
            'trial_type': [
                'study', 'study', 'study', 'recall', 'recall', 'recall',
                'study', 'study', 'study', 'recall', 'recall', 'recall',
            ],
            'position': [
                1, 2, 3, 1, 2, 3,
                1, 2, 3, 1, 2, 3
            ],
            'item': [
                'absence', 'hollow', 'pupil', 'hollow', 'pupil', 'empty',
                'fountain', 'piano', 'pillow', 'pillow', 'fountain', 'pillow',
            ],
            'item_index': [
                0, 1, 2, 1, 2, np.nan,
                3, 4, 5, 5, 3, 5,
            ],
            'task': [
                1, 2, 1, 2, 1, np.nan,
                1, 2, 1, 1, 1, 1,
            ]
        }
    )
    return raw


@pytest.fixture()
def data(raw):
    data = fr.merge_free_recall(
        raw, study_keys=['task'], list_keys=['item_index']
    )
    return data


@pytest.fixture()
def distances():
    mat = np.array([[0, 1, 2, 2, 2, 2],
                    [1, 0, 1, 2, 2, 2],
                    [2, 1, 0, 2, 2, 2],
                    [2, 2, 2, 0, 2, 3],
                    [2, 2, 2, 2, 0, 2],
                    [2, 2, 2, 3, 2, 0]])
    return mat


@pytest.fixture()
def distances2():
    distances = np.array([[0, 1, 3, 3, 3, 3],
                          [1, 0, 3, 3, 3, 3],
                          [3, 3, 0, 2, 1, 2],
                          [3, 3, 2, 0, 2, 2],
                          [3, 3, 1, 2, 0, 2],
                          [3, 3, 2, 2, 2, 0]])
    return distances


def test_merge(raw):
    study = raw.loc[raw['trial_type'] == 'study'].copy()
    recall = raw.loc[raw['trial_type'] == 'recall'].copy()
    merged = fr.merge_lists(study, recall)

    # correct recall
    correct = merged.query('item == "pupil"')
    correct = correct.reset_index().loc[0]
    assert correct['input'] == 3
    assert correct['study']
    assert correct['recall']
    assert correct['repeat'] == 0
    assert not correct['intrusion']

    # item not recalled
    forgot = merged.query('item == "piano"')
    forgot = forgot.reset_index().loc[0]
    assert forgot['input'] == 2
    assert forgot['study']
    assert not forgot['recall']
    assert not forgot['intrusion']

    # intrusion
    intrusion = merged.query('item == "empty"')
    intrusion = intrusion.reset_index().loc[0]
    assert np.isnan(intrusion['input'])
    assert not intrusion['study']
    assert intrusion['recall']
    assert intrusion['repeat'] == 0
    assert intrusion['intrusion']

    # repeat
    repeat = merged.query('item == "pillow" and output == 3')
    repeat = repeat.reset_index().loc[0]
    assert repeat['input'] == 3
    assert not repeat['study']
    assert repeat['recall']
    assert repeat['repeat'] == 1
    assert not repeat['intrusion']


def test_filter_raw_data(raw):
    filt = fr.filter_data(raw, 1, [1, 2], 'study', positions=[1, 2])
    assert filt['item'].to_list() == ['absence', 'hollow', 'fountain', 'piano']


def test_filter_merged_data(data):
    filt = fr.filter_data(data, 1, [1, 2], inputs=[1, 2])
    assert filt['item'].to_list() == ['absence', 'hollow', 'fountain', 'piano']

    filt = fr.filter_data(data, 1, [1, 2], outputs=1.0)
    assert filt['item'].to_list() == ['hollow', 'pillow']


def test_split_lists(data):
    study = fr.split_lists(data, 'study', ['item', 'input', 'task'])
    np.testing.assert_allclose(study['input'][1], np.array([1., 2., 3.]))
    recall = fr.split_lists(data, 'recall', ['input'], ['recalls'])
    np.testing.assert_allclose(recall['recalls'][0], np.array([2., 3., np.nan]))


def test_lists_input(data):
    m = transitions.TransitionMeasure('input', 'input')
    pool_lists = m.split_lists(data, 'study')
    pool_expected = {
        'items': [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]],
        'label': [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]],
        'test': None
    }
    assert pool_lists == pool_expected

    recall_lists = m.split_lists(data, 'recall')
    recall_expected = {
        'items': [[2.0, 3.0, np.nan], [3.0, 1.0, 3.0]],
        'label': [[2.0, 3.0, np.nan], [3.0, 1.0, 3.0]],
        'test': None
    }
    for key in recall_expected.keys():
        assert key in recall_lists
        np.testing.assert_array_equal(recall_lists[key], recall_expected[key])


def test_pnr(data):
    stat = fr.pnr(data)
    expected = np.array([0, 0.5, 0.5, 0.5, 0, 1, np.nan, np.nan, np.nan])
    observed = stat['prob'].to_numpy()
    np.testing.assert_array_equal(expected, observed)


def test_lag_crp(data):
    crp = fr.lag_crp(data)
    actual = np.array([1, 0, 0, 1, 0])
    possible = np.array([1, 2, 0, 1, 0])
    prob = np.array([1., 0., np.nan, 1., np.nan])

    np.testing.assert_array_equal(crp['actual'], actual)
    np.testing.assert_array_equal(crp['possible'], possible)
    np.testing.assert_array_equal(crp['prob'], prob)


def test_lag_crp_query(data):
    crp = fr.lag_crp(data, item_query='input != 2')
    actual = np.array([1, 0, 0, 0, 0])
    possible = np.array([1, 0, 0, 0, 0])
    prob = np.array([1., np.nan, np.nan, np.nan, np.nan])

    np.testing.assert_array_equal(crp['actual'], actual)
    np.testing.assert_array_equal(crp['possible'], possible)
    np.testing.assert_array_equal(crp['prob'], prob)


def test_lag_crp_cat(data):
    crp = fr.lag_crp(data, test_key='task', test=lambda x, y: x == y)
    actual = np.array([1, 0, 0, 0, 0])
    possible = np.array([1, 0, 0, 0, 0])
    prob = np.array([1., np.nan, np.nan, np.nan, np.nan])

    np.testing.assert_array_equal(crp['actual'], actual)
    np.testing.assert_array_equal(crp['possible'], possible)
    np.testing.assert_array_equal(crp['prob'], prob)


def test_distance_crp(data, distances2):
    edges = [.5, 1.5, 2.5, 3.5]
    crp = fr.distance_crp(data, 'item_index', distances2, edges, count_unique=False)

    actual = np.array([0, 1, 1])
    possible = np.array([1, 2, 1])
    prob = np.array([0, .5, 1])

    np.testing.assert_array_equal(crp['actual'], actual)
    np.testing.assert_array_equal(crp['possible'], possible)
    np.testing.assert_array_equal(crp['prob'], prob)


def test_distance_crp_unique(data, distances2):
    edges = [.5, 1.5, 2.5, 3.5]
    crp = fr.distance_crp(data, 'item_index', distances2, edges, count_unique=True)

    actual = np.array([0, 1, 1])
    possible = np.array([1, 1, 1])
    prob = np.array([0, 1, 1])

    np.testing.assert_array_equal(crp['actual'], actual)
    np.testing.assert_array_equal(crp['possible'], possible)
    np.testing.assert_array_equal(crp['prob'], prob)


def test_category_crp(data):
    # TODO: replace with more diagnostic test
    crp = fr.category_crp(data, 'task')

    assert crp['prob'].iloc[0] == 1
    assert crp['actual'].iloc[0] == 1
    assert crp['possible'].iloc[0] == 1


def test_lag_rank(data):
    stat = fr.lag_rank(data)
    expected = np.array([0.25])
    observed = stat['rank'].to_numpy()
    np.testing.assert_array_equal(expected, observed)


def test_distance_rank(data, distances):
    stat = fr.distance_rank(data, 'item_index', distances)
    expected = np.array([0.25])
    observed = stat['rank'].to_numpy()
    np.testing.assert_array_equal(expected, observed)
