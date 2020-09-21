"""Test high-level operations in the fr module."""

import numpy as np
import pandas as pd
import pytest
from psifr import fr


@pytest.fixture()
def raw():
    raw = pd.DataFrame(
        {'subject': [1, 1, 1, 1, 1, 1,
                     1, 1, 1, 1, 1, 1],
         'list': [1, 1, 1, 1, 1, 1,
                  2, 2, 2, 2, 2, 2],
         'trial_type': ['study', 'study', 'study',
                        'recall', 'recall', 'recall',
                        'study', 'study', 'study',
                        'recall', 'recall', 'recall'],
         'position': [1, 2, 3, 1, 2, 3,
                      1, 2, 3, 1, 2, 3],
         'item': ['absence', 'hollow', 'pupil',
                  'hollow', 'pupil', 'empty',
                  'fountain', 'piano', 'pillow',
                  'pillow', 'fountain', 'pillow'],
         'item_index': [0, 1, 2, 1, 2, np.nan,
                        3, 4, 5, 5, 3, 5],
         'task': [1, 2, 1, 2, 1, np.nan,
                  1, 2, 1, 1, 1, 1]})
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


def test_filter_raw_data(raw):
    filt = fr.filter_data(raw, 1, [1, 2], 'study', positions=[1, 2])
    assert filt['item'].to_list() == ['absence', 'hollow', 'fountain', 'piano']


def test_filter_merged_data(data):
    filt = fr.filter_data(data, 1, [1, 2], inputs=[1, 2])
    assert filt['item'].to_list() == ['absence', 'hollow', 'fountain', 'piano']


def test_split_lists(data):
    study = fr.split_lists(data, 'study', ['item', 'input', 'task'])
    np.testing.assert_allclose(study['input'][1], np.array([1., 2., 3.]))
    recall = fr.split_lists(data, 'recall', ['input'], ['recalls'])
    np.testing.assert_allclose(recall['recalls'][0], np.array([2., 3., np.nan]))


def test_pnr(data):
    stat = fr.pnr(data)
    expected = np.array([0, 0.5, 0.5, 0.5, 0, 1, np.nan, np.nan, np.nan])
    observed = stat['prob'].to_numpy()
    np.testing.assert_array_equal(expected, observed)


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
