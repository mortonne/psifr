"""Test high-level operations in the fr module."""

import numpy as np
import pandas as pd
import pytest
from psifr import fr


@pytest.fixture()
def frame():
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
    study = raw.query('trial_type == "study"').copy()
    recall = raw.query('trial_type == "recall"').copy()
    data = fr.merge_lists(study, recall, study_keys=['task'],
                          list_keys=['item_index'])
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


def test_split_lists(frame):
    study = fr.split_lists(frame, 'study', ['item', 'input', 'task'])
    np.testing.assert_allclose(study['input'][1], np.array([1., 2., 3.]))
    recall = fr.split_lists(frame, 'recall', ['input'], ['recalls'])
    np.testing.assert_allclose(recall['recalls'][0], np.array([2., 3., np.nan]))


def test_pnr(frame):
    stat = fr.pnr(frame)
    expected = np.array([0, 0.5, 0.5, 0.5, 0, 1, np.nan, np.nan, np.nan])
    observed = stat['prob'].to_numpy()
    np.testing.assert_array_equal(expected, observed)


def test_lag_rank(frame):
    stat = fr.lag_rank(frame)
    expected = np.array([0.25])
    observed = stat['rank'].to_numpy()
    np.testing.assert_array_equal(expected, observed)


def test_distance_rank(frame, distances):
    stat = fr.distance_rank(frame, 'item_index', distances)
    expected = np.array([0.25])
    observed = stat['rank'].to_numpy()
    np.testing.assert_array_equal(expected, observed)
