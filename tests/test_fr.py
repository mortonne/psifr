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


def test_split_lists(frame):
    study = fr.split_lists(frame, 'study', ['item', 'input', 'task'])
    np.testing.assert_allclose(study['input'][1], np.array([1., 2., 3.]))
    recall = fr.split_lists(frame, 'recall', ['input'], ['recalls'])
    np.testing.assert_allclose(recall['recalls'][0], np.array([2., 3., np.nan]))
