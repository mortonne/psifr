import unittest
import numpy as np
import pandas as pd
from psifr import transitions
from psifr import fr


class TransitionsMaskerTestCase(unittest.TestCase):

    def setUp(self) -> None:
        self.pool_position = [1, 2, 3, 4, 5, 6, 7, 8]
        self.pool_category = [1, 1, 1, 1, 2, 2, 2, 2]
        self.output_position = [1, 3, 4, 8, 5, 4, 7, 6]
        self.output_category = [1, 1, 1, 2, 2, 1, 2, 2]

    def test_position(self):
        """Test serial position output."""
        masker = transitions.transitions_masker(
            self.pool_position, self.output_position,
            self.pool_position, self.output_position)
        steps = [[x, y, z.tolist()] for x, y, z in masker]
        expected = [[1, 3, [2, 3, 4, 5, 6, 7, 8]],
                    [3, 4, [2, 4, 5, 6, 7, 8]],
                    [4, 8, [2, 5, 6, 7, 8]],
                    [8, 5, [2, 5, 6, 7]],
                    [7, 6, [2, 6]]]
        assert steps == expected

    def test_category(self):
        """Test category output."""
        masker = transitions.transitions_masker(
            self.pool_position, self.output_position,
            self.pool_category, self.output_category)
        steps = [[x, y, z.tolist()] for x, y, z in masker]
        expected = [[1, 1, [1, 1, 1, 2, 2, 2, 2]],
                    [1, 1, [1, 1, 2, 2, 2, 2]],
                    [1, 2, [1, 2, 2, 2, 2]],
                    [2, 2, [1, 2, 2, 2]],
                    [2, 2, [1, 2]]]
        assert steps == expected

    def test_position_cond_category(self):
        """Test position output for within-category transitions."""
        masker = transitions.transitions_masker(
            self.pool_position, self.output_position,
            self.pool_position, self.output_position,
            self.pool_category, self.output_category,
            lambda x, y: x == y)
        steps = [[x, y, z.tolist()] for x, y, z in masker]
        expected = [[1, 3, [2, 3, 4]],
                    [3, 4, [2, 4]],
                    [8, 5, [5, 6, 7]],
                    [7, 6, [6]]]
        assert steps == expected

    def test_category_cond_position(self):
        """Test category output for short-lag transitions."""
        masker = transitions.transitions_masker(
            self.pool_position, self.output_position,
            self.pool_category, self.output_category,
            self.pool_position, self.output_position,
            lambda x, y: np.abs(x - y) < 4)
        steps = [[x, y, z.tolist()] for x, y, z in masker]
        expected = [[1, 1, [1, 1, 1]],
                    [1, 1, [1, 1, 2, 2]],
                    [2, 2, [2, 2, 2]],
                    [2, 2, [2]]]
        assert steps == expected


class TransitionsMeasureTestCase(unittest.TestCase):

    def setUp(self) -> None:
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
        self.data = fr.merge_lists(study, recall, study_keys=['task'],
                                   list_keys=['item_index'])
        self.distances = np.array([[0, 1, 3, 3, 3, 3],
                                   [1, 0, 3, 3, 3, 3],
                                   [3, 3, 0, 2, 1, 2],
                                   [3, 3, 2, 0, 2, 2],
                                   [3, 3, 1, 2, 0, 2],
                                   [3, 3, 2, 2, 2, 0]])
        self.edges = [.5, 1.5, 2.5, 3.5]

    def test_lists_input(self):
        m = transitions.TransitionMeasure('input', 'input')
        pool_lists = m.split_lists(self.data, 'study')
        pool_expected = {'items': [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]],
                         'label': [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]],
                         'test': None}
        assert pool_lists == pool_expected

        recall_lists = m.split_lists(self.data, 'recall')
        recall_expected = {'items': [[2.0, 3.0, np.nan], [3.0, 1.0, 3.0]],
                           'label': [[2.0, 3.0, np.nan], [3.0, 1.0, 3.0]],
                           'test': None}
        for key in recall_expected.keys():
            assert key in recall_lists
            np.testing.assert_array_equal(recall_lists[key],
                                          recall_expected[key])

    def test_lag_crp(self):
        crp = fr.lag_crp(self.data)
        actual = np.array([1, 0, 0, 1, 0])
        possible = np.array([1, 2, 0, 1, 0])
        prob = np.array([1., 0., np.nan, 1., np.nan])

        np.testing.assert_array_equal(crp['actual'], actual)
        np.testing.assert_array_equal(crp['possible'], possible)
        np.testing.assert_array_equal(crp['prob'], prob)

    def test_lag_crp_query(self):
        crp = fr.lag_crp(self.data, item_query='input != 2')
        actual = np.array([1, 0, 0, 0, 0])
        possible = np.array([1, 0, 0, 0, 0])
        prob = np.array([1., np.nan, np.nan, np.nan, np.nan])

        np.testing.assert_array_equal(crp['actual'], actual)
        np.testing.assert_array_equal(crp['possible'], possible)
        np.testing.assert_array_equal(crp['prob'], prob)

    def test_lag_crp_cat(self):
        crp = fr.lag_crp(self.data, test_key='task', test=lambda x, y: x == y)
        actual = np.array([1, 0, 0, 0, 0])
        possible = np.array([1, 0, 0, 0, 0])
        prob = np.array([1., np.nan, np.nan, np.nan, np.nan])

        np.testing.assert_array_equal(crp['actual'], actual)
        np.testing.assert_array_equal(crp['possible'], possible)
        np.testing.assert_array_equal(crp['prob'], prob)

    def test_distance_crp(self):
        crp = fr.distance_crp(self.data, 'item_index', self.distances,
                              self.edges, count_unique=False)
        actual = np.array([0, 1, 1])
        possible = np.array([1, 2, 1])
        prob = np.array([0, .5, 1])

        np.testing.assert_array_equal(crp['actual'], actual)
        np.testing.assert_array_equal(crp['possible'], possible)
        np.testing.assert_array_equal(crp['prob'], prob)

    def test_distance_crp_unique(self):
        crp = fr.distance_crp(self.data, 'item_index', self.distances,
                              self.edges, count_unique=True)
        actual = np.array([0, 1, 1])
        possible = np.array([1, 1, 1])
        prob = np.array([0, 1, 1])

        np.testing.assert_array_equal(crp['actual'], actual)
        np.testing.assert_array_equal(crp['possible'], possible)
        np.testing.assert_array_equal(crp['prob'], prob)

    def test_category_crp(self):
        # TODO: replace with more diagnostic test
        crp = fr.category_crp(self.data, 'task')

        assert crp['prob'].iloc[0] == 1
        assert crp['actual'].iloc[0] == 1
        assert crp['possible'].iloc[0] == 1

    def test_percentile_rank(self):
        possible = [1, 2, 3, 4]
        rank = []
        for actual in possible:
            rank.append(transitions.percentile_rank(actual, possible))
        np.testing.assert_array_equal(np.array(rank),
                                      np.array([0, 1 / 3, 2 / 3, 1]))
