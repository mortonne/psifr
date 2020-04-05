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
                      'pupil', 'absence', 'empty',
                      'fountain', 'piano', 'pillow',
                      'pillow', 'fountain', 'pillow']})
        study = raw.query('trial_type == "study"').copy()
        recall = raw.query('trial_type == "recall"').copy()
        self.data = fr.merge_lists(study, recall)

    def test_lists_input(self):
        m = transitions.TransitionMeasure(self.data, 'input', 'input')
        pool_lists = m.split_lists(self.data, 'input')
        pool_expected = {'items': [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]],
                         'label': [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]}
        assert pool_lists == pool_expected

        recall_lists = m.split_lists(self.data, 'output')
        recall_expected = {'items': [[3.0, 1.0, np.nan], [3.0, 1.0, 3.0]],
                           'label': [[3.0, 1.0, np.nan], [3.0, 1.0, 3.0]]}
        for key in recall_expected.keys():
            assert key in recall_lists
            np.testing.assert_array_equal(recall_lists[key],
                                          recall_expected[key])
        assert 'test' not in recall_lists
