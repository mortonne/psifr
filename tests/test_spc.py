import unittest
import numpy as np
import pandas as pd
from psifr import fr


class SPCTestCase(unittest.TestCase):
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
             'task': [1, 2, 1, 2, 1, np.nan,
                      1, 2, 1, 1, 1, 1]})
        study = raw.query('trial_type == "study"').copy()
        recall = raw.query('trial_type == "recall"').copy()
        self.data = fr.merge_lists(study, recall, study_keys=['task'])

    def test_spc(self):
        recall = fr.spc(self.data)
        expected = np.array([.5, .5, 1])
        np.testing.assert_array_equal(recall['recall'].to_numpy(), expected)


if __name__ == '__main__':
    unittest.main()
