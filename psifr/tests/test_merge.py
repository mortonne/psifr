import unittest
import numpy as np
import pandas as pd
from psifr import fr


class MergeTestCase(unittest.TestCase):

    def setUp(self) -> None:
        self.data = pd.DataFrame(
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

    def test_merge(self):
        study = self.data.loc[self.data.trial_type == 'study'].copy()
        recall = self.data.loc[self.data.trial_type == 'recall'].copy()
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
        forgot = merged.query('item == "hollow"')
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


if __name__ == '__main__':
    unittest.main()
