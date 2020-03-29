import unittest
import numpy as np
from .. import transitions


class PairCRPTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.n_item = 8
        self.pool_items = [list(range(0, 4)),
                           list(range(4, 8))]
        self.recall_items = [[0, 2, 3, 0],
                             [7, 4, 5, 6]]

    def test_pair_count(self):
        actual, possible = transitions.count_pairs(self.n_item,
                                                   self.pool_items,
                                                   self.recall_items)
        actual_expected = np.array([[0, 0, 1, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 1, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 1, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 1, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 1, 0, 0, 0]])
        possib_expected = np.array([[0, 1, 1, 1, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 1, 0, 1, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 1, 1, 0],
                                    [0, 0, 0, 0, 0, 0, 1, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 1, 1, 1, 0]])
        np.testing.assert_array_equal(actual, actual_expected)
        np.testing.assert_array_equal(possible, possib_expected)


if __name__ == '__main__':
    unittest.main()
