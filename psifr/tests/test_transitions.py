import unittest
import numpy as np
import pandas as pd
from .. import fr


class TransitionsTestCase(unittest.TestCase):

    def setUp(self) -> None:
        self.seq = [0, 2, 3, 7, 4, 3, 6, 5]
        self.all_possible = list(range(8))
        self.lags = list(range(-7, 8))
        self.from_mask = np.array([1, 1, 1, 0, 1, 1, 1, 1], dtype=bool)
        self.to_mask = np.array([1, 1, 0, 1, 1, 1, 1, 1], dtype=bool)
        self.category = np.array([1, 1, 1, 1, 2, 2, 2, 2])
        self.actual = pd.Series(0, dtype='int', index=self.lags)
        self.possible = pd.Series(0, dtype='int', index=self.lags)

    def test_lag(self):
        masker = fr._transition_masker(self.seq, self.all_possible,
                                       self.from_mask, self.to_mask)
        for prev, curr, poss in masker:
            self.actual[curr - prev] += 1
            self.possible[np.subtract(poss, prev)] += 1

        np.testing.assert_array_equal(
            self.actual.to_numpy(),
            np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0]))

        np.testing.assert_array_equal(
            self.possible.to_numpy(),
            np.array([0, 0, 1, 0, 0, 1, 1, 0, 2, 2, 2, 2, 1, 1, 1]))

    def test_lag_no_mask(self):
        masker = fr._transition_masker(self.seq, self.all_possible)
        for prev, curr, poss in masker:
            self.actual[curr - prev] += 1
            self.possible[np.subtract(poss, prev)] += 1

        np.testing.assert_array_equal(
            self.actual.to_numpy(),
            np.array([0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0]))

        np.testing.assert_array_equal(
            self.possible.to_numpy(),
            np.array([0, 1, 1, 0, 1, 2, 3, 0, 3, 3, 3, 3, 2, 1, 1]))

    def test_within_category(self):
        masker = fr._transition_masker(self.seq, self.all_possible,
                                       test_values=self.category,
                                       test=lambda x, y: x == y)
        for prev, curr, poss in masker:
            self.actual[curr - prev] += 1
            self.possible[np.subtract(poss, prev)] += 1

        np.testing.assert_array_equal(
            self.actual.to_numpy(),
            np.array([0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0]))

        np.testing.assert_array_equal(
            self.possible.to_numpy(),
            np.array([0, 0, 0, 0, 1, 1, 3, 0, 2, 1, 1, 0, 0, 0, 0]))


class LagCRPTestCase(unittest.TestCase):

    def setUp(self) -> None:
        self.list_length = 6
        self.recalls = [[0, 2, 3],
                        [5, 4, 1, 2],
                        [0, 4, 1, 2, 5]]
        self.category = [[1, 1, 1, 1, 1, 1],
                         [1, 1, 1, 2, 2, 2],
                         [1, 2, 3, 4, 5, 6]]

    def test_lag_crp(self):
        actual, possible = fr._subject_lag_crp(self.recalls, self.list_length)

        np.testing.assert_array_equal(
            actual.to_numpy(),
            np.array([0, 0, 2, 0, 1, 0, 3, 1, 1, 1, 0]))

        np.testing.assert_array_equal(
            possible.to_numpy(),
            np.array([1, 2, 3, 3, 5, 0, 7, 5, 4, 3, 2]))

    def test_lag_crp_within(self):
        opt = {'test': lambda x, y: x == y, 'test_values': self.category}
        actual, possible = fr._subject_lag_crp(self.recalls, self.list_length,
                                               masker_kws=opt)

        np.testing.assert_array_equal(
            actual.to_numpy(),
            np.array([0, 0, 0, 0, 1, 0, 2, 1, 0, 0, 0]))

        np.testing.assert_array_equal(
            possible.to_numpy(),
            np.array([0, 0, 0, 1, 3, 0, 3, 2, 2, 1, 1]))


if __name__ == '__main__':
    unittest.main()
