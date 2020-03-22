import unittest
import numpy as np
import pandas as pd
from .. import fr


class TransitionsTestCase(unittest.TestCase):

    def setUp(self) -> None:
        self.seq = [1, 3, 4, 8, 5, 4, 7, 6]
        self.all_possible = list(range(1, 9))
        self.lags = list(range(-7, 8))
        self.from_mask = np.array([1, 1, 1, 0, 1, 1, 1, 1], dtype=bool)
        self.to_mask = np.array([1, 1, 0, 1, 1, 1, 1, 1], dtype=bool)
        self.actual = pd.Series(0, dtype='int', index=self.lags)
        self.possible = pd.Series(0, dtype='int', index=self.lags)

    def test_lag(self):
        masker = fr.transition_masker(self.seq, self.all_possible,
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
        masker = fr.transition_masker(self.seq, self.all_possible)
        for prev, curr, poss in masker:
            self.actual[curr - prev] += 1
            self.possible[np.subtract(poss, prev)] += 1

        np.testing.assert_array_equal(
            self.actual.to_numpy(),
            np.array([0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0]))

        np.testing.assert_array_equal(
            self.possible.to_numpy(),
            np.array([0, 1, 1, 0, 1, 2, 3, 0, 3, 3, 3, 3, 2, 1, 1]))


if __name__ == '__main__':
    unittest.main()
