import unittest
import numpy as np
from .. import transitions


class LagCRPTestCase(unittest.TestCase):

    def setUp(self) -> None:
        self.pool_position = [1, 2, 3, 4, 5, 6, 7, 8]
        self.pool_category = [1, 1, 1, 1, 2, 2, 2, 2]
        self.output_position = [1, 3, 4, 8, 5, 4, 7, 6]
        self.output_category = [1, 1, 1, 2, 2, 1, 2, 2]

    def test_lag_count(self):
        actual, possible = transitions.count_lags(
            self.pool_position, [self.output_position],
        )

        np.testing.assert_array_equal(
            actual.to_numpy(),
            np.array([0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0]))

        np.testing.assert_array_equal(
            possible.to_numpy(),
            np.array([0, 1, 1, 0, 1, 2, 3, 0, 3, 3, 3, 3, 2, 1, 1]))

    def test_lag_count_category(self):
        actual, possible = transitions.count_lags(
            self.pool_position, [self.output_position],
            [self.pool_category], [self.output_category],
            lambda x, y: x == y
        )

        np.testing.assert_array_equal(
            actual.to_numpy(),
            np.array([0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0]))

        np.testing.assert_array_equal(
            possible.to_numpy(),
            np.array([0, 0, 0, 0, 1, 1, 3, 0, 2, 1, 1, 0, 0, 0, 0]))
