import unittest
import numpy as np
from psifr import transitions


class DistanceCRPTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.pool_position = [0, 1, 2, 3, 4, 5, 6, 7]
        self.recall_position = [3, 2, 1, 7, 0, 6, 5]
        self.distance = np.array([[0, 1, 1, 1, 2, 2, 2, 2],
                                  [1, 0, 1, 1, 2, 2, 2, 2],
                                  [1, 1, 0, 1, 2, 2, 2, 2],
                                  [1, 1, 1, 0, 2, 2, 2, 2],
                                  [2, 2, 2, 2, 0, 3, 3, 3],
                                  [2, 2, 2, 2, 3, 0, 3, 3],
                                  [2, 2, 2, 2, 3, 3, 0, 3],
                                  [2, 2, 2, 2, 3, 3, 3, 0]])
        self.edges = [0.5, 1.5, 2.5, 3.5]

    def test_distance_crp_unique(self):
        actual, possible = transitions.count_distance(
            self.distance, self.edges,
            [self.pool_position], [self.recall_position],
            [self.pool_position], [self.recall_position],
            count_unique=True
        )
        expected_actual = np.array([2, 3, 1])
        expected_possible = np.array([3, 5, 2])
        np.testing.assert_array_equal(actual.to_numpy(), expected_actual)
        np.testing.assert_array_equal(possible.to_numpy(), expected_possible)

    def test_distance_crp(self):
        actual, possible = transitions.count_distance(
            self.distance, self.edges,
            [self.pool_position], [self.recall_position],
            [self.pool_position], [self.recall_position],
            count_unique=False
        )
        expected_actual = np.array([2, 3, 1])
        expected_possible = np.array([6, 16, 5])
        np.testing.assert_array_equal(actual.to_numpy(), expected_actual)
        np.testing.assert_array_equal(possible.to_numpy(), expected_possible)


if __name__ == '__main__':
    unittest.main()
