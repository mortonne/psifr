import unittest
from .. import transitions


class TransitionsMaskerTestCase(unittest.TestCase):

    def setUp(self) -> None:
        self.pool_position = [1, 2, 3, 4, 5, 6, 7, 8]
        self.pool_category = [1, 1, 1, 1, 2, 2, 2, 2]
        self.output_position = [1, 3, 4, 8, 5, 4, 7, 6]
        self.output_category = [1, 1, 1, 2, 2, 1, 2, 2]

    def test_position(self):
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
