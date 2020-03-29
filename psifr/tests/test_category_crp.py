import unittest
from .. import transitions


class CategoryCRPTestCase(unittest.TestCase):

    def setUp(self) -> None:
        self.pool_position = [1, 2, 3, 4, 5, 6, 7, 8]
        self.pool_category = [1, 1, 1, 1, 2, 2, 2, 2]
        self.output_position = [1, 3, 4, 8, 5, 4, 2, 7, 6]
        self.output_category = [1, 1, 1, 2, 2, 1, 1, 2, 2]

    def test_category_count(self):
        # six transitions, but only five have a valid within-category
        # transition that is possible
        actual, possible = transitions.count_category(
            self.pool_position, [self.output_position],
            [self.pool_category], [self.output_category]
        )
        assert actual == 4
        assert possible == 5


if __name__ == '__main__':
    unittest.main()
