import unittest
from logs.splits import loop_split


class TestLoopLogSplit(unittest.TestCase):

    def test_split_with_2_partitions_of_size_1(self):
        partitions = [{1}, {2}]

        log = {
            (1, 2, 1): 2,
        }

        expected_split_logs = [
            {(1,): 4},
            {(2,): 2},
        ]

        split_logs = loop_split(log, partitions)

        self.assertEqual(expected_split_logs, split_logs)

    def test_split_with_more_than_2_partitions(self):
        partitions = [{1}, {2}, {3}]

        log = {
            (1, 2, 1, 3, 1, 2, 1): 1,
            (1,): 1,
            (1, 2, 1): 1,
            (1, 3, 1): 1,
        }

        expected_split_logs = [
            {(1,): 9},
            {(2,): 3},
            {(3,): 2},
        ]

        split_logs = loop_split(log, partitions)

        self.assertEqual(expected_split_logs, split_logs)

    def test_split_with_partitions_with_more_than_one_element(self):
        partitions = [{1, 2, 3}, {4, 5, 6}, {7, 8}]

        log = {
            (1, 2, 3, 4, 5, 6, 1, 2, 3): 1,
            (1, 2, 3, 7, 8, 1, 3): 1,
            (1, 3, 5, 4, 6, 1, 2, 3, 7, 8, 1, 3): 1,
        }

        expected_split_logs = [
            {(1, 2, 3): 4, (1, 3): 3},
            {(4, 5, 6): 1, (5, 4, 6): 1},
            {(7, 8): 2},
        ]

        split_logs = loop_split(log, partitions)

        self.assertEqual(expected_split_logs, split_logs)


if __name__ == "__main__":
    unittest.main()
