import unittest
from logs.splits import sequence_split


class TestSequenceLogSplit(unittest.TestCase):

    def test_split_with_2_partitions_of_size_1(self):
        partitions = [{1}, {2}]

        log = {
            (1, 2): 2,
        }

        expected_split_logs = [
            {(1,): 2},
            {(2,): 2},
        ]

        split_logs = sequence_split(log, partitions)

        self.assertEqual(expected_split_logs, split_logs)

    def test_split_with_more_than_2_partitions(self):
        partitions = [{1, 2}, {3, 4}, {5, 6}]

        log = {
            (1, 2, 3, 4, 5, 6): 1,
            (2, 1, 4, 3, 6, 5): 1,
        }

        expected_split_logs = [
            {(1, 2): 1, (2, 1): 1},
            {(3, 4): 1, (4, 3): 1},
            {(5, 6): 1, (6, 5): 1},
        ]

        split_logs = sequence_split(log, partitions)

        self.assertEqual(expected_split_logs, split_logs)

    def test_split_when_partition_is_skipped(self):
        partitions = [{1}, {2, 3}, {4}]

        log = {
            (1, 2, 3, 4): 1,
            (1, 3, 2, 4): 1,
            (1, 4): 3,
        }

        expected_split_logs = [
            {(1,): 5},
            {(2, 3): 1, (3, 2): 1, (): 3},
            {(4,): 5},
        ]

        split_logs = sequence_split(log, partitions)

        self.assertEqual(expected_split_logs, split_logs)


if __name__ == "__main__":
    unittest.main()
