import unittest
from logs.splits import parallel_split


class TestParallelLogSplit(unittest.TestCase):

    def test_parallel_split_with_2_partitions_of_size_1(self):
        partitions = [{1}, {2}]

        log = {
            (1, 2): 2,
            (2, 1): 4,
        }

        expected_split_logs = [
            {(1,): 6},
            {(2,): 6},
        ]

        split_logs = parallel_split(log, partitions)

        self.assertEqual(expected_split_logs, split_logs)

    def test_parallel_split_with_2_partitions_of_size_2(self):
        partitions = [{1, 2}, {3, 4}]

        log = {
            (1, 2, 3, 4): 1,
            (1, 2, 4, 3): 1,
            (1, 4, 2, 3): 1,
            (1, 4, 3, 2): 1,
            (1, 3, 2, 4): 1,
            (1, 3, 4, 2): 1,
            (2, 1, 3, 4): 1,
            (2, 1, 4, 3): 1,
            (2, 3, 1, 4): 1,
            (2, 3, 4, 1): 1,
            (2, 4, 1, 3): 1,
            (2, 4, 3, 1): 1,
            (3, 1, 2, 4): 1,
            (3, 1, 4, 2): 1,
            (3, 2, 1, 4): 1,
            (3, 2, 4, 1): 1,
            (3, 4, 1, 2): 1,
            (3, 4, 2, 1): 1,
            (4, 1, 2, 3): 1,
            (4, 1, 3, 2): 1,
            (4, 2, 1, 3): 1,
            (4, 2, 3, 1): 1,
            (4, 3, 1, 2): 1,
            (4, 3, 2, 1): 1,
        }

        expected_split_logs = [
            {(1, 2): 12, (2, 1): 12},
            {(3, 4): 12, (4, 3): 12},
        ]

        split_logs = parallel_split(log, partitions)

        self.assertEqual(expected_split_logs, split_logs)

    def test_empty_trace_is_ignored(self):
        partitions = [{1}, {2}]

        log = {
            (1, 2): 2,
            (2, 1): 1,
            (): 3,
        }

        expected_split_logs = [
            {(1,): 3},
            {(2,): 3},
        ]

        split_logs = parallel_split(log, partitions)

        self.assertEqual(expected_split_logs, split_logs)

    def test_empty_trace_in_sublogs_when_not_all_events_in_trace(self):
        partitions = [{1}, {2}]

        log = {
            (1, 2): 2,
            (2, 1): 1,
            (1,): 3,
            (2,): 1,
        }

        expected_split_logs = [
            {(1,): 6, (): 1},
            {(2,): 4, (): 3},
        ]

        split_logs = parallel_split(log, partitions)

        self.assertEqual(expected_split_logs, split_logs)


if __name__ == "__main__":
    unittest.main()
