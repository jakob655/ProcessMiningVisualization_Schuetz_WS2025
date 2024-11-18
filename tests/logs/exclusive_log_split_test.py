from logs.splits import exclusive_split
import unittest


class TestExclusiveLogSplit(unittest.TestCase):

    def test_exclusive_split_with_2_partitions(self):
        partitions = [{1, 2}, {3, 4}]

        log = {
            (1, 2): 2,
            (3, 4): 1,
        }

        expected_split_logs = [
            {(1, 2): 2},
            {(3, 4): 1},
        ]

        split_logs = exclusive_split(log, partitions)

        self.assertEqual(expected_split_logs, split_logs)

    def test_exclusive_split_with_3_partitions(self):
        partitions = [{1, 2}, {3, 4}, {5}]

        log = {
            (1, 2): 2,
            (3, 4): 1,
            (5,): 3,
        }

        expected_split_logs = [
            {(1, 2): 2},
            {(3, 4): 1},
            {(5,): 3},
        ]

        split_logs = exclusive_split(log, partitions)

        self.assertEqual(expected_split_logs, split_logs)

    def test_empty_trace_is_ignored(self):
        partitions = [{1, 2}, {3, 4}]

        log = {
            (1, 2): 2,
            (3, 4): 1,
            (): 3,
        }

        expected_split_logs = [
            {(1, 2): 2},
            {(3, 4): 1},
        ]

        split_logs = exclusive_split(log, partitions)

        self.assertEqual(expected_split_logs, split_logs)

    # This test is not necessary, but it is good to have it to ensure that the implementation is correct
    # This Case is not possible in real logs, with exclusive partitions
    def test_trace_with_events_in_multiple_partitions_is_in_no_split(self):
        partitions = [{1, 2}, {3, 4}]

        log = {
            (1, 2, 3, 4): 2,
        }

        expected_split_logs = [
            {},
            {},
        ]

        split_logs = exclusive_split(log, partitions)

        self.assertEqual(expected_split_logs, split_logs)


if __name__ == "__main__":
    unittest.main()
