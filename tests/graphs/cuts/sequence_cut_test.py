import unittest
from graphs.dfg import DFG
from graphs.cuts import sequence_cut
from transformations.dataframe_transformations import DataframeTransformations
import pandas as pd


class TestSequenceCut(unittest.TestCase):

    def test_sequence_cut_with_partitions_of_size_1(self):
        log = [["A", "B", "C", "D", "E", "F"]]
        dfg = DFG(log)
        cuts = sequence_cut(dfg)

        self.assertEqual(len(cuts), len(log[0]))
        for i in range(len(log[0])):
            self.assertEqual(set(log[0][i]), cuts[i])

    def test_cut_with_skipped_partitions(self):
        log = [[1, 2, 3], [1, 3]]
        dfg = DFG(log)
        cuts = sequence_cut(dfg)

        self.assertEqual(len(cuts), 3)
        self.assertEqual(cuts[0], {1})
        self.assertEqual(cuts[1], {2})
        self.assertEqual(cuts[2], {3})

    def test_cut_with_partitions_of_different_sizes(self):
        log = [
            ["A", "G", "B", "C", "D"],
            ["A", "G", "C", "B", "D"],
            ["A", "G", "E", "F", "D"],
            ["G", "A", "E", "F", "D"],
            ["G", "Y", "A", "E", "F", "D"],
        ]

        dfg = DFG(log)

        cuts = sequence_cut(dfg)

        self.assertEqual(len(cuts), 3)
        self.assertEqual(cuts[0], {"A", "G", "Y"})
        self.assertEqual(cuts[1], {"B", "C", "E", "F"})
        self.assertEqual(cuts[2], {"D"})

    def test_not_cut_returns_none(self):
        log = [["B", "C"], ["C", "B"]]
        dfg = DFG(log)
        cuts = sequence_cut(dfg)

        self.assertIsNone(cuts)

    def test_with_test_csv_dataset(self):
        dataframe_transformations = DataframeTransformations()
        df = pd.read_csv("tests/testcsv/test_csv.csv")
        dataframe_transformations.set_dataframe(df)

        log = dataframe_transformations.dataframe_to_cases_dict(
            "timestamp", "case", "event"
        )
        dfg = DFG(log)

        cuts = sequence_cut(dfg)

        self.assertEqual(len(cuts), 3)
        self.assertEqual(cuts[0], {"a"})
        self.assertEqual(cuts[1], {"b", "c", "d"})
        self.assertEqual(cuts[2], {"e"})


if __name__ == "__main__":
    unittest.main()
