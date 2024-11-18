import unittest
from graphs.dfg import DFG
from graphs.cuts import parallel_cut


class TestParallelCut(unittest.TestCase):
    def test_no_cut_returns_none(self):
        log = [["A", "B", "C"], ["A", "B", "C"]]
        dfg = DFG(log)
        cuts = parallel_cut(dfg)

        self.assertIsNone(cuts)

    def test_cut_return_partitions(self):
        log = [["B", "C"], ["C", "B"]]
        dfg = DFG(log)
        cuts = parallel_cut(dfg)

        self.assertEqual(len(cuts), 2)
        self.assertIn({"B"}, cuts)
        self.assertIn({"C"}, cuts)

    def test_partitions_without_start_or_end_node_merge(self):
        log = [
            ["A", "B", "C"],
            ["A", "C", "B"],
            ["B", "A", "C"],
            ["B", "C", "A"],
        ]
        dfg = DFG(log)
        cuts = parallel_cut(dfg)

        self.assertEqual(len(cuts), 2)
        # C is merged with other nodes
        self.assertNotIn({"C"}, cuts)

    def test_partitions_with_start_node_are_first_merged_with_end_node_partitions(self):
        log = [
            ["A", "B", "C", "D"],
            ["A", "B", "D", "C"],
            ["A", "D", "B", "C"],
            ["A", "C", "B", "D"],
            ["B", "A", "C", "D"],
            ["B", "C", "A", "D"],
            ["B", "C", "D", "A"],
            ["D", "A", "B", "C"],
        ]
        dfg = DFG(log)
        cuts = parallel_cut(dfg)

        self.assertEqual(len(cuts), 3)
        self.assertIn({"A"}, cuts)
        self.assertIn({"B", "C"}, cuts)
        self.assertIn({"D"}, cuts)


if __name__ == "__main__":
    unittest.main()
