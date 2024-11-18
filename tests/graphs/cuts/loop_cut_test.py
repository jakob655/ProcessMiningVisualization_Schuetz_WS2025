import unittest
from graphs.dfg import DFG
from graphs.cuts import loop_cut


class TestLoopCut(unittest.TestCase):

    def test_loop_cut_with_partitions_of_size_1(self):
        log = [["A", "B", "A", "B", "A", "B", "A"]]
        dfg = DFG(log)
        cuts = loop_cut(dfg)

        self.assertEqual(len(cuts), 2)

        self.assertEqual(cuts[0], {"A"})
        self.assertEqual(cuts[1], {"B"})

    def test_loop_cut_returns_None_without_cut(self):
        log = [["A", "B", "C", "D", "E", "F"]]
        dfg = DFG(log)
        cuts = loop_cut(dfg)

        self.assertIsNone(cuts)

    def test_loop_cut_with_more_than_1_redo_part(self):
        log = [
            [1, 2, 3, 1, 2, 3, 1, 4, 1],
            [1, 4, 1, 5, 6, 7, 1, 2, 3, 1],
            [1, 2, 3, 1, 5, 6, 7, 1],
        ]
        dfg = DFG(log)
        cuts = loop_cut(dfg)

        self.assertEqual(len(cuts), 4)
        self.assertEqual(cuts[0], {1})
        self.assertIn({2, 3}, cuts)
        self.assertIn({4}, cuts)
        self.assertIn({5, 6, 7}, cuts)

    def test_partition_merged_when_edge_from_start(self):
        log = [[1, 2, 3, 2, 3, 2, 4, 5, 6, 1, 2, 3, 4]]
        dfg = DFG(log)
        cuts = loop_cut(dfg)

        self.assertEqual(len(cuts), 2)
        self.assertIn(2, cuts[0])
        self.assertIn(3, cuts[0])
        self.assertIn({5, 6}, cuts)

    def test_partition_merged_when_edge_to_end(self):
        log = [[1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 7, 4]]
        dfg = DFG(log)
        cuts = loop_cut(dfg)

        self.assertEqual(len(cuts), 2)
        self.assertIn(7, cuts[0])
        self.assertIn({5, 6}, cuts)

    def test_partition_merged_when_missing_edge_from_end_nodes(self):
        log = [[1, 2, 3, 1, 2], [1, 2, 1, 2, 1, 3, 1], [1, 4, 1, 2]]
        dfg = DFG(log)
        cuts = loop_cut(dfg)

        self.assertEqual(len(cuts), 2)
        self.assertIn(4, cuts[0])
        self.assertIn({3}, cuts)

    def test_partition_merged_when_missing_edge_to_start_nodes(self):
        log = [[1, 2], [2, 1, 2, 3, 4, 1, 2], [1, 2, 5, 1, 2, 5, 2]]
        dfg = DFG(log)
        cuts = loop_cut(dfg)

        self.assertEqual(len(cuts), 2)
        self.assertIn(3, cuts[0])
        self.assertIn(4, cuts[0])
        self.assertIn({5}, cuts)


if __name__ == "__main__":
    unittest.main()
