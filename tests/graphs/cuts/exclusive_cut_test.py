import unittest
from graphs.dfg import DFG
from graphs.cuts import exclusive_cut


class TestExclusiveCut(unittest.TestCase):

    def test_log_with_exclusive_cut_of_size_2(self):
        log = [["A", "B", "C"], ["A", "C", "B"], ["D", "E"]]
        dfg = DFG(log)
        cuts = exclusive_cut(dfg)

        self.assertEqual(len(cuts), 2)
        self.assertIn({"A", "B", "C"}, cuts)
        self.assertIn({"D", "E"}, cuts)

    def test_finding_exclusive_cut_of_size_3(self):
        log = [["A", "B"], ["C", "D"], ["E", "F"]]
        dfg = DFG(log)
        cuts = exclusive_cut(dfg)

        self.assertEqual(len(cuts), 3)
        self.assertIn({"A", "B"}, cuts)
        self.assertIn({"E", "F"}, cuts)
        self.assertIn({"C", "D"}, cuts)

    def test_no_cut_returns_none(self):
        log = [["A", "B", "C"], ["A", "C", "B"]]
        dfg = DFG(log)
        cuts = exclusive_cut(dfg)

        self.assertIsNone(cuts)


if __name__ == "__main__":
    unittest.main()
