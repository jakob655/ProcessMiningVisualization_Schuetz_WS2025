import unittest
from graphs.dfg import DFG


class TestDFG(unittest.TestCase):

    def setUp(self):
        log = [["A", "B", "C"], ["A", "C", "B"]]
        self.dfg = DFG(log)

    def test_dfg_init(self):
        self.assertEqual(self.dfg.get_nodes(), {"A", "B", "C"})
        self.assertEqual(
            self.dfg.get_edges(),
            {("A", "B"), ("B", "C"), ("A", "C"), ("C", "B")},
        )
        self.assertEqual(self.dfg.get_start_nodes(), {"A"})
        self.assertEqual(self.dfg.get_end_nodes(), {"B", "C"})

    def test_add_node(self):
        self.dfg.add_node("D")
        self.assertEqual(self.dfg.get_nodes(), {"A", "B", "C", "D"})

    def test_add_edge(self):
        self.dfg.add_edge("C", "D")
        self.assertEqual(
            self.dfg.get_edges(),
            {("A", "B"), ("B", "C"), ("A", "C"), ("C", "B"), ("C", "D")},
        )

    def test_get_nodes(self):
        self.assertEqual(self.dfg.get_nodes(), {"A", "B", "C"})

    def test_get_edges(self):
        self.assertEqual(
            self.dfg.get_edges(),
            {("A", "B"), ("B", "C"), ("A", "C"), ("C", "B")},
        )

    def test_get_start_nodes(self):
        self.assertEqual(self.dfg.get_start_nodes(), {"A"})

    def test_get_end_nodes(self):
        self.assertEqual(self.dfg.get_end_nodes(), {"B", "C"})

    def test_is_in_start_nodes(self):
        self.assertTrue(self.dfg.is_in_start_nodes("A"))
        self.assertFalse(self.dfg.is_in_start_nodes("B"))
        self.assertFalse(self.dfg.is_in_start_nodes("C"))

    def test_is_in_end_nodes(self):
        self.assertFalse(self.dfg.is_in_end_nodes("A"))
        self.assertTrue(self.dfg.is_in_end_nodes("B"))
        self.assertTrue(self.dfg.is_in_end_nodes("C"))

    def test_contains_node(self):
        self.assertTrue(self.dfg.contains_node("A"))
        self.assertFalse(self.dfg.contains_node("D"))

    def test_contains_edge(self):
        self.assertTrue(self.dfg.contains_edge("A", "B"))
        self.assertFalse(self.dfg.contains_edge("B", "A"))

    def test_connected_components_with(self):
        self.assertEqual(self.dfg.get_connected_components(), [{"A", "B", "C"}])

    def test_connected_components_does_ignore_directions(self):
        log = [["A", "B"], ["B", "C"], ["C", "A"], ["D", "E"], ["D", "A"]]
        dfg = DFG(log)

        self.assertEqual(dfg.get_connected_components(), [{"A", "B", "C", "D", "E"}])

    def test_connected_components_with_more_than_one_connected_component(self):
        self.dfg.add_edge("D", "E")

        connected_components = self.dfg.get_connected_components()
        self.assertIn({"D", "E"}, connected_components)
        self.assertIn({"A", "B", "C"}, connected_components)

    def test_get_reachable_nodes(self):
        self.assertEqual(self.dfg.get_reachable_nodes("A"), {"A", "B", "C"})
        self.assertEqual(self.dfg.get_reachable_nodes("B"), {"B", "C"})
        self.assertEqual(self.dfg.get_reachable_nodes("C"), {"C", "B"})
        self.assertEqual(self.dfg.get_reachable_nodes("D"), {"D"})

    def test_reachable_nodes_with_node_without_edges(self):
        self.dfg.add_node("D")
        self.assertEqual(self.dfg.get_reachable_nodes("D"), {"D"})

    def test_node_without_edges_is_stored_in_dfg(self):
        log = [["A", "B"], ["B", "C"], ["C", "A"], ["D", "E"], ["D", "A"], ["F"]]

        dfg = DFG(log)

        self.assertEqual(dfg.get_nodes(), {"A", "B", "C", "D", "E", "F"})

    def test_inverting_dfg(self):
        inverted_dfg = self.dfg.invert()

        self.assertEqual(inverted_dfg.get_nodes(), {"A", "B", "C"})
        self.assertEqual(
            inverted_dfg.get_edges(),
            {("A", "B"), ("B", "A"), ("A", "C"), ("C", "A")},
        )

    def test_inverting_graph_with_no_edges(self):
        dfg = DFG([["A"], ["B"], ["C"]])

        inverted_dfg = dfg.invert()

        self.assertEqual(inverted_dfg.get_nodes(), {"A", "B", "C"})
        self.assertEqual(
            inverted_dfg.get_edges(),
            {("A", "B"), ("B", "A"), ("A", "C"), ("C", "A"), ("B", "C"), ("C", "B")},
        )

    def test_inverting_dfg_with_node_without_edges(self):
        self.dfg.add_node("D")
        inverted_dfg = self.dfg.invert()

        self.assertEqual(inverted_dfg.get_nodes(), {"A", "B", "C", "D"})
        self.assertEqual(
            inverted_dfg.get_edges(),
            {
                ("A", "B"),
                ("B", "A"),
                ("A", "C"),
                ("C", "A"),
                ("A", "D"),
                ("D", "A"),
                ("B", "D"),
                ("D", "B"),
                ("C", "D"),
                ("D", "C"),
            },
        )

    def test_create_dfg_without_nodes_whithout_edges(self):
        dfg_without_nodes = self.dfg.create_dfg_without_nodes({"A", "B"})

        self.assertEqual(dfg_without_nodes.get_nodes(), {"C"})
        self.assertEqual(dfg_without_nodes.get_edges(), set())

    def test_create_dfg_without_nodes_with_edges(self):
        dfg_without_nodes = self.dfg.create_dfg_without_nodes({"A"})

        self.assertEqual(dfg_without_nodes.get_nodes(), {"B", "C"})
        self.assertEqual(dfg_without_nodes.get_edges(), {("B", "C"), ("C", "B")})


if __name__ == "__main__":
    unittest.main()
