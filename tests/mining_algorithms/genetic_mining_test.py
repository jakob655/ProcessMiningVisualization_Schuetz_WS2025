import unittest
from collections import deque

import pandas as pd

from graphs.visualization.base_graph import BaseGraph
from graphs.visualization.genetic_graph import GeneticGraph
from logger import get_logger
from mining_algorithms.genetic_mining import GeneticMining
from transformations.dataframe_transformations import DataframeTransformations
from transformations.utils import cases_list_to_dict


class TestGeneticMining(unittest.TestCase):

    def __init__(self, methodName: str = "runTest"):
        super().__init__(methodName)
        self.logger = get_logger("GeneticMining")

    def setUp(self):
        self.log = {
            ("A", "B", "H"): 1,
            ("A", "C", "H"): 1,
            ("A", "D", "E", "F", "G", "H"): 1,
            ("A", "D", "F", "E", "G", "H"): 1
        }
        self.gm = GeneticMining(self.log)

    def test_initialize_dependency_matrix(self):
        self.gm._initialize_dependency_matrix()
        self.assertTrue(self.gm.dependency_matrix, "Dependency matrix was not initialized")
        self.assertTrue(self.gm.start_measures, "Start measures not set")
        self.assertTrue(self.gm.end_measures, "End measures not set")

    def test_create_heuristic_individual(self):
        self.gm._initialize_dependency_matrix()
        activities = list(self.gm.events)
        ind = self.gm._create_heuristic_individual(activities, 1)
        self.assertIn("I", ind, "Individual missing input sets")
        self.assertIn("O", ind, "Individual missing output sets")

    def test_evaluate_fitness(self):
        self.gm._initialize_dependency_matrix()
        activities = list(self.gm.events)
        ind = self.gm._create_heuristic_individual(activities, 1)
        fitness = self.gm._evaluate_fitness(ind)
        self.assertGreaterEqual(fitness, 0.0)
        self.assertLessEqual(fitness, 1.0)

    def test_crossover_and_mutation(self):
        self.gm._initialize_dependency_matrix()
        activities = list(self.gm.events)
        p1 = self.gm._create_heuristic_individual(activities, 1)
        p2 = self.gm._create_heuristic_individual(activities, 1)
        child1, child2 = self.gm._crossover(p1, p2)
        self.assertIn("I", child1)
        self.assertIn("O", child2)
        self.gm._mutate(child1, 0.9)
        self.assertIn("I", child1)

    def test_resolve_overlaps_produces_disjoint_sets(self):
        merged_set_1 = [{1, 2, 3}, {4, 5}, {2, 5}, {4, 6}, {7}]
        resolved_set_1 = self.gm._resolve_overlaps(merged_set_1)

        self.logger.debug(f"Resolved Set 1: {resolved_set_1}")

        all_elems = set()
        for s in resolved_set_1:
            self.assertTrue(all_elems.isdisjoint(s), f"Overlap detected in Set 1 with subset {s}")
            all_elems |= s

        merged_set_2 = [{1, 2, 3}, {2, 3, 4}, {3, 4, 5}, {1, 5}, {2, 5}, {4, 6}, {5, 6}, {6, 7}, {1, 7}]
        resolved_set_2 = self.gm._resolve_overlaps(merged_set_2)

        self.logger.debug(f"Resolved Set 2: {resolved_set_2}")

        all_elems = set()
        for s in resolved_set_2:
            self.assertTrue(all_elems.isdisjoint(s), f"Overlap detected in Set 2 with subset {s}")
            all_elems |= s

    def test_generate_graph_empty_events(self):
        self.gm.generate_graph(1.0, 0, 0, 100, 200, 0.8, 0.1, 0.2, 5, 1, 0.9)
        self.assertIsInstance(self.gm.graph, GeneticGraph)
        self.assertTrue(self.gm.graph.contains_node("Start"))
        self.assertTrue(self.gm.graph.contains_node("End"))

    def test_parse_trace_token_game(self):
        log = {
            ("A", "B", "H"): 1,
            ("A", "C", "H"): 1,
            ("A", "D", "E", "F", "G", "H"): 1,
            ("A", "D", "F", "E", "G", "H"): 1,
        }
        gm = GeneticMining(log)

        individual = {
            "activities": {"A", "B", "C", "D", "E", "F", "G", "H"},
            "I": {
                "A": [set()],
                "B": [{"A"}],
                "C": [{"A"}],
                "D": [{"A"}],
                "E": [{"D"}],
                "F": [{"D"}],
                "G": [{"E"}, {"F"}],
                "H": [{"B", "C", "G"}],
            },
            "O": {
                "A": [{"B", "C", "D"}],
                "B": [{"H"}],
                "C": [{"H"}],
                "D": [{"E"}, {"F"}],
                "E": [{"G"}],
                "F": [{"G"}],
                "G": [{"H"}],
                "H": [set()],
            },
            "C": {("A", "B"), ("A", "C"), ("A", "D"),
                  ("D", "E"), ("D", "F"), ("E", "G"),
                  ("F", "G"), ("G", "H"), ("B", "H"), ("C", "H")}
        }
        gm.start_nodes = {"A"}
        gm.end_nodes = {"H"}

        traces = [
            ["A", "B", "H"],
            ["A", "C", "H"],
            ["A", "D", "E", "F", "G", "H"],
            ["A", "D", "F", "E", "G", "H"],
        ]

        for trace in traces:
            with self.subTest(trace=trace):
                parsed_count, is_completed = gm._parse_trace_token_game(individual, trace)

                self.assertEqual(parsed_count, len(trace), f"Trace {trace} was not fully parsed.")
                self.assertTrue(is_completed, f"Trace {trace} did not complete correctly (tokens stuck).")

    def test_parse_trace_token_game_deadlock(self):
        log = {
            ("A", "B", "H"): 1,
            ("A", "C", "H"): 1,
            ("A", "D", "E", "F", "G", "H"): 1,
            ("A", "D", "F", "E", "G", "H"): 1,
        }
        gm = GeneticMining(log)

        individual = {
            "activities": {"A", "B", "C", "D", "E", "F", "G", "H"},
            "I": {
                "A": [set()],
                "B": [{"A"}],
                "C": [{"A"}],
                "D": [{"A"}],
                "E": [{"D"}],
                "F": [{"D"}],
                "G": [{"F"}],
                "H": [{"B", "C", "G"}],
            },
            "O": {
                "A": [{"B", "C", "D"}],
                "B": [{"H"}],
                "C": [{"H"}],
                "D": [{"E"}, {"F"}],
                "E": [],
                "F": [{"G"}],
                "G": [{"H"}],
                "H": [set()],
            },
            "C": {("A", "B"), ("A", "C"), ("A", "D"),
                  ("D", "E"), ("D", "F"), ("E", "G"),
                  ("F", "G"), ("G", "H"), ("B", "H"), ("C", "H")}
        }
        gm.start_nodes = {"A"}
        gm.end_nodes = {"H"}

        traces = [
            ["A", "D", "E", "F", "G", "H"],
            ["A", "D", "F", "E", "G", "H"],
        ]

        for trace in traces:
            with self.subTest(trace=trace):
                parsed_count, is_completed = gm._parse_trace_token_game(individual, trace)
                self.assertEqual(parsed_count, len(trace) - 1, f"Trace {trace} was fully parsed.")
                self.assertFalse(is_completed, f"Trace {trace} did not complete correctly (tokens stuck).") # TODO: change error message

    def test_parse_trace_token_game_deadlock_set(self):
        log = {
            ("A", "B", "H"): 1,
            ("A", "C", "H"): 1,
            ("A", "D", "E", "F", "G", "H"): 1,
            ("A", "D", "F", "E", "G", "H"): 1,
        }
        gm = GeneticMining(log)

        individual = {
            "activities": {"A", "B", "C", "D", "E", "F", "G", "H"},
            "I": {
                "A": [set()],
                "B": [{"A"}],
                "C": [{"A"}],
                "D": [{"A"}],
                "E": [{"D"}],
                "F": [{"D"}],
                "G": [{"F"}],
                "H": [{"B", "C", "G"}],
            },
            "O": {
                "A": [{"B", "C", "D"}],
                "B": [{"H"}],
                "C": [{"H"}],
                "D": [{"E"}, {"F"}],
                "E": [set()],
                "F": [{"G"}],
                "G": [{"H"}],
                "H": [set()],
            },
            "C": {("A", "B"), ("A", "C"), ("A", "D"),
                  ("D", "E"), ("D", "F"), ("E", "G"),
                  ("F", "G"), ("G", "H"), ("B", "H"), ("C", "H")}
        }
        gm.start_nodes = {"A"}
        gm.end_nodes = {"H"}

        traces = [
            ["A", "D", "E", "F", "G", "H"],
            ["A", "D", "F", "E", "G", "H"],
        ]

        for trace in traces:
            with self.subTest(trace=trace):
                parsed_count, is_completed = gm._parse_trace_token_game(individual, trace)
                self.assertEqual(parsed_count, len(trace) - 1, f"Trace {trace} was fully parsed.")
                self.assertFalse(is_completed, f"Trace {trace} did not complete correctly (tokens stuck).") # TODO: change error message

    def test_parse_trace_token_game_filtered(self):
        individual = {
            "activities": {"A", "H"},
            "I": {"A": [set()], "H": [{"A"}]},
            "O": {"A": [{"H"}], "H": [set()]},
            "C": {("A", "H")}
        }
        self.gm.start_nodes = {"A"}
        self.gm.end_nodes = {"H"}

        trace = ["A", "H"]
        parsed_count, is_completed = self.gm._parse_trace_token_game(individual, trace)

        self.assertEqual(parsed_count, len(trace),
                         "filtered trace was not fully parsed.")

        self.assertTrue(is_completed,
                        "trace was incorrectly marked as not properly completed.")

    def test_parse_trace_token_game_L1L(self):
        individual = {
            "activities": {"A"},
            "I": {"A": [set(), {"A"}]},
            "O": {"A": [{"A"}, set()]},
            "C": {("A", "A")}
        }
        self.gm.start_nodes = {"A"}
        self.gm.end_nodes = {"A"}

        trace = ["A"]
        parsed_count, is_completed = self.gm._parse_trace_token_game(individual, trace)

        self.assertEqual(parsed_count, len(trace),
                         "L1L trace was not fully parsed (self-loop not handled correctly).")

        self.assertFalse(is_completed,
                         "L1L trace was incorrectly marked as properly completed.")

    def test_parse_trace_token_game_L1L_2(self):
        individual = {
            "activities": {"A", "B", "C"},
            "I": {"A": [set()], "B": [{"A", "B"}], "C": [{"B"}]},
            "O": {"A": [{"B"}], "B": [{"B", "C"}], "C": [set()]},
            "C": {("A", "B"), ("B", "B"), ("B", "C")}
        }
        self.gm.start_nodes = {"A"}
        self.gm.end_nodes = {"C"}

        trace = ["A", "B", "B", "C"]
        parsed_count, is_completed = self.gm._parse_trace_token_game(individual, trace)

        self.assertEqual(parsed_count, len(trace),
                         "L1L trace was not fully parsed (self-loop not handled correctly).")

        self.assertFalse(is_completed,
                         "L1L trace was incorrectly marked as properly completed.")

    def test_parse_trace_token_game_L2L(self):
        individual = {
            "activities": {"A", "B"},
            "I": {"A": [set(), {"B"}], "B": [{"A"}]},
            "O": {"A": [{"B"}], "B": [{"A"}, set()]},
            "C": {("A", "B"), ("B", "A")}
        }
        self.gm.start_nodes = {"A"}
        self.gm.end_nodes = {"B"}

        trace = ["A", "B"]
        parsed_count, is_completed = self.gm._parse_trace_token_game(individual, trace)

        self.assertNotEqual(parsed_count, len(trace),
                            "L2L trace was fully parsed (two-step loop not handled correctly).")

        self.assertFalse(is_completed,
                        # TODO: ACTUALLY NEEDS TO BE CHANGED TO .assertFalse, but only if completed workaround is solved!
                         "L2L trace was incorrectly marked as properly completed.")

    def test_and_or_semantics(self):
        # OR case: Z can fire if A OR B fired
        individual_or = {
            "activities": {"A", "B", "Z"},
            "I": {"A": [set()], "B": [set()], "Z": [{"A", "B"}]},
            "O": {"A": [{"Z"}], "B": [{"Z"}], "Z": [set()]},
            "C": {("A", "Z"), ("B", "Z")}
        }
        self.gm.start_nodes = {"A", "B"}
        self.gm.end_nodes = {"Z"}

        trace_or = ["A", "Z"]
        parsed_count_or, completed_or = self.gm._parse_trace_token_game(individual_or, trace_or)
        self.assertEqual(parsed_count_or, 2, "OR semantics failed: A should enable Z")
        self.assertTrue(completed_or)

        # AND case: Z requires both A and B
        individual_and = {
            "activities": {"A", "B", "Z"},
            "I": {"A": [set()], "B": [set()], "Z": [{"A"}, {"B"}]},
            "O": {"A": [{"Z"}], "B": [{"Z"}], "Z": [set()]},
            "C": {("A", "Z"), ("B", "Z")}
        }
        trace_and_valid = ["A", "B", "Z"]  # both provided
        parsed_count_and, completed_and = self.gm._parse_trace_token_game(individual_and, trace_and_valid)
        self.assertEqual(parsed_count_and, 3, "AND semantics failed: A and B should enable Z")
        self.assertTrue(completed_and)

        trace_and_invalid = ["A", "Z"]  # only A, not B
        parsed_count_and_inv, completed_and_inv = self.gm._parse_trace_token_game(individual_and, trace_and_invalid)
        self.assertLess(parsed_count_and_inv, 2, "AND semantics failed: Z should not fire without both inputs")
        self.assertFalse(completed_and_inv) # TODO: change error message

        # AND-of-ORs case: Z requires (A OR B) AND C
        individual_andor = {
            "activities": {"A", "B", "C", "Z"},
            "I": {"A": [set()], "B": [set()], "C": [set()], "Z": [{"A", "B"}, {"C"}]},
            "O": {"A": [{"Z"}], "B": [{"Z"}], "C": [{"Z"}], "Z": [set()]},
            "C": {("A", "Z"), ("B", "Z"), ("C", "Z")}
        }
        self.gm.start_nodes = {"A", "B", "C"}
        self.gm.end_nodes = {"Z"}

        trace_andor_valid = ["A", "C", "Z"]  # A satisfies first subset, C second
        parsed_count_andor, completed_andor = self.gm._parse_trace_token_game(individual_andor, trace_andor_valid)
        self.assertEqual(parsed_count_andor, 3, "AND-of-ORs semantics failed: (A or B) and C should enable Z")
        self.assertTrue(completed_andor)

        trace_andor_invalid = ["A", "Z"]  # missing C
        parsed_count_andor_inv, completed_andor_inv = self.gm._parse_trace_token_game(individual_andor,
                                                                                      trace_andor_invalid)
        self.assertLess(parsed_count_andor_inv, 2, "AND-of-ORs semantics failed: Z should not fire without C")
        self.assertFalse(completed_andor_inv) # TODO: change error message

        # Multi-OR: Z can fire if any of A, B, or C fired
        individual_multi_or = {
            "activities": {"A", "B", "C", "Z"},
            "I": {"A": [set()], "B": [set()], "C": [set()], "Z": [{"A", "B", "C"}]},
            "O": {"A": [{"Z"}], "B": [{"Z"}], "C": [{"Z"}], "Z": [set()]},
            "C": {("A", "Z"), ("B", "Z"), ("C", "Z")}
        }
        trace_multi_or = ["B", "Z"]
        parsed_count_multi_or, completed_multi_or = self.gm._parse_trace_token_game(individual_multi_or, trace_multi_or)
        self.assertEqual(parsed_count_multi_or, 2, "Multi-OR semantics failed: any of A,B,C should enable Z")
        self.assertTrue(completed_multi_or)

        # Multi-AND: Z requires A AND B AND C
        individual_multi_and = {
            "activities": {"A", "B", "C", "Z"},
            "I": {"A": [set()], "B": [set()], "C": [set()], "Z": [{"A"}, {"B"}, {"C"}]},
            "O": {"A": [{"Z"}], "B": [{"Z"}], "C": [{"Z"}], "Z": [set()]},
            "C": {("A", "Z"), ("B", "Z"), ("C", "Z")}
        }
        trace_multi_and_valid = ["A", "B", "C", "Z"]
        parsed_count_multi_and, completed_multi_and = self.gm._parse_trace_token_game(individual_multi_and,
                                                                                      trace_multi_and_valid)
        self.assertEqual(parsed_count_multi_and, 4, "Multi-AND semantics failed: all three inputs should enable Z")
        self.assertTrue(completed_multi_and)

        trace_multi_and_invalid = ["A", "B", "Z"]  # missing C
        parsed_count_multi_and_inv, completed_multi_and_inv = self.gm._parse_trace_token_game(individual_multi_and,
                                                                                              trace_multi_and_invalid)
        self.assertLess(parsed_count_multi_and_inv, 3, "Multi-AND semantics failed: Z should not fire without C")
        self.assertFalse(completed_multi_and_inv)

    def test_fitness_perfect_example_from_paper(self):
        log = {
            ("A", "B", "H"): 1,
            ("A", "C", "H"): 1,
            ("A", "D", "E", "F", "G", "H"): 1,
            ("A", "D", "F", "E", "G", "H"): 1,
        }
        gm = GeneticMining(log)

        individual = {
            "activities": {"A", "B", "C", "D", "E", "F", "G", "H"},
            "I": {
                "A": [set()],
                "B": [{"A"}],
                "C": [{"A"}],
                "D": [{"A"}],
                "E": [{"D"}],
                "F": [{"D"}],
                "G": [{"E"}, {"F"}],
                "H": [{"B", "C", "G"}],
            },
            "O": {
                "A": [{"B", "C", "D"}],
                "B": [{"H"}],
                "C": [{"H"}],
                "D": [{"E"}, {"F"}],
                "E": [{"G"}],
                "F": [{"G"}],
                "G": [{"H"}],
                "H": [set()],
            },
            "C": {("A", "B"), ("A", "C"), ("A", "D"),
                  ("D", "E"), ("D", "F"), ("E", "G"),
                  ("F", "G"), ("G", "H"), ("B", "H"), ("C", "H")}
        }

        fitness = gm._evaluate_fitness(individual)
        self.assertEqual(fitness, 1.0, msg=f"Expected perfect fitness=1.0 but got {fitness}")

def read(filename, timeLabel="Timestamp", caseLabel="Case ID", eventLabel="Activity"):
    dataframe_transformations = DataframeTransformations()
    dataframe_transformations.set_dataframe(pd.read_csv(filename))
    return dataframe_transformations.dataframe_to_cases_dict(
        timeLabel, caseLabel, eventLabel
    )


class TestGeneticMiningIntegration(unittest.TestCase):

    def test_csv_log_integration(self):
        log_dict = read("tests/testcsv/genetic_test.csv")

        gm = GeneticMining(log_dict)
        gm.generate_graph(0.2, 0, 0, 200, 100, 0.8, 0.2, 0.1, 2, 1, 0.95)

        self.__check_graph_integrity(gm.graph)

    def test_txt_log_integration(self):
        log = self.__read_cases("tests/testlogs/genetic_test.txt")
        log_dict = cases_list_to_dict(log)

        gm = GeneticMining(log_dict)
        gm.generate_graph(0.5, 1, 4, 200, 100, 0.8, 0.2, 0.1, 2, 1, 0.95)

        self.__check_graph_integrity(gm.graph)

    @staticmethod
    def __read_cases(filename):
        log = []
        with open(filename, "r") as f:
            for line in f.readlines():
                assert isinstance(line, str)
                log.append(list(line.split()))
        return log

    def __check_graph_integrity(self, graph: BaseGraph):
        self.assertTrue(graph.contains_node("Start"), "Start node not found.")
        self.assertTrue(graph.contains_node("End"), "End node not found.")

        # Check there is a 'start' node that connects to at least 1 other node
        self.assertTrue(
            len(list(filter(lambda edge: edge.source == "Start", graph.get_edges())))
            >= 1,
            "Start node does not connect to any other nodes.",
        )

        # Check if all nodes are reachable from the 'start' node
        reachable_nodes = set()
        queue = deque(["Start"])
        while queue:
            node = queue.popleft()
            reachable_nodes.add(node)
            for edge in graph.get_edges():
                if edge.source == node:
                    if edge.destination not in reachable_nodes:
                        queue.append(edge.destination)

        # Check if End is reachable
        self.assertIn("End", reachable_nodes, "End node not reachable from Start.")


if __name__ == "__main__":
    unittest.main()
