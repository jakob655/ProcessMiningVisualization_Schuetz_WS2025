"""
This unittest tests the integrity of the heuristic_graph_controller which holds the HeuristicMining Model.
"""

import unittest
from collections import deque

import pandas as pd

from graphs.visualization.base_graph import BaseGraph
from mining_algorithms.heuristic_mining import HeuristicMining
from transformations.dataframe_transformations import DataframeTransformations
from transformations.utils import cases_list_to_dict


def read(filename, timeLabel="timestamp", caseLabel="case", eventLabel="event"):
    dataframe_transformations = DataframeTransformations()
    dataframe_transformations.set_dataframe(pd.read_csv(filename))
    return dataframe_transformations.dataframe_to_cases_dict(
        timeLabel, caseLabel, eventLabel
    )


class TestHeuristic(unittest.TestCase):

    def test_create_dependency_graph_using_preprocessed_txt(self):
        print("-------------- Running test.txt ----------------")
        self.__run_test_txt("test0")
        print("passed test 0")
        self.__run_test_txt("test1")
        print("passed test 1")
        self.__run_test_txt("test2")
        print("passed test 2")
        self.__run_test_txt("test3")
        print("passed test 3")
        print("---------------- test.txt passed! ----------------")

    def test_create_dependency_graph_using_test_csv(self):
        print("-------------- Running test_csv ----------------")
        self.__run_test_csv(0.5, 1, 0.5, 0.0, 0.0)
        print("passed test 1")
        self.__run_test_csv(0.1, 1, 0.5, 0.2, 0.5)
        print("passed test 2")
        self.__run_test_csv(0.9, 1, 0.5, 0.5, 0.2)
        print("passed test 3")
        self.__run_test_csv(0.5, 10, 0.5, 1, 1)
        print("passed test 4")
        print("---------------- test_csv passed! ----------------")

    def test_create_dependency_graph_using_CallcenterExample(self):
        print("-------------- Running large CallcenterExample ----------------")
        self.__run_CallcenterExample_csv(0.5, 1, 0.5, 0.2)
        print("---------------- CallcenterExample passed! ----------------")

    # read test cases that are txt files for testing
    def __read_cases(self, filename):
        log = []
        # cwd = os.getcwd()
        # path = os.path.join(cwd, filename)
        with open(filename, "r") as f:
            for line in f.readlines():
                assert isinstance(line, str)
                log.append(list(line.split()))
        return log

    def __run_test_txt(self, filename):
        heuristicMining = HeuristicMining(
            cases_list_to_dict(self.__read_cases("tests/testlogs/" + filename + ".txt"))
        )
        heuristicMining.generate_graph(0.5, 1, 40, 0.5, 20, 0.5)
        self.__check_graph_integrity(heuristicMining.get_graph())

    def __run_test_csv(self, spm_threshold, node_freq_threshold, edge_freq_threshold, dependency_threshold,
                       min_frequency):
        heuristicMining = HeuristicMining(read("tests/testcsv/test_csv.csv"))
        heuristicMining.generate_graph(spm_threshold, node_freq_threshold, 40, edge_freq_threshold, 20,
                                       dependency_threshold)
        self.__check_graph_integrity(heuristicMining.get_graph())

    def __run_CallcenterExample_csv(self, spm_threshold, node_freq_threshold, edge_freq_threshold,
                                    dependency_threshold):
        heuristicMining = HeuristicMining(
            read(
                "tests/testcsv/CallcenterExample.csv",
                caseLabel="Service ID",
                eventLabel="Operation",
                timeLabel="Start Date",
            )
        )
        heuristicMining.generate_graph(spm_threshold, node_freq_threshold, 40, edge_freq_threshold, 20,
                                       dependency_threshold)
        self.__check_graph_integrity(heuristicMining.get_graph())

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

        self.assertEqual(
            reachable_nodes,
            set(map(lambda node: node.id, graph.get_nodes())),
            "Not all nodes are reachable from the 'start' node.",
        )
