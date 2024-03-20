from graphviz import Digraph
import numpy as np
from mining_algorithms.ddcal_clustering import DensityDistributionClusterAlgorithm

class AlphaMining():
    def __init__(self, cases):
        # convert it to set, since alpha miner doesn't care how many times a process was done
        self.cases = cases
        self.start_nodes = self.get_start_nodes()
        self.end_nodes = self.get_end_nodes()
        self.unique_events = self.get_unique_events()
        print("sad")

    # step 1
    # each activity in cases corresponds to a transition in sigma(cases)
    def get_unique_events(self):
        unique_events = []

        for case in self.cases:
            for event in case:
                unique_events.append(event)

        return set(unique_events)

    # step 2
    # the set of start activities - that is, the first element of each trace
    def get_start_nodes(self):
        start_nodes = []

        for case in self.cases:
            start_nodes.append(case[0])

        return set(start_nodes)

    # step 3
    # the set of end activities - that is, elements that appear last in trace
    def get_end_nodes(self):
        end_nodes = []

        for case in self.cases:
            end_nodes.append(case[len(case)-1])

        return set(end_nodes)


