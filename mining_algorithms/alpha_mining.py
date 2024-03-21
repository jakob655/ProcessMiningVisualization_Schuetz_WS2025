import itertools

from graphviz import Digraph
import numpy as np
from mining_algorithms.ddcal_clustering import DensityDistributionClusterAlgorithm


class AlphaMining:
    def __init__(self, cases):
        self.cases = cases
        self.start_events = self.get_start_events()
        self.end_events = self.get_end_events()
        self.unique_events = self.get_unique_events()
        self.direct_succession = self.direct_succession()
        self.causality = self.causality(self.direct_succession)
        self.parallel = self.parallel(self.direct_succession)
        self.choice = self.choice(self.unique_events, self.causality, self.parallel)
        self.footprint = self.footprint()
        print(self.footprint)

    # This implementation follows the steps outlined in the lecture by Professor Wil van der Aalst on process mining.
    # The lecture video can be found at: https://www.youtube.com/watch?v=ATBEEEDxHTQ
    # Credit for the algorithm and methodology goes to Professor Wil van der Aalst.

    ####################################################################################################################
    # ALPHA MINER ALGORITHM IMPLEMENTATION BEGIN

    # step 1
    # each activity in cases corresponds to a transition in sigma(cases)
    # returns list converted to set to avoid duplicates
    def get_unique_events(self):
        unique_events = []

        for case in self.cases:
            for event in case:
                unique_events.append(event)

        return set(unique_events)

    # step 2
    # the set of start activities - that is, the first element of each trace
    # returns list converted to set to avoid duplicates
    def get_start_events(self):
        start_events = []

        for case in self.cases:
            start_events.append(case[0])

        return set(start_events)

    # step 3
    # the set of end activities - that is, elements that appear last in trace
    # returns list converted to set to avoid duplicates
    def get_end_events(self):
        end_events = []

        for case in self.cases:
            end_events.append(case[len(case)-1])

        return set(end_events)

    # TODO - STEP 4 XL from lecture BPI 5 - Petri Nets & Alpha Algorithm - Professor Wil van der Aalst
    # step 4
    # Find pairs (A,B) of sets of activities such that every element a∈B and every element b∈B are causally related,
    # (i.e., a->L b), all  elements in A are independent (a1#La2), and all elements in B are independent (b1#Lb2)

    def generate_set_xl(self):
        pass

    # TODO - STEP 5 YL from lecture BPI 5 - Petri Nets & Alpha Algorithm - Professor Wil van der Aalst
    # step 5
    # Delete from set XL all pairs (A,B) that are not maximal

    def generate_set_yl(self):
        pass

    # ALPHA MINER ALGORITHM IMPLEMENTATION END
    ####################################################################################################################

    ####################################################################################################################
    # ALPHA MINER ALGORITHM ESSENTIALS BEGIN

    # essential for alpha algorithm: finding direct succession, together with causality, parallel and choice
    # noted with >, for example a > b, b > c, c > e in a process ['a', 'b', 'c', 'e']
    # returns list converted to set to avoid duplicates
    def direct_succession(self):

        direct_succession = []
        for case in self.cases:
            for i in range(len(case) - 1):
                x = case[i]
                y = case[i + 1]
                direct_succession.append((x, y))

        return set(direct_succession)

    # essential for alpha algorithm: finding causality, together with direct succession, parallel and choice
    # noted with ->, for example a -> b, b -> c, but not b -> b in a process ['a', 'b', 'b', 'c']
    # returns list converted to set to avoid duplicates
    @staticmethod
    def causality(direct_succession):

        causality = []
        for pair in direct_succession:
            pair_reversed = (pair[1], pair[0])
            if pair_reversed not in direct_succession:
                pair_not_reversed = (pair[0], pair[1])
                causality.append(pair_not_reversed)

        return set(causality)

    # essential for alpha algorithm: finding parallels, together with direct succession, causality and choice
    # noted with ||, for example b || b in a process ['a', 'b', 'b', 'c']
    # returns list converted to set to avoid duplicates
    @staticmethod
    def parallel(direct_succession):

        parallel = []
        for pair in direct_succession:
            pair_reversed = (pair[1], pair[0])
            if pair_reversed in direct_succession:
                pair_not_reversed = (pair[0], pair[1])
                parallel.append(pair_not_reversed)

        return set(parallel)

    # essential for alpha algorithm: finding choice, together with direct succession, causality and parallel
    # noted with #, for example 'a # c', 'c # a' in a process ['a', 'b', 'b', 'c']
    # returns list converted to set to avoid duplicates
    @staticmethod
    def choice(unique_events, causality, parallel):

        choice = []
        for event1 in unique_events:
            for event2 in unique_events:
                if (event1 != event2) and ((event1, event2) not in causality) and (
                        (event2, event1) not in causality) and ((event1, event2) not in parallel):
                    choice.append((event1, event2))

        return set(choice)

    def footprint(self):
        footprint = ["All transitions: {}".format(self.unique_events),
                     "Direct succession: {}".format(self.direct_succession), "Causality: {}".format(self.causality),
                     "Parallel: {}".format(self.parallel), "Choice: {}".format(self.choice)]
        return '\n'.join(footprint)

    # ALPHA MINER ALGORITHM ESSENTIALS END
    ####################################################################################################################

