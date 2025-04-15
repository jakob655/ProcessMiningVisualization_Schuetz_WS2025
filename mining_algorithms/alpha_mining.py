import itertools
from logger import get_logger
from graphs.visualization.alpha_graph import AlphaGraph
from mining_algorithms.base_mining import BaseMining


class AlphaMining(BaseMining):
    def __init__(self, log):
        super().__init__(log)
        self.logger = get_logger("AlphaMining")

        self.__calculate_model_state()

    # This implementation follows the steps outlined in the lecture by Professor Wil van der Aalst on process mining.
    # The lecture video can be found at: https://www.youtube.com/watch?v=ATBEEEDxHTQ
    # Credit for the algorithm and methodology goes to Professor Wil van der Aalst.

    ####################################################################################################################
    # ALPHA MINER ALGORITHM IMPLEMENTATION BEGIN

    # step 1
    # each activity in cases corresponds to a transition in sigma(cases)
    # returns list converted to set to avoid duplicates
    # -> moved to BaseMining

    # step 2
    # the set of start activities - that is, the first element of each trace
    # returns list converted to set to avoid duplicates
    # -> moved to BaseMining

    # step 3
    # the set of end activities - that is, elements that appear last in trace
    # returns list converted to set to avoid duplicates
    # -> moved to BaseMining

    # step 4
    # Find pairs (A,B) of sets of activities such that every element a∈B and every element b∈B are causally related,
    # (i.e., a->L b), all  elements in A are independent (a1#La2), and all elements in B are independent (b1#Lb2)
    # returns set

    def generate_set_xl(self, unique_events, choice, causality):
        xl_set = []

        subsets = itertools.chain.from_iterable(
            itertools.combinations(unique_events, r) for r in range(1, len(unique_events) + 1))
        subsets_in_choice = [_set for _set in subsets if self.__is_set_in_choice(_set, choice)]
        for a, b in itertools.product(subsets_in_choice, subsets_in_choice):
            if self.__is_set_in_causality((a, b), causality):
                xl_set.append((a, b))

        return set(xl_set)

    # step 5
    # Delete from set XL all pairs (A,B) that are not maximal
    # returns set

    def generate_set_yl(self, xl_set, parallel):
        # create a yl set the superior of xl_set, containing the maximum
        yl_set = xl_set
        s_all = itertools.combinations(yl_set, 2)

        # generate maximum set
        for pair in s_all:
            if self.__is_subset(pair[0], pair[1]):
                yl_set.discard(pair[0])
            elif self.__is_subset(pair[1], pair[0]):
                yl_set.discard(pair[1])

        # remove self-loops
        # e.g. if (a,b),(b,c) in YL, and (b,b) in Parallel, then we need to remove (a,b),(b,c)
        # (a,b) is equal to (a,bb), also b||b, thus a and bb cannot make a pair, only "#" relations can.
        self_loop = set()
        for pair in parallel:
            if pair == pair[::-1]:  # if we found pairs like (b,b), add b into self-loop sets
                self_loop.add(pair[0])

        # define a set of to be deleted sets and remove them from yl_set
        to_be_deleted = set()
        for pair in yl_set:
            if self.__contains(pair, self_loop):
                to_be_deleted.add(pair)
        for pair in to_be_deleted:
            yl_set.discard(pair)
        return yl_set

    # Step 6
    def draw_graph(self, spm_threshold):
        self.graph = AlphaGraph()

        # Add the start node
        self.graph.add_start_node()

        # Add the end node
        self.graph.add_end_node()

        self.spm_threshold = spm_threshold
        self.spm_filtered_events = self.get_spm_filtered_events()
        self.spm_filtered_log = self.get_spm_filtered_log()


        if not self.spm_filtered_events:
            # only show start and end if nothing to visualize
            return

        # Add empty circles for start and end
        self.graph.add_empty_circle("empty_circle_start")
        self.graph.add_empty_circle("empty_circle_end")

        self.__calculate_model_state()

        # Add nodes (events to draw combined with start and end nodes)
        nodes_to_draw = self.__events_to_draw().union(self.start_nodes, self.end_nodes)

        # Add nodes (either all filtered or only relevant via yl_set)
        for node in nodes_to_draw:
            self.graph.add_node(str(node))

        # Connect the start node to the empty circle for start nodes
        self.graph.create_edge("Start", "empty_circle_start", penwidth=0.1)

        # Add and connect nodes based on yl_set logic
        for _set in self.yl_set:
            if len(_set) == 2:
                A, B = _set[0], _set[1]
                circle_id = f"{A}_{B}"

                # Case 1: Single-to-Single
                if len(A) == 1 and len(B) == 1:
                    self.graph.add_empty_circle(circle_id)
                    self.graph.create_edge(str(A[0]), circle_id, penwidth=0.1)
                    self.graph.create_edge(circle_id, str(B[0]), penwidth=0.1)

                # Case 2: Single-to-Multiple
                elif len(A) == 1 and len(B) == 2:
                    if self.__is_set_in_choice(B, self.choice_set):
                        # XOR-split
                        self.graph.add_empty_circle(circle_id)
                        self.graph.create_edge(str(A[0]), circle_id, penwidth=0.1)
                        for event in B:
                            self.graph.create_edge(circle_id, str(event), penwidth=0.1)
                    elif self.__is_set_in_parallel(B, self.parallel_set):
                        # AND-split (two separate circles)
                        circle_id1 = f"{A}_{B}_1"
                        circle_id2 = f"{B}_{A}_2"
                        self.graph.add_empty_circle(circle_id1)
                        self.graph.add_empty_circle(circle_id2)
                        self.graph.create_edge(str(A[0]), circle_id1, penwidth=0.1)
                        self.graph.create_edge(str(A[0]), circle_id2, penwidth=0.1)
                        self.graph.create_edge(circle_id1, str(B[0]), penwidth=0.1)
                        self.graph.create_edge(circle_id2, str(B[1]), penwidth=0.1)

                # Case 3: Multiple-to-Single
                elif len(A) == 2 and len(B) == 1:
                    if self.__is_set_in_choice(A, self.choice_set):
                        # XOR-join
                        self.graph.add_empty_circle(circle_id)
                        for event in A:
                            self.graph.create_edge(str(event), circle_id, penwidth=0.1)
                        self.graph.create_edge(circle_id, str(B[0]), penwidth=0.1)
                    elif self.__is_set_in_parallel(A, self.parallel_set):
                        # AND-join (two separate circles)
                        circle_id1 = f"{A}_{B}_1"
                        circle_id2 = f"{B}_{A}_2"
                        self.graph.add_empty_circle(circle_id1)
                        self.graph.add_empty_circle(circle_id2)
                        self.graph.create_edge(str(A[0]), circle_id1, penwidth=0.1)
                        self.graph.create_edge(str(A[1]), circle_id2, penwidth=0.1)
                        self.graph.create_edge(circle_id1, str(B[0]), penwidth=0.1)
                        self.graph.create_edge(circle_id2, str(B[0]), penwidth=0.1)

        # Connect the empty circle to the actual start and end nodes
        for node in self.start_nodes.intersection(self.spm_filtered_events):
            self.graph.create_edge("empty_circle_start", str(node), penwidth=0.1)

        for node in self.end_nodes.intersection(self.spm_filtered_events):
            self.graph.create_edge(str(node), "empty_circle_end", penwidth=0.1)

        # Connect the empty circle to the end node
        self.graph.create_edge("empty_circle_end", "End", penwidth=0.1)

    # ALPHA MINER ALGORITHM IMPLEMENTATION END
    ####################################################################################################################

    ####################################################################################################################
    # ALPHA MINER ALGORITHM ESSENTIALS BEGIN

    # essential for alpha algorithm: finding direct succession, together with causality, parallel and choice
    # noted with >, for example a > b, b > c, c > e in a process ['a', 'b', 'c', 'e']
    # returns list converted to set to avoid duplicates
    def _calculate_direct_succession(self):
        direct_succession = []
        for case in self.spm_filtered_log:
            for i in range(len(case) - 1):
                x = case[i]
                y = case[i + 1]
                direct_succession.append((x, y))
        return set(direct_succession)

    # essential for alpha algorithm: finding causality, together with direct succession, parallel and choice
    # noted with ->, for example a -> b, b -> c, but not b -> b in a process ['a', 'b', 'b', 'c']
    # returns list converted to set to avoid duplicates
    @staticmethod
    def _calculate_causality(direct_succession):
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
    def _calculate_parallel(direct_succession):
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
    def _calculate_choice(unique_events, causality, parallel):
        choice = []
        for event1 in unique_events:
            for event2 in unique_events:
                if (event1 != event2) and ((event1, event2) not in causality) and (
                        (event2, event1) not in causality) and ((event1, event2) not in parallel):
                    choice.append((event1, event2))
        return set(choice)

    def generate_footprint(self):
        footprint = ["All transitions: {}".format(self.events),
                     "Direct succession: {}".format(self.direct_succession_set), "Causality: {}".format(self.causality_set),
                     "Parallel: {}".format(self.parallel_set), "Choice: {}".format(self.choice_set)]
        return '\n'.join(footprint)

    # ALPHA MINER ALGORITHM ESSENTIALS END
    ####################################################################################################################

    ####################################################################################################################
    # ALPHA MINER ALGORITHM HELPER METHODS BEGIN

    # searches for a given set if the given set is found in choice set
    # returns boolean
    @staticmethod
    def __is_set_in_choice(_set, choice):
        if len(_set) == 1:
            return True
        else:
            for i in range(len(_set)):
                for j in range(i + 1, len(_set)):
                    if (_set[i], _set[j]) not in choice:
                        return False
            return True

    # searches for a given set if the given set is found in causality set
    # returns boolean
    @staticmethod
    def __is_set_in_causality(_set, causality):
        a, b = _set[0], _set[1]
        all_possibilities = itertools.product(a, b)
        for pair in all_possibilities:
            if pair not in causality:
                return False
        return True

    # searches for a given set if the given set is found in parallel set
    # returns boolean
    @staticmethod
    def __is_set_in_parallel(_set, parallel):
        a, b = _set[0], _set[1]
        all_possibilities = itertools.product(a, b)
        for pair in all_possibilities:
            if pair not in parallel:
                return False
        return True

    # check if the first set in 'a' is a subset of the first set in 'b'
    # and if the second set in 'a' is a subset of the second set in 'b'
    # Return True if both conditions are True, otherwise return False
    @staticmethod
    def __is_subset(a, b):
        first_subset = set(a[0]).issubset(b[0])
        second_subset = set(a[1]).issubset(b[1])
        return first_subset and second_subset

    # check if a set in b equals to a set in a
    @staticmethod
    def __contains(a, b):
        for i in a:
            for j in b:
                if j == i[0]:
                    return True
        return False

    # defines unique events that are not self loop which are needed to be drawn
    def __events_to_draw(self):
        events_to_draw = []
        for _set in self.yl_set:
            for event in _set:
                for node in event:
                    events_to_draw.append(node)
        return set(events_to_draw)

    def __calculate_model_state(self):
        self.direct_succession_set = self._calculate_direct_succession()
        self.causality_set = self._calculate_causality(self.direct_succession_set)
        self.parallel_set = self._calculate_parallel(self.direct_succession_set)
        self.choice_set = self._calculate_choice(self.events, self.causality_set, self.parallel_set)
        self.xl_set = self.generate_set_xl(self.events, self.choice_set, self.causality_set)
        self.yl_set = self.generate_set_yl(self.xl_set, self.parallel_set)
        self.start_nodes = self._get_start_nodes()
        self.end_nodes = self._get_end_nodes()
        self.footprint = self.generate_footprint()

    # ALPHA MINER ALGORITHM HELPER METHODS END
    ####################################################################################################################