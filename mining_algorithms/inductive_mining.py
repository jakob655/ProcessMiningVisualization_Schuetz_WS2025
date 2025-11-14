import itertools
from components.petri_net import PetriNetToolkit, add_petri_net_to_graph
from graphs.visualization.genetic_graph import GeneticGraph
from graphs.cuts import exclusive_cut, parallel_cut, sequence_cut, loop_cut
from graphs.dfg import DFG
from graphs.visualization.inductive_graph import InductiveGraph
from logger import get_logger
from logs.filters import filter_traces
from logs.splits import (
    exclusive_split,
    parallel_split,
    sequence_split,
    loop_split,
)
from mining_algorithms.base_mining import BaseMining


class InductiveMining(BaseMining):
    """A class to generate a graph from a log using the Inductive Mining algorithm."""

    def __init__(self, log):
        """Constructor for the InductiveMining class.

        Parameters
        ----------
        log : dict[tuple[str, ...], int]
            A dictionary containing the traces and their frequencies in the log.
        """
        super().__init__(log)
        self.logger = get_logger("InductiveMining")
        self.traces_threshold = 0.2
        self.filtered_log = None
        self.use_petri_net = False
        self.petri_toolkit = PetriNetToolkit()
        self.petri_net = None

    def generate_graph(self, spm_threshold: float, node_freq_threshold_normalized: float,
                       node_freq_threshold_absolute: int, traces_threshold: float, use_petri_net: bool = False):
        """Generate a graph from the log using the Inductive Mining algorithm.

        Parameters
        ----------
        spm_threshold : float
            The threshold for the SPM (Search Process Model) score. Events with an SPM score below this value
            will be filtered out before generating the graph.
        node_freq_threshold_normalized : float
            The threshold for the normalized frequency of nodes (events).
        node_freq_threshold_absolute: : int
            The threshold for the absolute frequency of nodes (events).
        traces_threshold : float
            The traces threshold for the filtering of the log.
            All traces with a frequency lower than the threshold * max_trace_frequency will be removed.
        use_petri_net : bool, optional
            If True, renders a Petri net representation instead of the process tree view.
        """
        self.traces_threshold = traces_threshold
        self.spm_threshold = spm_threshold
        self.node_freq_threshold_normalized = node_freq_threshold_normalized
        self.node_freq_threshold_absolute = node_freq_threshold_absolute
        self.use_petri_net = use_petri_net


        self.recalculate_model_filters()

        if not self.filtered_events:
            if use_petri_net:
                self.graph = GeneticGraph()
                self.graph.add_start_node()
                self.graph.add_end_node()
                self.graph.create_edge("Start", "End")
            else:
                self.graph = InductiveGraph(self.filtered_events)
                self.graph.add_edge("Start", "End", None)
            return

        self.node_sizes = {k: self.calculate_node_size(k) for k in self.filtered_events}

        events_to_remove = self.get_events_to_remove(node_freq_threshold_normalized)

        self.logger.debug(f"Events to remove: {events_to_remove}")
        min_traces_frequency = self.calculate_minimum_traces_frequency(traces_threshold)

        filtered_log = filter_traces(self.node_frequency_filtered_log, min_traces_frequency)

        self.filtered_log = filtered_log

        self.logger.info("Start Inductive Mining")
        process_tree = self.inductive_mining(self.filtered_log)
        node_stats_map = {stat["node"]: stat for stat in self.get_node_statistics()}
        
        if use_petri_net:
            self._render_inductive_petri_net(process_tree, node_stats_map)
            return
        
        self.graph = InductiveGraph(
            process_tree,
            frequency=self.filtered_appearance_freqs,
            node_sizes=self.node_sizes,
            node_stats_map=node_stats_map,
        )
        
    def _render_inductive_petri_net(self, process_tree, node_stats_map):
        # Build Petri net from inductive process tree
        petri_net = self._build_inductive_petri_net(process_tree)
        self.petri_net = petri_net
        self.graph = GeneticGraph()
        self.graph.add_start_node()
        self.graph.add_end_node()
        add_petri_net_to_graph(
            self.graph,
            petri_net,
            self.filtered_events,
            node_stats_map,
            self.filtered_appearance_freqs,
            logger=self.logger,
        )

    def _build_inductive_petri_net(self, process_tree):
        toolkit = self.petri_toolkit
        net, start_place, end_place = toolkit.create_base_net()
        place_counter = itertools.count()
        tau_counter = itertools.count()
        
        def new_place(prefix):
            place_id = f"p_im_{prefix}_{next(place_counter)}"
            toolkit.register_place(net, place_id)
            return place_id

        def new_tau(prefix):
            return f"tau_im_{prefix}_{next(tau_counter)}"

        def connect_places(source_place, target_place, prefix):
            # Connect two places by inserting a silent between
            tau_id = new_tau(prefix)
            toolkit.register_transition(net, tau_id, visible=False)
            toolkit.add_arc(net, source_place, tau_id)
            toolkit.add_arc(net, tau_id, target_place)

        def register_visible_transition(label):
            base = str(label)
            trans_id = base
            suffix = 1
            while trans_id in net['transitions']:  # If multiple transitions have the same label -> append incrementing suffix
                suffix += 1
                trans_id = f"{base}__{suffix}"
            toolkit.register_transition(net, trans_id, visible=True, label=base)
            return trans_id

        def build_fragment(tree):
            if isinstance(tree, (str, int)):
                label = str(tree)
                # Create entry and exit places for the leaf
                entry = new_place("leaf_in")
                exit_place = new_place("leaf_out")
                if label == 'tau':
                    trans_id = new_tau("silent")
                    toolkit.register_transition(net, trans_id, visible=False)
                else:
                    trans_id = register_visible_transition(label)
                toolkit.add_arc(net, entry, trans_id)
                toolkit.add_arc(net, trans_id, exit_place)
                return entry, exit_place

            if not isinstance(tree, tuple) or not tree:
                raise ValueError('Invalid process tree node')

            op = tree[0]
            children = tree[1:]

            if op == 'seq':
                entry, current_exit = build_fragment(children[0])
                for child in children[1:]:
                    child_entry, child_exit = build_fragment(child)
                    connect_places(current_exit, child_entry, 'seq')
                    current_exit = child_exit
                return entry, current_exit

            if op == 'xor':
                entry = new_place('xor_in')
                exit_place = new_place('xor_out')
                
                # Each branch gets its own fragment, connected with silent transitions
                for child in children:
                    child_entry, child_exit = build_fragment(child)
                    connect_places(entry, child_entry, 'xor_in')
                    connect_places(child_exit, exit_place, 'xor_out')
                return entry, exit_place

            if op == 'par':   # Parallel (AND)
                entry = new_place('par_in')
                exit_place = new_place('par_out')
                # Explicit split and join silent transitions.
                split_id = new_tau('par_split')
                join_id = new_tau('par_join')
                
                toolkit.register_transition(net, split_id, visible=False)
                toolkit.register_transition(net, join_id, visible=False)
                toolkit.add_arc(net, entry, split_id)
                toolkit.add_arc(net, join_id, exit_place)
                
                # Each parallel branch connects to split and join
                for child in children:
                    child_entry, child_exit = build_fragment(child)
                    toolkit.add_arc(net, split_id, child_entry)
                    toolkit.add_arc(net, child_exit, join_id)
                return entry, exit_place

            if op == 'loop':
                entry = new_place('loop_in')
                exit_place = new_place('loop_out')
                
                # First child is loop body
                body_entry, body_exit = build_fragment(children[0])
                
                connect_places(entry, body_entry, 'loop_entry')
                connect_places(body_exit, exit_place, 'loop_exit')
                
                 # Remaining childs are redo/return branches
                for redo in children[1:]:
                    redo_entry, redo_exit = build_fragment(redo)
                    connect_places(body_exit, redo_entry, 'loop_redo_in')
                    connect_places(redo_exit, body_entry, 'loop_redo_back')
                return entry, exit_place

            raise ValueError(f'Unsupported process tree operator: {op}')

        # Build root fragment and connect it to global start and end places
        if process_tree:
            entry, exit_place = build_fragment(process_tree)
            connect_places(start_place, entry, 'root_start')
            connect_places(exit_place, end_place, 'root_end')
        else:
        # Empty model → direct connection between start and end
            toolkit.add_arc(net, start_place, end_place)

        toolkit.finalize_net(net)
        return net


    def inductive_mining(self, log):
        """Generate a process tree from the log using the Inductive Mining algorithm.
        This is a recursive function that generates the process tree from the log,
        by splitting the log into partitions and generating the tree for each partition.
        This function uses the base cases, the cut methods and the fallthrough method to generate the tree.
        If the log is a base case, the corresponding tree is returned. Otherwise, the log is split into partitions using the cut methods.
        If a cut is found, the tree is generated for each partition. If no cut is found, the fallthrough method is used to generate the tree.

        Parameters
        ----------
        log : dict[tuple[str, ...], int]
            A dictionary containing the traces and their

        Returns
        -------
        tuple
            A tuple representing the process tree. The first element is the operation of the node, the following elements are the children of the node.
            The children are either strings representing the events or tuples representing a subtree.
        """
        if tree := self.base_cases(log):
            self.logger.debug(f"Base case: {tree}")
            return tree

        if tuple() not in log:
            if partitions := self.calculate_cut(log):
                self.logger.debug(f"Cut: {partitions}")
                operation = partitions[0]
                return (operation, *list(map(self.inductive_mining, partitions[1:])))

        return self.fallthrough(log)

    def base_cases(self, log) -> str | None:
        """Check if the log is a base case and return the corresponding tree.
        The base cases are:
        - an empty log
        - a log with a single event

        Parameters
        ----------
        log : dict[tuple[str, ...], int]
            A dictionary containing the traces and their frequencies in the log.

        Returns
        -------
        str | None
            The event in the log if it is a base case, otherwise None.
        """
        if len(log) > 1:
            return None

        if len(log) == 1:
            trace = list(log.keys())[0]
            if len(trace) == 0:
                return "tau"
            if len(trace) == 1:
                return trace[0]

        return None

    def calculate_cut(self, log) -> tuple | None:
        """Find a partitioning of the log using the different cut methods.
        The cut methods are:
        - exclusive_cut
        - sequence_cut
        - parallel_cut
        - loop_cut

        Parameters
        ----------
        log : dict[tuple[str, ...], int]
            A dictionary containing the traces and their frequencies in the log.

        Returns
        -------
        tuple | None
            A process tree representing the partitioning of the log if a cut was found, otherwise None.
        """
        dfg = DFG(log)

        if partitions := exclusive_cut(dfg):
            return ("xor", *exclusive_split(log, partitions))
        elif partitions := sequence_cut(dfg):
            return ("seq", *sequence_split(log, partitions))
        elif partitions := parallel_cut(dfg):
            return ("par", *parallel_split(log, partitions))
        elif partitions := loop_cut(dfg):
            return ("loop", *loop_split(log, partitions))

        return None

    def fallthrough(self, log):
        """Generate a process tree for the log using a fallthrough method.
        The following fallthrough method is used:
        - if there is a empty trace in the log, make an xor split with tau and the inductive mining of the log without the empty trace
        - if there is a single event in the log and it occures more than once in a trace, make a loop split with the event and tau
        - if there are multiple events in the log, return a flower model with all the events

        Parameters
        ----------
        log : dict[tuple[str, ...], int]
            A dictionary containing the traces and their frequencies in the log.

        Returns
        -------
        tuple
            A tuple representing the process tree. The first element is the operation of the node, the following elements are the children of the node.
        """
        log_alphabet = self.get_log_alphabet(log)

        # if there is a empty trace in the log
        # make an xor split with tau and the inductive mining of the log without the empty trace
        if tuple() in log:
            empty_log = {tuple(): log[tuple()]}
            del log[tuple()]
            return ("xor", self.inductive_mining(empty_log), self.inductive_mining(log))

        # if there is a single event in the log
        # and it occures more than once in a trace
        # make a loop split with the event and tau
        # the event has to occure more than once in a trace,
        #  otherwise it would be a base case
        if len(log_alphabet) == 1:
            return ("loop", list(log_alphabet)[0], "tau")

        # if there are multiple events in the log
        # return a flower model with all the events
        return ("loop", "tau", *log_alphabet)

    def get_log_alphabet(self, log) -> set[str]:
        """Get the alphabet of the log. The alphabet is the set of all unique events in the log.

        Parameters
        ----------
        log : dict[tuple[str, ...], int]
            A dictionary containing the traces and their frequencies in the log.

        Returns
        -------
        set[str]
            A set containing all unique events in the log.
        """
        return set(event for case in log for event in case if event in self.filtered_events)

    def get_traces_threshold(self) -> float:
        """Get the traces threshold used for filtering the log.

        Returns
        -------
        float
            The traces threshold
        """
        return self.traces_threshold
