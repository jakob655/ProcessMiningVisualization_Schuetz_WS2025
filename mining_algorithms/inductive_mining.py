from mining_algorithms.base_mining import BaseMining
from logs.splits import (
    exclusive_split,
    parallel_split,
    sequence_split,
    loop_split,
)
from graphs.cuts import exclusive_cut, parallel_cut, sequence_cut, loop_cut
from graphs.dfg import DFG
from graphs.visualization.inductive_graph import InductiveGraph
from logs.filters import filter_events, filter_traces
from logger import get_logger


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
        self.activity_threshold = 0.0
        self.traces_threshold = 0.2
        self.filtered_log = None

    def generate_graph(self, spm_threshold: float, node_freq_threshold: float, edge_freq_threshold: float,
                       activity_threshold: float,
                       traces_threshold: float):
        """Generate a graph from the log using the Inductive Mining algorithm.

        Parameters
        ----------
        spm_threshold : float
            The threshold for the SPM (Search Process Model) score. Events with an SPM score below this value
            will be filtered out before generating the graph.
        node_freq_threshold : float
            The threshold for the normalized frequency of nodes (events). Only nodes with a relative frequency
            equal to or higher than this threshold will be kept.
        edge_freq_threshold : float
            The threshold for the normalized frequency of edges (direct event transitions). Only edges with a
            relative frequency equal to or higher than this threshold will be included in the graph.
        activity_threshold : float
            The activity threshold for the filtering of the log.
            All events with a frequency lower than the threshold * max_event_frequency will be removed.
        traces_threshold : float
            The traces threshold for the filtering of the log.
            All traces with a frequency lower than the threshold * max_trace_frequency will be removed.
        """
        self.activity_threshold = activity_threshold
        self.traces_threshold = traces_threshold
        self.spm_threshold = spm_threshold
        self.node_freq_threshold = node_freq_threshold
        self.edge_freq_threshold = edge_freq_threshold

        self.recalculate_model_filters()

        if not self.filtered_events:
            self.graph = InductiveGraph(self.filtered_events)
            self.graph.add_edge("Start", "End", None)
            return

        self.node_sizes = {k: self.calculate_node_size(k) for k in self.filtered_events}

        events_to_remove = self.get_events_to_remove(activity_threshold)

        self.logger.debug(f"Events to remove: {events_to_remove}")
        min_traces_frequency = self.calculate_minimum_traces_frequency(traces_threshold)

        filtered_log_before_edges = filter_traces(self.node_frequency_filtered_log, min_traces_frequency)
        filtered_log_before_edges = filter_events(filtered_log_before_edges, events_to_remove)
        filtered_log, disconnected = self._filter_edges_by_frequency(filtered_log_before_edges)
        filtered_log = self._append_disconnected_nodes_as_traces(filtered_log, disconnected)

        self.filtered_log = filtered_log

        self.logger.info("Start Inductive Mining")
        process_tree = self.inductive_mining(self.filtered_log)
        node_stats_map = {stat["node"]: stat for stat in self.get_node_statistics()}
        edge_stats_map = {(edge["source"], edge["target"]): edge for edge in self.get_edge_statistics()}
        self.graph = InductiveGraph(
            process_tree,
            frequency=self.filtered_appearance_freqs,
            node_sizes=self.node_sizes,
            node_stats_map=node_stats_map,
            edge_stats_map=edge_stats_map
        )

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

    def get_activity_threshold(self) -> float:
        """Get the activity threshold used for filtering the log.

        Returns
        -------
        float
            The activity threshold
        """
        return self.activity_threshold

    def get_traces_threshold(self) -> float:
        """Get the traces threshold used for filtering the log.

        Returns
        -------
        float
            The traces threshold
        """
        return self.traces_threshold

    def _filter_edges_by_frequency(self, log: dict[tuple[str, ...], int]) -> dict[tuple[str, ...], int]:
        """
        Filter the directly-follows relations in the log based on edge frequency threshold.

        Edges (event transitions) with a normalized frequency below `edge_freq_threshold`
        are removed from the traces. Additionally, all nodes that become disconnected due to removed edges
        are tracked for potential reinsertion later.

        Parameters
        ----------
        log : dict[tuple[str, ...], int]
            A dictionary containing the traces and their frequencies.

        Returns
        -------
        tuple[dict[tuple[str, ...], int], set[str]]
            A tuple containing:
            - The filtered log where only edges meeting the frequency threshold are retained.
            - A set of nodes that became disconnected due to edge filtering.
        """
        if self.edge_freq_threshold <= 0.0:
            self.logger.debug("[_filter_edges_by_frequency] Skipping filtering due to 0.0 threshold")
            return log, set()

        filtered_log = {}
        dropped_events = set()

        for trace, freq in log.items():
            new_trace = [trace[0]] if trace else []
            for i in range(len(trace) - 1):
                source, target = trace[i], trace[i + 1]
                passes_filter = self.filter_edge(source, target)
                self.logger.debug(f"[_filter_edges_by_frequency] {source} → {target} | Passes: {passes_filter}")

                if passes_filter:
                    new_trace.append(target)
                else:
                    self.logger.debug(f"[_filter_edges_by_frequency] Dropping edge {source} → {source}")
                    dropped_events.update([source, target])

            if len(new_trace) > 1:
                filtered_log[tuple(new_trace)] = freq
            elif len(new_trace) == 1:
                # Retain single-node traces if they still make sense
                filtered_log[(new_trace[0],)] = freq

        return filtered_log, dropped_events

    def _append_disconnected_nodes_as_traces(self, filtered_log: dict[tuple[str, ...], int],
                                             edge_filtered_nodes: set[str]) -> dict[tuple[str, ...], int]:
        """
        Reintroduce nodes that were disconnected due to edge filtering as isolated traces.

        Events removed from the graph structure because all of their connecting edges were filtered out
        are added back as individual traces. This ensures these nodes are still represented in the process model.

        Parameters
        ----------
        filtered_log : dict[tuple[str, ...], int]
            The log after edge-based filtering.

        edge_filtered_nodes : set[str]
            A set of nodes that became disconnected due to edge filtering.

        Returns
        -------
        dict[tuple[str, ...], int]
            The log with disconnected but valid events re-added as single-node traces.
        """
        connected_events = self.get_connected_events(filtered_log)
        disconnected_events = edge_filtered_nodes - connected_events

        self.logger.debug(
            f"[_append_disconnected_nodes_as_traces] Disconnected events to re-added as individual traces: {disconnected_events}")

        for node in disconnected_events:
            freq = self.filtered_appearance_freqs.get(node, 1)
            filtered_log[(node,)] = freq

        return filtered_log