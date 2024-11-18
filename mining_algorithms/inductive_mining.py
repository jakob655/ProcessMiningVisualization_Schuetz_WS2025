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
        self.node_sizes = {k: self.calulate_node_size(k) for k in self.events}
        self.activity_threshold = 0.0
        self.traces_threshold = 0.2
        self.filtered_log = None

    def generate_graph(self, activity_threshold: float, traces_threshold: float):
        """Generate a graph from the log using the Inductive Mining algorithm.

        Parameters
        ----------
        activity_threshold : float
            The activity threshold for the filtering of the log.
            All events with a frequency lower than the threshold * max_event_frequency will be removed.
        traces_threshold : float
            The traces threshold for the filtering of the log.
            All traces with a frequency lower than the threshold * max_trace_frequency will be removed.
        """
        self.activity_threshold = activity_threshold
        self.traces_threshold = traces_threshold

        events_to_remove = self.get_events_to_remove(activity_threshold)

        self.logger.debug(f"Events to remove: {events_to_remove}")
        min_traces_frequency = self.calulate_minimum_traces_frequency(traces_threshold)

        filtered_log = filter_traces(self.log, min_traces_frequency)
        filtered_log = filter_events(filtered_log, events_to_remove)

        if filtered_log == self.filtered_log:
            return

        self.filtered_log = filtered_log

        self.logger.info("Start Inductive Mining")
        process_tree = self.inductive_mining(self.filtered_log)
        self.graph = InductiveGraph(
            process_tree,
            frequency=self.appearance_frequency,
            node_sizes=self.node_sizes,
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
            if partitions := self.calulate_cut(log):
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

    def calulate_cut(self, log) -> tuple | None:
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
        return set([event for case in log for event in case])

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
