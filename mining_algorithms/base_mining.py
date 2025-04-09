import numpy as np
from mining_algorithms.mining_interface import MiningInterface
from logger import get_logger


class BaseMining(MiningInterface):
    """BaseMining is a class that provides the base functionality for the mining algorithms.
    It contains the basic methods for all the mining algorithms, such as creating a succession matrix, calculating node sizes, and filtering out events.
    This base class can be extended by other mining algorithms to implement their specific functionality on top of the base functionality.
    """

    def __init__(self, log: dict[tuple[str, ...], int]):
        """Constructor for BaseMining.

        Parameters
        ----------
        log : dict[tuple[str,...], int]
            A dictionary containing the traces and their frequencies in the log.
            The key is a tuple of strings representing the trace, and the value is an integer representing the frequency of the trace in the log.
        """
        super().__init__()
        self.logger = get_logger("BaseMining")
        self.log = log
        # self.events contains all events(unique!), appearance_activities are dictionaries, events:appearances ex. {'a':3, ...}
        self.events, self.appearance_frequency = self.__filter_out_all_events()
        self.logger.debug(f"Events: {self.events}")
        self.logger.debug(f"Appearance Frequency: {self.appearance_frequency}")
        self.event_positions = {event: i for i, event in enumerate(self.events)}
        self.succession_matrix = self.__create_succession_matrix()
        self.logger.debug(f"Succession Matrix: {self.succession_matrix}")

        self.event_freq_sorted, self.event_freq_labels_sorted = self.get_clusters(
            list(self.appearance_frequency.values())
        )

        self.start_nodes = self._get_start_nodes()
        self.end_nodes = self._get_end_nodes()

        self.spm_threshold = 0.5

        self.logger.debug(f"Start Nodes: {self.start_nodes}")
        self.logger.debug(f"End Nodes: {self.end_nodes}")

    def __filter_out_all_events(self) -> tuple[list[str], dict[str, int]]:
        """create a list of all events and a dictionary of all events with their frequencies

        Returns
        -------
        tuple[list[str], dict[str, int]]
            A tuple containing a list of all unique events and a dictionary of all events with their frequencies
        """
        dic = {}
        for trace, frequency in self.log.items():
            for activity in trace:
                if activity in dic:
                    dic[activity] = dic[activity] + frequency
                else:
                    dic[activity] = frequency

        activities = list(dic.keys())
        return activities, dic

    def calulate_node_size(self, node: str) -> tuple[float, float]:
        """calculate the size of a node based on the frequency of the event.
        The size is determined by the scale factor and the minimum node size.

        Parameters
        ----------
        node : str
            The event for which the size should be calculated

        Returns
        -------
        tuple[float, float]
            A tuple containing the width and height of the node
        """
        scale_factor = self.get_scale_factor(node)

        width = (scale_factor / 2) + self.min_node_size
        height = width / 3
        return width, height

    def get_scale_factor(self, node: str) -> float:
        """get the scale factor for a node based on the frequency of the event.
        The scale factor is computed based on the frequency of the event and the clustering of the frequencies.

        Parameters
        ----------
        node : str
            The event for which the scale factor should be calculated

        Returns
        -------
        float
            The scale factor for the node
        """
        node_freq = self.appearance_frequency.get(node)
        scale_factor = self.event_freq_labels_sorted[
            self.event_freq_sorted.index(node_freq)
        ]
        return scale_factor

    def _get_start_nodes(self) -> set[str]:
        """get all start nodes from the log. A start node is an event that is the first event in a trace.

        Returns
        -------
        set[str]
            A set containing all start nodes
        """
        return set([trace[0] for trace in self.log.keys() if len(trace) > 0])

    def _get_end_nodes(self) -> set[str]:
        """get all end nodes from the log. An end node is an event that is the last event in a trace.

        Returns
        -------
        set[str]
            A set containing all end nodes
        """
        return set([trace[-1] for trace in self.log.keys() if len(trace) > 0])

    def __create_succession_matrix(self) -> np.ndarray:
        """create a succession matrix from the log. The matrix contains the frequencies of the transitions between events.
        Connections from the start node to an event and from an event to the end node are not included in the matrix.

        Returns
        -------
        np.ndarray
            The succession matrix
        """
        succession_matrix = np.zeros((len(self.events), len(self.events)))
        for trace, frequency in self.log.items():
            indices = [self.event_positions[event] for event in trace]
            source_indices = indices[:-1]
            target_indices = indices[1:]
            # https://numpy.org/doc/stable/reference/generated/numpy.ufunc.at.html
            np.add.at(succession_matrix, (source_indices, target_indices), frequency)

        return succession_matrix

    def get_events_to_remove(self, threshold: float) -> set[str]:
        """get all events that have a frequency below a certain threshold. The threshold is a percentage of the maximum frequency of an event.

        Parameters
        ----------
        threshold : float
            The threshold for the frequency of an event. The threshold is a percentage of the maximum frequency of an event.

        Returns
        -------
        set[str]
            A set containing all events that have a frequency below the threshold
        """
        if threshold > 1.0:
            threshold = 1.0
        elif threshold < 0.0:
            threshold = 0.0

        minimum_event_freq = round(max(self.appearance_frequency.values()) * threshold)

        return set(
            [
                event
                for event, freq in self.appearance_frequency.items()
                if freq < minimum_event_freq
            ]
        )

    def calulate_minimum_traces_frequency(self, threshold: float) -> int:
        """calculate the minimum frequency of a trace based on a threshold. The threshold is a percentage of the maximum frequency of a trace.

        Parameters
        ----------
        threshold : float
            The threshold for the frequency of a trace. The threshold is a percentage of the maximum frequency of a trace.

        Returns
        -------
        int
            The minimum frequency of a trace, based on the threshold.
        """
        if threshold > 1.0:
            threshold = 1.0
        elif threshold < 0.0:
            threshold = 0.0

        minimum_trace_freq = round(max(self.log.values()) * threshold)

        return minimum_trace_freq

    def calculate_degree(self, node, events, succession_matrix):
        """Calculate the degree of a node in the succession matrix.

        The degree is defined as the sum of the node's in-degree and out-degree,
        based on the succession matrix.

        Parameters
        ----------
        node : str
            The event node whose degree is being calculated.
        events : list[str]
            List of all event names in the log.
        succession_matrix : np.ndarray
            Matrix representing direct succession counts between events.

        Returns
        -------
        int
            The total degree (in-degree + out-degree) of the node.
        """
        index = events.index(node)
        in_degree = sum(succession_matrix[:, index])
        out_degree = sum(succession_matrix[index, :])
        return in_degree + out_degree

    def calculate_spm(self, node, events, node_frequency, A, L, succession_matrix):
        """Calculate the Search Process Model (SPM) score for a node.

        Parameters
        ----------
        node : str
            The event node being evaluated.
        events : list[str]
            List of all events in the log.
        node_frequency : int
            Number of times the node appears in the log.
        A : int
            Number of distinct activities in the log.
        L : int
            Total number of events in the log.
        succession_matrix : np.ndarray
            Matrix representing direct succession counts between events.

        Returns
        -------
        float
            The calculated SPM score for the node.
        """
        degree = self.calculate_degree(node, events, succession_matrix)
        frequency = node_frequency
        spm = 1 - ((degree * A) / (frequency * L))
        return spm

    def calculate_A_and_L(self):
        """Calculate the number of distinct activities (A) and total events (L) in the log.

        Returns
        -------
        tuple[int, int]
            A tuple containing A (number of distinct activities) and L (total number of events).
        """
        activities = set()
        total_events = 0
        for trace in self.log:
            total_events += len(trace)
            activities.update(trace)
        return len(activities), total_events

    def get_spm_threshold(self):
        """Get the current threshold used for the SPM filter.

            Returns
            -------
            float
                The SPM threshold value.
        """
        return self.spm_threshold

    def get_spm_filtered_events(self) -> set[str]:
        """Filter events based on their SPM value using the current threshold.

        This method calculates the SPM value for all events in the log and filters out those
        which are below the current `spm_threshold`. The result is a set of events that passed
        the filter and are considered relevant for process model generation.

        Returns
        -------
        set[str]
            A set of event labels that have an SPM value greater than or equal to the threshold.
        """
        _A, _L = self.calculate_A_and_L()
        filtered_nodes = []
        for node in self.events:
            node_freq = self.appearance_frequency.get(node)
            spm = self.calculate_spm(node, self.events, node_freq, _A, _L, self.succession_matrix)
            if spm >= self.spm_threshold:
                filtered_nodes.append(node)
        return filtered_nodes
