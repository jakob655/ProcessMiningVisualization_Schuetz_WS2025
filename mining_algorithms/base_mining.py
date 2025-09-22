from collections import Counter

import numpy as np
import streamlit as st

from logger import get_logger
from logs.filters import filter_events
from mining_algorithms.mining_interface import MiningInterface


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
        self.events, self.appearance_frequency = self.__filter_out_all_events(self.log)
        self.logger.debug(f"Events: {self.events}")
        self.logger.debug(f"Appearance Frequency: {self.appearance_frequency}")
        self.event_positions = self.get_event_positions(self.events)
        self.succession_matrix = self.__create_succession_matrix(self.log, self.events, self.event_positions)
        self.logger.debug(f"Succession Matrix: \n {self.succession_matrix}")

        self._init_filter_thresholds()

        self._filter_state = {
            "spm_threshold": self.spm_threshold,
            "node_freq_threshold_normalized": self.node_freq_threshold_normalized,
            "node_freq_threshold_absolute": self.node_freq_threshold_absolute,
            "edge_freq_threshold_normalized": self.edge_freq_threshold_normalized,
            "edge_freq_threshold_absolute": self.edge_freq_threshold_absolute
        }

        self.spm_filtered_events = self.get_spm_filtered_events()
        self.spm_filtered_log = filter_events(self.log, set(self.events) - set(
            self.spm_filtered_events))

        self.logger.debug(f"SPM Filtered Events: {self.spm_filtered_events}")
        self.logger.debug(f"SPM Filtered Log: {self.spm_filtered_log}")

        self.node_frequencies = self.get_node_frequencies()
        self.node_frequency_filtered_events = list(self.node_frequencies.keys())
        self.node_frequency_filtered_log = filter_events(self.spm_filtered_log, set(self.spm_filtered_events) - set(
            self.node_frequency_filtered_events))

        self.logger.debug(f"SPM & Frequency Filtered Events: {self.node_frequency_filtered_events}")
        self.logger.debug(f"SPM & Frequency Filtered Log: {self.node_frequency_filtered_log}")

        self.edge_absolute_counts = None

        self.normalized_edge_frequencies = self.get_normalized_edge_frequencies()

        self.start_nodes = self._get_start_nodes()
        self.end_nodes = self._get_end_nodes()

        self.logger.debug(f"Start Nodes: {self.start_nodes}")
        self.logger.debug(f"End Nodes: {self.end_nodes}")

        self.filtered_event_positions = self.get_event_positions(self.node_frequency_filtered_events)
        self.filtered_succession_matrix = self.__create_succession_matrix(self.node_frequency_filtered_log,
                                                                          self.node_frequency_filtered_events,
                                                                          self.filtered_event_positions)
        self.logger.debug(f"Filtered succession Matrix: \n {self.filtered_succession_matrix}")

        self.filtered_events, self.filtered_appearance_freqs = self.__filter_out_all_events(
            self.node_frequency_filtered_log)
        self.logger.debug(f"Filtered Events: {self.filtered_events}")

        self.node_freq_threshold_absolute = int(
            round(max(self.filtered_appearance_freqs.values(), default=0) * self.node_freq_threshold_normalized))
        if self.edge_absolute_counts:
            self.edge_freq_threshold_absolute = int(
                round(max(self.edge_absolute_counts.values(), default=0) * self.edge_freq_threshold_normalized))
        else:
            self.edge_freq_threshold_absolute = 0

        # Session-state-initialization for thresholds
        for norm_attr, abs_attr in [
            ("node_freq_threshold_normalized", "node_freq_threshold_absolute"),
            ("edge_freq_threshold_normalized", "edge_freq_threshold_absolute"),
        ]:
            norm_val = getattr(self, norm_attr)
            abs_val = getattr(self, abs_attr)

            st.session_state[norm_attr] = norm_val
            st.session_state[abs_attr] = abs_val
            st.session_state[f"last_{norm_attr}"] = norm_val
            st.session_state[f"last_{abs_attr}"] = abs_val

        self.edge_freq = self.filtered_succession_matrix.flatten()
        self.edge_freq = np.unique(self.edge_freq[self.edge_freq >= 0.0])
        if self.edge_freq.size == 0:
            self.edge_freq_sorted, self.edge_freq_labels_sorted = [0.0], [1.0]
        else:
            self.edge_freq_sorted, self.edge_freq_labels_sorted = self.get_clusters(self.edge_freq)

        freq_values = list(self.filtered_appearance_freqs.values())
        if len(freq_values) <= 1 or max(freq_values) == min(freq_values):
            self.logger.debug("Clustering skipped: not enough distinct node frequencies.")
            self.event_freq_sorted, self.event_freq_labels_sorted = freq_values, [0] * len(freq_values)
        else:
            self.event_freq_sorted, self.event_freq_labels_sorted = self.get_clusters(freq_values)

    @staticmethod
    def __filter_out_all_events(log) -> tuple[list[str], dict[str, int]]:
        """create a list of all events and a dictionary of all events with their frequencies

        Returns
        -------
        tuple[list[str], dict[str, int]]
            A tuple containing a list of all unique events and a dictionary of all events with their frequencies
        """
        dic = {}
        for trace, frequency in log.items():
            for activity in trace:
                if activity in dic:
                    dic[activity] = dic[activity] + frequency
                else:
                    dic[activity] = frequency

        activities = list(dic.keys())
        return activities, dic

    @staticmethod
    def get_event_positions(events):
        return {event: i for i, event in enumerate(events)}

    def calculate_node_size(self, node: str) -> tuple[float, float]:
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
        node_freq = self.filtered_appearance_freqs.get(node)
        scale_factor = self.event_freq_labels_sorted[
            self.event_freq_sorted.index(node_freq)
        ]
        return scale_factor

    def get_edge_scale_factor(self, source, target):
        """get the scale factor for an edge based on its frequency.
        The scale factor is computed based on the frequency of the edge and the clustering of the frequencies.

        Parameters
        ----------
        source : str
            The source event of the edge connection for which the scale factor should be calculated
        target : str
            The target event of the edge connection for which the scale factor should be calculated
        Returns
        -------
        float
            The scale factor for the edge
        """
        scale_factor = self.edge_freq_labels_sorted[
            self.edge_freq_sorted.index(self.filtered_succession_matrix[source][target])
        ]
        return scale_factor

    def get_edge_scale_factor_abs_freq(self, abs_freq: float) -> float:
        """get the scale factor for an edge on the absolute frequency of the edge.
        The scale factor is computed based on the frequency of the edge and the clustering of the frequencies.

        Parameters
        ----------
        abs_freq : float
            The absolute frequency of the edge connection for which the scale factor should be calculated
        Returns
        -------
        float
            The scale factor for the edge
        """
        if abs_freq in self.edge_freq_sorted:
            idx = self.edge_freq_sorted.index(abs_freq)
        else:
            idx = min(range(len(self.edge_freq_sorted)), key=lambda k: abs(abs_freq - self.edge_freq_sorted[k]))

        return self.edge_freq_labels_sorted[idx]

    def _get_start_nodes(self) -> set[str]:
        """get all start nodes from the log. A start node is an event that is the first event in a trace.

        Returns
        -------
        set[str]
            A set containing all start nodes
        """
        return set([trace[0] for trace in self.node_frequency_filtered_log.keys() if len(trace) > 0])

    def _get_end_nodes(self) -> set[str]:
        """get all end nodes from the log. An end node is an event that is the last event in a trace.

        Returns
        -------
        set[str]
            A set containing all end nodes
        """
        return set([trace[-1] for trace in self.node_frequency_filtered_log.keys() if len(trace) > 0])

    @staticmethod
    def __create_succession_matrix(log, events, event_positions) -> np.ndarray:
        """create a succession matrix from the log. The matrix contains the frequencies of the transitions between events.
        Connections from the start node to an event and from an event to the end node are not included in the matrix.

        Parameters
        ----------
        log : dict[tuple[str, ...], int]
            A dictionary of traces and their frequencies. Each trace is a tuple of event labels,
            and the value is the number of times the trace occurred.
        events : list[str]
            A list of all unique event labels present in the log. The order of events determines the
            row/column indexing in the resulting matrix.
        event_positions : dict[str, int]
            A dictionary mapping each event label to its index in the `events` list.

        Returns
        -------
        np.ndarray
            A 2D NumPy array of shape (len(events), len(events)) containing the frequency of direct successions
            between events. Entry [i][j] represents how often event `events[i]` is directly followed by `events[j]`.
        """
        succession_matrix = np.zeros((len(events), len(events)))
        for trace, frequency in log.items():
            indices = [event_positions[event] for event in trace]
            source_indices = indices[:-1]
            target_indices = indices[1:]
            # https://numpy.org/doc/stable/reference/generated/numpy.ufunc.at.html
            np.add.at(succession_matrix, (source_indices, target_indices), frequency)

        return succession_matrix

    def get_events_to_remove(self, threshold: float) -> set[str]:
        """get all filtered events that have a frequency below a certain threshold. The threshold is a percentage of the maximum frequency of an event.

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

        minimum_event_freq = round(max(self.filtered_appearance_freqs.values()) * threshold)

        return set(
            [
                event
                for event, freq in self.filtered_appearance_freqs.items()
                if freq < minimum_event_freq
            ]
        )

    def calculate_minimum_traces_frequency(self, threshold: float) -> int:
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

        minimum_trace_freq = round(max(self.node_frequency_filtered_log.values()) * threshold)

        return minimum_trace_freq

    @staticmethod
    def calculate_degree(node, events, succession_matrix):
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

    def get_spm_filtered_events(self) -> list[str]:
        """Filter events based on their SPM value using the current threshold.

        Calculates the SPM value for all events in the log and filters out those
        which are below the current `spm_threshold`. The result is a set of events that passed
        the filter and are considered relevant for process model generation.

        Returns
        -------
        set[str]
            A set of event labels that have an SPM value greater than or equal to the threshold.
        """
        A, L = self.calculate_A_and_L()
        filtered_events = []
        for node in self.events:
            node_freq = self.appearance_frequency.get(node)
            spm = self.calculate_spm(node, self.events, node_freq, A, L, self.succession_matrix)
            if spm >= self.spm_threshold:
                filtered_events.append(node)
        return filtered_events

    def get_spm_threshold(self):
        """Get the current threshold used for the SPM filter.

        Returns
        -------
        float
            The SPM threshold value.
        """
        return self.spm_threshold

    def get_node_frequencies(self) -> dict:
        """
        Calculate the normalized frequency of each node (event) in the SPM-filtered log.

        Frequencies are calculated across all traces in the log and normalized by the
        maximum event count. This helps identify which events are most common
        in the process after applying the SPM filtering step.

        Returns
        -------
        dict[str, int]
            A dictionary with each allowed node label mapped to its absolute frequency.
        """
        node_counts = Counter()
        for trace, freq in self.spm_filtered_log.items():
            for event in trace:
                node_counts[event] += freq

        if not node_counts:
            return {}

        max_count = max(node_counts.values())
        normalized = {node: count / max_count for node, count in node_counts.items()}

        allowed_nodes = {
            node: freq for node, freq in normalized.items()
            if freq >= self.node_freq_threshold_normalized
        }

        return allowed_nodes

    def get_node_statistics(self) -> list[dict]:
        """
        Returns statistics for each node in the current filtered event list.
        Includes: node label, normalized frequency, and SPM value.

        Returns
        -------
        list[dict]
            List of node statistics dictionaries.
        """
        A, L = self.calculate_A_and_L()
        stats = []

        max_abs = max(self.node_frequencies.values(), default=0)

        for node in self.filtered_events:
            abs_freq = self.node_frequencies.get(node, 0)
            freq = (abs_freq / max_abs) if max_abs > 0 else 0.0
            spm = self.calculate_spm(
                node,
                self.events,
                self.appearance_frequency.get(node, 0),
                A,
                L,
                self.succession_matrix
            )
            stats.append({
                "node": node,
                "frequency": freq,
                "absolute_frequency": abs_freq,
                "spm": spm
            })

        return stats

    def get_node_frequency_threshold_normalized(self):
        """Get the current normalized threshold used for the node frequency filter.

        Returns
        -------
        float
            The normalized node frequency.
        """
        return self.node_freq_threshold_normalized

    def get_node_frequency_threshold_absolute(self):
        """Get the current absolute threshold used for the node frequency filter.

        Returns
        -------
        float
            The absolute node frequency.
        """
        return self.node_freq_threshold_absolute

    def get_normalized_edge_frequencies(self) -> dict:
        """
        Calculate the normalized frequency of each directly-follows edge in the filtered log.

        Frequencies are based on the number of times one event is followed directly
        by another in the node-frequency-filtered log, normalized by the maximum count
        of such transitions.

        Returns
        -------
        dict[tuple[str, str], float]
            A dictionary with each allowed edge (as a tuple of source and target event labels)
            mapped to its normalized frequency.
        """
        edge_counts = Counter()
        for trace, freq in self.node_frequency_filtered_log.items():
            for i in range(len(trace) - 1):
                edge = (trace[i], trace[i + 1])
                edge_counts[edge] += freq

        self.edge_absolute_counts = dict(edge_counts)

        if not edge_counts:
            return {}

        max_count = max(edge_counts.values())
        return {edge: count / max_count for edge, count in edge_counts.items()}

    def get_edge_statistics(self) -> list[dict]:
        """
        Returns statistics for each edge (filtered) including frequency.

        Returns
        -------
        list[dict]
            List of edge statistics with source, target, and normalized frequency.
        """
        return [
            {
                "source": a,
                "target": b,
                "normalized_frequency": freq,
                "absolute_frequency": self.edge_absolute_counts.get((a, b), 0)
            }
            for (a, b), freq in self.normalized_edge_frequencies.items()
            if self.filter_edge(a, b)
        ]

    def get_edge_frequency_threshold_normalized(self):
        """Get the current normalized threshold used for the edge frequency filter.

        Returns
        -------
        float
            The normalized edge frequency.
        """
        return self.edge_freq_threshold_normalized

    def get_edge_frequency_threshold_absolute(self):
        """Get the current absolute threshold used for the edge frequency filter.

        Returns
        -------
        float
            The absolute edge frequency.
        """
        return self.edge_freq_threshold_absolute

    def filter_edge(self, a: str, b: str) -> bool:
        """Return True if edge (a -> b) passes frequency threshold"""
        return self.normalized_edge_frequencies.get((a, b), 0) >= self.edge_freq_threshold_normalized

    def _init_filter_thresholds(
            self,
            spm_threshold: float = 0.10,
            node_freq_threshold_normalized: float = 0.25,
            node_freq_threshold_absolute: int = 1,
            edge_freq_threshold_normalized: float = 0.25,
            edge_freq_threshold_absolute: int = 1,
    ):
        """Initialize default thresholds for all filtering steps.

        Parameters
        ----------
        spm_threshold : float, optional
            Minimum SPM value required to keep an event (default: 0.15).
        node_freq_threshold_normalized : float, optional
            Minimum normalized node frequency to keep an event (default: 0.25).
        node_freq_threshold_absolute : int, optional
            Minimum absolute node frequency to keep an event (default: 1).
        edge_freq_threshold_normalized : float, optional
            Minimum normalized edge frequency to keep a directly-follows edge (default: 0.25).
        edge_freq_threshold_absolute : int, optional
            Minimum absolute edge frequency to keep a directly-follows edge (default: 1).

        """
        self.spm_threshold = spm_threshold
        self.node_freq_threshold_normalized = node_freq_threshold_normalized
        self.node_freq_threshold_absolute = node_freq_threshold_absolute
        self.edge_freq_threshold_normalized = edge_freq_threshold_normalized
        self.edge_freq_threshold_absolute = edge_freq_threshold_absolute

    def recalculate_model_filters(self):
        """Recalculate all filtered events, logs, and frequencies based on current thresholds, if they have changed."""

        current_state = {
            "spm_threshold": self.spm_threshold,
            "node_freq_threshold_normalized": self.node_freq_threshold_normalized,
            "node_freq_threshold_absolute": self.node_freq_threshold_absolute,
            "edge_freq_threshold_normalized": self.edge_freq_threshold_normalized,
            "edge_freq_threshold_absolute": self.edge_freq_threshold_absolute
        }

        if current_state == self._filter_state:
            return

        self._filter_state = current_state

        self.spm_filtered_events = self.get_spm_filtered_events()
        self.spm_filtered_log = filter_events(self.log, set(self.events) - set(
            self.spm_filtered_events))

        self._sync_thresholds("node_freq_threshold_normalized", "node_freq_threshold_absolute",
                              self.appearance_frequency)

        self.node_frequencies = self.get_node_frequencies()
        self.node_frequency_filtered_events = list(self.node_frequencies.keys())
        self.node_frequency_filtered_log = filter_events(self.spm_filtered_log, set(self.spm_filtered_events) - set(
            self.node_frequency_filtered_events))

        self.normalized_edge_frequencies = self.get_normalized_edge_frequencies()

        self.start_nodes = self._get_start_nodes()
        self.end_nodes = self._get_end_nodes()

        self.filtered_event_positions = self.get_event_positions(self.node_frequency_filtered_events)
        self.filtered_succession_matrix = self.__create_succession_matrix(self.node_frequency_filtered_log,
                                                                          self.node_frequency_filtered_events,
                                                                          self.filtered_event_positions)

        self.filtered_events, self.filtered_appearance_freqs = self.__filter_out_all_events(
            self.node_frequency_filtered_log)

        self._sync_thresholds("node_freq_threshold_normalized", "node_freq_threshold_absolute",
                              self.filtered_appearance_freqs)
        self._sync_thresholds("edge_freq_threshold_normalized", "edge_freq_threshold_absolute",
                              self.edge_absolute_counts or {})

        self.edge_freq = self.filtered_succession_matrix.flatten()
        self.edge_freq = np.unique(self.edge_freq[self.edge_freq >= 0.0])
        if self.edge_freq.size == 0:
            self.edge_freq_sorted, self.edge_freq_labels_sorted = [0.0], [1.0]
        else:
            self.edge_freq_sorted, self.edge_freq_labels_sorted = self.get_clusters(self.edge_freq)

        if self.node_frequency_filtered_events:
            freq_values = list(self.filtered_appearance_freqs.values())
            if len(freq_values) <= 1 or max(freq_values) == min(freq_values):
                self.logger.debug("Clustering skipped: not enough distinct node frequencies.")
                self.event_freq_sorted, self.event_freq_labels_sorted = freq_values, [0] * len(freq_values)
            else:
                self.event_freq_sorted, self.event_freq_labels_sorted = self.get_clusters(freq_values)

    def _sync_thresholds(self, norm_attr: str, abs_attr: str, freqs: dict[str, int]) -> None:
        """Synchronize normalized and absolute thresholds bidirectionally, keeping them consistent also if max_val changes after filtering."""
        max_val = max(freqs.values()) if freqs else 0

        if max_val <= 0:
            norm_val, abs_val = 0.0, 0
        else:
            # set current session values into valid ranges
            abs_val = min(st.session_state.get(abs_attr, 0), max_val)
            norm_val = min(max(st.session_state.get(norm_attr, 0.0), 0.0), 1.0)

            # check which one was moved last
            if st.session_state.get(f"last_{norm_attr}") != norm_val:
                abs_val = int(round(norm_val * max_val))
            elif st.session_state.get(f"last_{abs_attr}") != abs_val:
                norm_val = (abs_val / max_val) if max_val > 0 else 0.0
            else:
                # recompute absolute after max_val change
                abs_val = int(round(norm_val * max_val))

        st.session_state[abs_attr] = abs_val
        st.session_state[norm_attr] = norm_val
        st.session_state[f"last_{norm_attr}"] = norm_val
        st.session_state[f"last_{abs_attr}"] = abs_val

        setattr(self, norm_attr, norm_val)
        setattr(self, abs_attr, abs_val)
