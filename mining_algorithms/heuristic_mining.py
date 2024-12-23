import numpy as np
from graphs.visualization.heuristic_graph import HeuristicGraph
from mining_algorithms.base_mining import BaseMining
from logger import get_logger


class HeuristicMining(BaseMining):
    def __init__(self, log):
        super().__init__(log)
        self.logger = get_logger("HeuristicMining")

        self.succession_matrix = self.__create_succession_matrix()
        self.dependency_matrix = self.__create_dependency_matrix()

        # Graph modifiers
        self.spm_threshold = 0.5
        self.min_edge_thickness = 1
        self.min_frequency = 1
        self.dependency_threshold = 0.5

    def create_dependency_graph_with_graphviz(
        self, dependency_threshold, min_frequency, spm_threshold
    ):
        dependency_graph = self.__create_dependency_graph(
            dependency_threshold, min_frequency
        )
        self.start_nodes = self._get_start_nodes()
        self.end_nodes = self._get_end_nodes()

        self.dependency_threshold = dependency_threshold
        self.min_frequency = min_frequency
        self.spm_threshold = spm_threshold

        self.graph = HeuristicGraph()

        _A, _L = self.calculate_A_and_L()

        filtered_nodes = []
        for node in self.events:
            node_freq = self.appearance_frequency.get(node)
            spm = self.calculate_spm(node, self.events, node_freq, _A, _L, self.succession_matrix)
            if spm >= spm_threshold:
                filtered_nodes.append(node)

        # add nodes to graph
        for node in filtered_nodes:
            node_freq = self.appearance_frequency.get(node)
            w, h = self.calulate_node_size(node)
            self.graph.add_event(node, node_freq, (w, h))

        # cluster the edge thickness sizes based on frequency
        edge_frequencies = self.dependency_matrix.flatten()
        edge_frequencies = np.unique(edge_frequencies[edge_frequencies >= 0.0])
        edge_freq_sorted, edge_freq_labels_sorted = self.get_clusters(
            edge_frequencies
        )

        # add edges to graph
        for i in range(len(filtered_nodes)):
            column_total = 0.0
            row_total = 0.0
            for j in range(len(filtered_nodes)):
                column_total = column_total + dependency_graph[i][j]
                row_total = row_total + dependency_graph[j][i]
                if dependency_graph[i][j] == 1.:
                    if dependency_threshold == 0:
                        edge_thickness = 0.1
                    else:
                        edge_thickness = edge_freq_labels_sorted[
                                             edge_freq_sorted.index(self.dependency_matrix[i][j])] + self.min_edge_thickness

                    self.graph.create_edge(
                        source=str(filtered_nodes[i]),
                        destination=str(filtered_nodes[j]),
                        size=edge_thickness,
                        weight=int(self.succession_matrix[i][j])
                    )

                if j == len(filtered_nodes) - 1 and column_total == 0 and filtered_nodes[i] not in self.end_nodes:
                    self.end_nodes.add(filtered_nodes[i])
                if j == len(filtered_nodes) - 1 and row_total == 0 and filtered_nodes[i] not in self.start_nodes:
                    self.start_nodes.add(filtered_nodes[i])

        # add start and end nodes
        self.graph.add_start_node()
        self.graph.add_end_node()

        # add starting and ending edges from the log to the graph. Only if they are filtered
        self.graph.add_starting_edges(self.start_nodes.intersection(filtered_nodes))
        self.graph.add_ending_edges(self.end_nodes.intersection(filtered_nodes))

        # get filtered sources and sinks from the dependency graph
        source_nodes = self.__get_sources_from_dependency_graph(
            dependency_graph
        ).intersection(filtered_nodes)
        sink_nodes = self.__get_sinks_from_dependency_graph(
            dependency_graph
        ).intersection(filtered_nodes)

        # add starting and ending edges from the dependency graph to the graph
        self.graph.add_starting_edges(source_nodes - self.start_nodes)
        self.graph.add_ending_edges(sink_nodes - self.end_nodes)

    def calculate_degree(self, node, events, succession_matrix):
        index = events.index(node)
        in_degree = sum(succession_matrix[:, index])
        out_degree = sum(succession_matrix[index, :])
        return in_degree + out_degree

    def calculate_spm(self, node, events, node_frequency, A, L, succession_matrix):
        degree = self.calculate_degree(node, events, succession_matrix)
        frequency = node_frequency
        spm = 1 - ((degree * A) / (frequency * L))
        return spm

    def calculate_A_and_L(self):
        activities = set()
        total_events = 0
        for trace in self.log:
            total_events += len(trace)
            activities.update(trace)
        return len(activities), total_events

    def get_min_frequency(self):
        return self.min_frequency

    def get_threshold(self):
        return self.dependency_threshold

    def get_spm_threshold(self):
        return self.spm_threshold

    def get_max_frequency(self):
        max_freq = 0
        for value in list(self.appearance_frequency.values()):
            if value > max_freq:
                max_freq = value
        return max_freq

    def __create_succession_matrix(self):
        succession_matrix = np.zeros((len(self.events), len(self.events)))
        for trace in self.log:
            index_x = -1
            for element in trace:

                if index_x < 0:
                    index_x += 1
                    continue
                x = self.events.index(trace[index_x])
                y = self.events.index(element)
                succession_matrix[x][y] += 1
                index_x += 1
        return succession_matrix

    def __create_dependency_matrix(self):
        dependency_matrix = np.zeros(self.succession_matrix.shape)
        y = 0
        for row in self.succession_matrix:
            x = 0
            for i in row:
                if x == y:
                    dependency_matrix[x][y] = self.succession_matrix[x][y] / (self.succession_matrix[x][y] + 1)
                else:
                    dependency_matrix[x][y] = (self.succession_matrix[x][y] - self.succession_matrix[y][x]) / (
                            self.succession_matrix[x][y] + self.succession_matrix[y][x] + 1)
                x += 1
            y += 1
        return dependency_matrix

    def __create_dependency_graph(self, dependency_threshold, min_frequency):
        dependency_graph = np.zeros(self.dependency_matrix.shape)
        y = 0
        for row in dependency_graph:
            for x in range(len(row)):
                if (self.dependency_matrix[y][x] >= dependency_threshold and
                        self.succession_matrix[y][x] >= min_frequency):
                    dependency_graph[y][x] += 1
            y += 1

        return dependency_graph

    def __get_sources_from_dependency_graph(self, dependency_graph):
        indices = self.__get_all_axis_with_all_zero(dependency_graph, axis=0)
        return set([self.events[i] for i in indices])

    def __get_sinks_from_dependency_graph(self, dependency_graph):
        indices = self.__get_all_axis_with_all_zero(dependency_graph, axis=1)
        return set([self.events[i] for i in indices])

    def __get_all_axis_with_all_zero(self, dependency_graph, axis=0):
        filter_matrix = dependency_graph == 0
        # edges from and to the same node are not considered
        np.fill_diagonal(filter_matrix, True)
        return np.where(filter_matrix.all(axis=axis))[0]