import numpy as np

from graphs.visualization.heuristic_graph import HeuristicGraph
from logger import get_logger
from mining_algorithms.base_mining import BaseMining


class HeuristicMining(BaseMining):
    def __init__(self, log):
        super().__init__(log)
        self.logger = get_logger("HeuristicMining")

        self.dependency_matrix = {}

        # Graph modifiers
        self.min_edge_thickness = 1
        self.dependency_threshold = 0.5

    def generate_graph(
            self, spm_threshold, node_freq_threshold, edge_freq_threshold, dependency_threshold):
        self.graph = HeuristicGraph()

        self.start_nodes = self._get_start_nodes()
        self.end_nodes = self._get_end_nodes()

        self.spm_threshold = spm_threshold
        self.node_freq_threshold = node_freq_threshold
        self.edge_freq_threshold = edge_freq_threshold
        self.dependency_threshold = dependency_threshold

        self.recalculate_model_filters()
        self.dependency_matrix = self.__create_dependency_matrix()

        dependency_graph = self.__create_dependency_graph(dependency_threshold)

        node_stats_map = {stat["node"]: stat for stat in self.get_node_statistics()}

        self.graph.add_start_node()
        self.graph.add_end_node()

        if not self.filtered_events:
            self.graph.create_edge(
                source=str("Start"),
                destination=str("End"),
                size=0.1,
            )
            return

        # add nodes to graph
        for node in self.filtered_events:
            w, h = self.calculate_node_size(node)
            stat = node_stats_map.get(node, {})
            spm = stat.get("spm", 0.0)
            norm_freq = stat.get("frequency", 0.0)
            abs_freq = self.filtered_appearance_freqs.get(node, 0)

            self.graph.add_event(
                title=node,
                spm=spm,
                normalized_frequency=norm_freq,
                absolute_frequency=abs_freq,
                size=(w, h)
            )

        # cluster the edge thickness sizes based on frequency
        if self.dependency_matrix.any():
            edge_frequencies = self.dependency_matrix.flatten()
            edge_frequencies = np.unique(edge_frequencies[edge_frequencies >= 0.0])
            edge_freq_sorted, edge_freq_labels_sorted = self.get_clusters(
                edge_frequencies
            )

        # add edges to graph
        for i in range(len(self.filtered_events)):
            column_total = 0.0
            row_total = 0.0
            for j in range(len(self.filtered_events)):
                column_total = column_total + dependency_graph[i][j]
                row_total = row_total + dependency_graph[j][i]
                source = self.filtered_events[i]
                target = self.filtered_events[j]

                edge_stats = self.get_edge_statistics()
                edge_stats_map = {(edge["source"], edge["target"]): edge for edge in edge_stats}

                if dependency_graph[i][j] == 1.:
                    norm_frequency = edge_stats_map.get((source, target), {}).get("normalized_frequency", 0.0)
                    abs_frequency = edge_stats_map.get((source, target), {}).get("absolute_frequency", 0)
                    if dependency_threshold == 0:
                        edge_thickness = 0.1
                    else:
                        edge_thickness = edge_freq_labels_sorted[
                                             edge_freq_sorted.index(
                                                 self.dependency_matrix[i][j])] + self.min_edge_thickness

                    self.graph.create_edge(
                        source=source,
                        destination=target,
                        size=edge_thickness,
                        normalized_frequency=norm_frequency,
                        absolute_frequency=abs_frequency
                    )

                if j == len(self.filtered_events) - 1 and column_total == 0 and \
                        self.filtered_events[i] not in self.end_nodes:
                    self.end_nodes.add(self.filtered_events[i])
                if j == len(self.filtered_events) - 1 and row_total == 0 and \
                        self.filtered_events[i] not in self.start_nodes:
                    self.start_nodes.add(self.filtered_events[i])

        # add starting and ending edges from the log to the graph. Only if they are filtered
        self.graph.add_starting_edges(self.start_nodes.intersection(self.filtered_events))
        self.graph.add_ending_edges(self.end_nodes.intersection(self.filtered_events))

        # get filtered sources and sinks from the dependency graph
        source_nodes = self.__get_sources_from_dependency_graph(
            dependency_graph
        ).intersection(self.filtered_events)
        sink_nodes = self.__get_sinks_from_dependency_graph(
            dependency_graph
        ).intersection(self.filtered_events)

        # add starting and ending edges from the dependency graph to the graph
        self.graph.add_starting_edges(source_nodes - self.start_nodes)
        self.graph.add_ending_edges(sink_nodes - self.end_nodes)

    def get_threshold(self):
        return self.dependency_threshold

    def __create_dependency_matrix(self):
        dependency_matrix = np.zeros(self.filtered_succession_matrix.shape)
        y = 0
        for row in self.filtered_succession_matrix:
            x = 0
            for i in row:
                if x == y:
                    dependency_matrix[x][y] = self.filtered_succession_matrix[x][y] / (
                                self.filtered_succession_matrix[x][y] + 1)
                else:
                    dependency_matrix[x][y] = (self.filtered_succession_matrix[x][y] -
                                               self.filtered_succession_matrix[y][x]) / (
                                                      self.filtered_succession_matrix[x][y] +
                                                      self.filtered_succession_matrix[y][x] + 1)
                x += 1
            y += 1
        return dependency_matrix

    def __create_dependency_graph(self, dependency_threshold):
        dependency_graph = np.zeros(self.dependency_matrix.shape)
        y = 0
        for row in dependency_graph:
            for x in range(len(row)):
                a = self.filtered_events[y]
                b = self.filtered_events[x]

                if (
                        self.dependency_matrix[y][x] >= dependency_threshold and
                        self.filter_edge(a, b)
                ):
                    dependency_graph[y][x] += 1
            y += 1

        return dependency_graph

    def __get_sources_from_dependency_graph(self, dependency_graph):
        indices = self.__get_all_axis_with_all_zero(dependency_graph, axis=0)
        return set([self.filtered_events[i] for i in indices])

    def __get_sinks_from_dependency_graph(self, dependency_graph):
        indices = self.__get_all_axis_with_all_zero(dependency_graph, axis=1)
        return set([self.filtered_events[i] for i in indices])

    def __get_all_axis_with_all_zero(self, dependency_graph, axis=0):
        filter_matrix = dependency_graph == 0
        # edges from and to the same node are not considered
        np.fill_diagonal(filter_matrix, True)
        return np.where(filter_matrix.all(axis=axis))[0]