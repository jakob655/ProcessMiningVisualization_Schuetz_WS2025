from graphviz import Digraph
import numpy as np
from mining_algorithms.ddcal_clustering import DensityDistributionClusterAlgorithm


def calculate_degree(node, events, succession_matrix):
    index = events.index(node)
    in_degree = sum(succession_matrix[:, index])
    out_degree = sum(succession_matrix[index, :])
    return in_degree + out_degree


def calculate_spm(node, events, node_frequency, A, L, succession_matrix):
    degree = calculate_degree(node, events, succession_matrix)
    frequency = node_frequency
    spm = 1 - ((degree * A) / (frequency * L))
    return spm


class HeuristicMining():
    def __init__(self, log):
        self.log = log
        self.events, self.appearence_frequency = self.__filter_out_all_events()
        self.succession_matrix = self.__create_succession_matrix()
        self.dependency_matrix = self.__create_dependency_matrix()

        # Graph modifiers
        self.min_edge_thickness = 1
        # self.max_node_size = 10
        self.min_node_size = 1.5
        self.min_frequency = 1
        self.dependency_threshold = 0.5
        self.spm_threshold = 0.5

    def create_dependency_graph_with_graphviz(self, dependency_threshold, spm_threshold, min_frequency):
        dependency_graph = self.__create_dependency_graph(dependency_threshold, min_frequency)
        start_nodes = self.__get_start_nodes()
        end_nodes = self.__get_end_nodes()
        self.dependency_threshold = dependency_threshold
        self.min_frequency = min_frequency
        self.spm_threshold = spm_threshold

        # create graph
        graph = Digraph()
        # cluster the node sizes based on frequency
        cluster = DensityDistributionClusterAlgorithm(list(self.appearence_frequency.values()))
        freq_sorted = list(cluster.sorted_data)
        freq_labels_sorted = list(cluster.labels_sorted_data)

        _A, _L = self.calculate_A_and_L()

        filtered_nodes = []
        for node in self.events:
            node_freq = self.appearence_frequency.get(node)
            spm = calculate_spm(node, self.events, node_freq, _A, _L, self.succession_matrix)
            if spm >= spm_threshold:
                filtered_nodes.append(node)

        # add nodes to graph
        for node in filtered_nodes:
            node_freq = self.appearence_frequency.get(node)
            w = freq_labels_sorted[freq_sorted.index(node_freq)] / 2 + self.min_node_size
            h = w / 3
            graph.node(str(node), label=f"<{node}<BR/><font color='red'>{node_freq}</font>>", width=str(w), height=str(h), shape="box",
                       style="filled", fillcolor='#FDFFF5')

        # cluster the edge thickness sizes based on frequency
        edge_frequencies = self.dependency_matrix.flatten()
        edge_frequencies = edge_frequencies[edge_frequencies >= 0.0]
        edge_frequencies = np.unique(edge_frequencies)
        # print(edge_frequencies)
        cluster = DensityDistributionClusterAlgorithm(edge_frequencies)
        freq_sorted = list(cluster.sorted_data)
        freq_labels_sorted = list(cluster.labels_sorted_data)

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
                        edge_thickness = freq_labels_sorted[
                                             freq_sorted.index(self.dependency_matrix[i][j])] + self.min_edge_thickness

                    graph.edge(str(filtered_nodes[i]), str(filtered_nodes[j]), penwidth=str(edge_thickness),
                               label=str(int(self.succession_matrix[i][j])))

                if j == len(filtered_nodes) - 1 and column_total == 0 and filtered_nodes[i] not in end_nodes:
                    end_nodes.append(filtered_nodes[i])
                if j == len(filtered_nodes) - 1 and row_total == 0 and filtered_nodes[i] not in start_nodes:
                    start_nodes.append(filtered_nodes[i])

        # add start node
        graph.node("start", label="start", shape='doublecircle', style='filled', fillcolor='green')
        for node in start_nodes:
            if node in filtered_nodes:
                node_freq = self.appearence_frequency.get(node)
                graph.node(str(node), label=f"<{node}<BR/><font color='red'>{node_freq}</font>>", shape="box", style="filled", fillcolor='#FDFFF5')
                graph.edge("start", str(node), penwidth=str(0.1))

        # add end node
        graph.node("end", label="end", shape='doublecircle', style='filled', fillcolor='red')
        for node in end_nodes:
            if node in filtered_nodes:
                node_freq = self.appearence_frequency.get(node)
                graph.node(str(node), label=f"<{node}<BR/><font color='red'>{node_freq}</font>>", shape="box", style="filled", fillcolor='#FDFFF5')
                graph.edge(str(node), "end", penwidth=str(0.1))

        return graph

    def filter_log(self, filtered_nodes):
        filtered_log = []
        for trace in self.log:
            filtered_trace = [event for event in trace if event in filtered_nodes]
            filtered_log.append(filtered_trace)
        return filtered_log

    def calculate_A_and_L(self):
        activities = set()
        total_events = 0
        for trace in self.log:
            total_events += len(trace)
            activities.update(trace)
        return len(activities), total_events

    def get_max_frequency(self):
        max_freq = 0
        for value in list(self.appearence_frequency.values()):
            if value > max_freq:
                max_freq = value
        return max_freq

    def get_min_frequency(self):
        return self.min_frequency

    def get_threshold(self):
        return self.dependency_threshold

    def __filter_out_all_events(self):
        dic = {}
        for trace in self.log:
            for activity in trace:
                if activity in dic:
                    dic[activity] = dic[activity] + 1
                else:
                    dic[activity] = 1

        activities = list(dic.keys())
        return activities, dic

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

    def __get_start_nodes(self):
        # a start node is a node where an entire column in the succession_matrix is 0.
        start_nodes = []
        for case in self.log:
            if case[0] not in start_nodes:
                start_nodes.append(case[0])

        return start_nodes

    def __get_end_nodes(self):
        # an end node is a node where an entire row in the succession_matrix is 0.
        end_nodes = []
        for case in self.log:
            last_index = len(case) - 1
            if case[last_index] not in end_nodes:
                end_nodes.append(case[last_index])

        return end_nodes

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
