from collections import deque
from transformations.utils import cases_list_to_dict


class DFG:
    """Implementation of a Directly Follows Graph (DFG)"""

    def __init__(
        self, log: list[list[str]] | dict[tuple[str, ...], int] = None
    ) -> None:
        """Initialize the DFG.

        Parameters
        ----------
        log : list[list[str]] | dict[tuple[str,...],int], optional
            A list of traces or a dictionary containing the traces and their frequencies in the log
            If the log is a list of traces, it will be converted to a dictionary with the frequencies set to 1, by default None
        """
        self.start_nodes: set[str | int] = set()
        self.end_nodes: set[str | int] = set()
        self.successor_list: dict[str | int, set[str | int]] = dict()
        self.predecessor_list: dict[str | int, set[str | int]] = dict()
        if log:
            self.__build_graph_from_log(log)

    def __build_graph_from_log(
        self, log: list[list[str]] | dict[tuple[str, ...], int]
    ) -> None:
        """Build the DFG from a log. A edge is added between two events if they are directly followed in a trace.

        Parameters
        ----------
        log : list[list[str]] | dict[tuple[str, ...], int]
            A list of traces or a dictionary containing the traces and their frequencies in the log
        """
        _log = log
        if isinstance(log, list):
            _log = cases_list_to_dict(log)

        for trace, frequency in _log.items():
            if len(trace) == 0:
                continue

            self.start_nodes.add(trace[0])
            self.end_nodes.add(trace[-1])

            self.add_node(trace[0])
            self.add_node(trace[-1])

            for i in range(len(trace) - 1):
                self.add_edge(trace[i], trace[i + 1])

    def add_edge(self, source: str | int, destination: str | int) -> None:
        """Add an edge between two nodes.

        Parameters
        ----------
        source : str | int
            source node of the edge
        destination : str | int
            destination node of the edge
        """

        if not self.contains_node(source):
            self.add_node(source)

        if not self.contains_node(destination):
            self.add_node(destination)

        if source not in self.successor_list:
            self.successor_list[source] = set()

        if destination not in self.predecessor_list:
            self.predecessor_list[destination] = set()

        self.successor_list[source].add(destination)
        self.predecessor_list[destination].add(source)

    def add_node(self, node: str | int) -> None:
        """Add a node to the DFG.

        Parameters
        ----------
        node : str | int
            The node to add to the DFG
        """
        if node not in self.successor_list:
            self.successor_list[node] = set()

        if node not in self.predecessor_list:
            self.predecessor_list[node] = set()

    def get_connected_components(self) -> list[set[str | int]]:
        """Get the connected components of the DFG. A connected component is a set of nodes where each node is reachable from every other node in the set.
        The connected components are found using a breadth-first search. The search is done on the undirected graph.

        Returns
        -------
        list[set[str | int]]
            A list of connected components
        """
        connected_components = []
        visited = set()

        for node in self.get_nodes():
            if node not in visited:
                component = self.__bfs(node, directed=False)
                connected_components.append(component)
                visited.update(component)

        return connected_components

    def __bfs(self, starting_node: str | int, directed=True) -> set[str | int]:
        """Perform a breadth-first search on the graph starting from the starting node.

        Parameters
        ----------
        starting_node : str | int
            The node to start the search from
        directed : bool, optional
            If True, the search is done on the directed graph, otherwise on the undirected graph, by default True

        Returns
        -------
        set[str | int]
            A set of visited nodes
        """
        queue = deque([starting_node])
        visited = set([starting_node])

        while queue:
            current_node = queue.popleft()
            neighbors = self.get_successors(current_node)

            if not directed:
                neighbors = neighbors.union(self.get_predecessors(current_node))

            for neighbor in neighbors:
                if neighbor not in visited:
                    queue.append(neighbor)
                    visited.add(neighbor)

        return visited

    def get_successors(self, node: str | int) -> set[str | int]:
        """Get the successors of a node.

        Parameters
        ----------
        node : str | int
            The node for which the successors should be returned

        Returns
        -------
        set[str | int]
            A set of successors of the node
        """
        return self.successor_list.get(node, set())

    def get_predecessors(self, node: str | int) -> set[str | int]:
        """Get the predecessors of a node.

        Parameters
        ----------
        node : str | int
            The node for which the predecessors should be returned

        Returns
        -------
        set[str | int]
            A set of predecessors of the node
        """
        return self.predecessor_list.get(node, set())

    def get_nodes(self) -> set[str | int]:
        """Get all nodes in the DFG.

        Returns
        -------
        set[str | int]
            A set of all nodes in the DFG
        """
        return set(self.successor_list.keys())

    def get_edges(self) -> set[tuple[str | int, str | int]]:
        """Get all edges in the DFG.

        Returns
        -------
        set[tuple[str | int, str | int]]
            A set of all edges in the DFG

        """
        edges = set()
        for source, destinations in self.successor_list.items():
            for destination in destinations:
                edges.add((source, destination))
        return edges

    def get_start_nodes(self) -> set[str | int]:
        """Get all start nodes in the DFG.

        Returns
        -------
        set[str | int]
            A set of all start nodes in the DFG
        """
        return self.start_nodes

    def get_end_nodes(self) -> set[str | int]:
        """Get all end nodes in the DFG.

        Returns
        -------
        set[str | int]
            A set of all end nodes in the DFG
        """
        return self.end_nodes

    def is_in_start_nodes(self, node: str | int) -> bool:
        """Check if a node is in the start nodes of the DFG.

        Parameters
        ----------
        node : str | int
            The node to check

        Returns
        -------
        bool
            True if the node is in the start nodes, False otherwise
        """
        return node in self.start_nodes

    def is_in_end_nodes(self, node: str | int) -> bool:
        """Check if a node is in the end nodes of the DFG.

        Parameters
        ----------
        node : str | int
            The node to check

        Returns
        -------
        bool
            True if the node is in the end nodes, False otherwise
        """
        return node in self.end_nodes

    def contains_node(self, node: str | int) -> bool:
        """Check if a node is in the DFG.

        Parameters
        ----------
        node : str | int
            The node to check

        Returns
        -------
        bool
            True if the node is in the DFG, False otherwise
        """
        return node in self.successor_list.keys()

    def contains_edge(self, source: str | int, destination: str | int) -> bool:
        """Check if an edge exists between two nodes.

        Parameters
        ----------
        source : str | int
            source node of the edge
        destination : str | int
            destination node of the edge

        Returns
        -------
        bool
            True if the edge exists, False otherwise
        """
        return destination in self.successor_list.get(source, set())

    def get_reachable_nodes(self, node: str | int) -> set[str | int]:
        """Get all nodes that are reachable from a node.

        Parameters
        ----------
        node : str | int
            The node for which the reachable nodes should be returned

        Returns
        -------
        set[str | int]
            A set of reachable nodes
        """
        visited = self.__bfs(node)

        return visited

    def invert(self):
        """Invert the DFG. All double edges are removed and if no edge exists between two nodes or only one edge exists, a double edge is added.

        Returns
        -------
        DFG
            The inverted DFG
        """
        inverted_dfg = DFG()
        nodes = list(self.get_nodes())

        for node in nodes:
            inverted_dfg.add_node(node)

        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                node_1 = nodes[i]
                node_2 = nodes[j]

                if not self.contains_edge(node_1, node_2) or not self.contains_edge(
                    node_2, node_1
                ):
                    inverted_dfg.add_edge(node_1, node_2)
                    inverted_dfg.add_edge(node_2, node_1)

        return inverted_dfg

    def create_dfg_without_nodes(self, nodes: set[str | int]):
        """Create a new DFG without the specified nodes.

        Parameters
        ----------
        nodes : set[str  |  int]
            The nodes to remove from the DFG

        Returns
        -------
        DFG
            A new DFG without the specified nodes
        """
        dfg_without_nodes = DFG()

        for node in self.get_nodes():
            if node not in nodes:
                dfg_without_nodes.add_node(node)

        for edge in self.get_edges():
            source, destination = edge
            if source not in nodes and destination not in nodes:
                dfg_without_nodes.add_edge(source, destination)

        return dfg_without_nodes
