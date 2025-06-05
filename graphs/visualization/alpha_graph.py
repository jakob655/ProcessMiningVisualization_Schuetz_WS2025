from graphs.visualization.base_graph import BaseGraph


class AlphaGraph(BaseGraph):
    """A class to represent an AlphaGraph."""

    def __init__(self) -> None:
        """Initialize the AlphaGraph object."""
        super().__init__(rankdir="LR")
        self.adjacency = {}

    def create_edge(self, source: str, destination: str, weight: int = None, **edge_data) -> None:
        """Create an edge between two nodes.

        Parameters
        ----------
        source : str
            source node id
        destination : str
            destination node id
        weight : int, optional
            weight of the edge
        **edge_data
            additional data for the edge
        """
        self.adjacency.setdefault(source, []).append(destination)
        # Convert numerical attributes to strings if necessary
        edge_data = {key: str(value) if isinstance(value, (int, float)) else value for key, value in edge_data.items()}
        super().add_edge(source, destination, weight, **edge_data)

    def add_empty_circle(self, circle_id: str) -> None:
        """Add an empty circle node to the graph.

        Parameters
        ----------
        circle_id : str
            ID for the circle node
        """
        super().add_node(
            id=circle_id,
            label=" ",
            shape="circle",
            style="filled",
            fillcolor="#FDFFF5",
        )
