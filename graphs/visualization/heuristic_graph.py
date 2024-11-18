from graphs.visualization.base_graph import BaseGraph


class HeuristicGraph(BaseGraph):
    """A class to represent a HeuristicGraph."""

    def __init__(
        self,
    ) -> None:
        """Initialize the HeuristicGraph object."""
        super().__init__(rankdir="TB")

    def add_event(
        self,
        title: str,
        frequency: int,
        size: tuple[int, int],
        **event_data,
    ) -> None:
        """Add an event to the graph.

        Parameters
        ----------
        title : str
            name of the event
        frequency : int
            frequency of the event
        size : tuple[int, int]
            size of the node, width and height
        **event_data
            additional data for the event
        """
        event_data["frequency"] = frequency
        width, height = size
        label = f"{title}\n{frequency}"
        super().add_node(
            id=title,
            label=label,
            data=event_data,
            width=str(width),
            height=str(height),
            shape="box",
            style="rounded, filled",
            fillcolor="#FFFFFF",
        )

    def create_edge(
        self,
        source: str,
        destination: str,
        size: float,
        weight: int = None,
    ) -> None:
        """Create an edge between two nodes.

        Parameters
        ----------
        source : str
            source node id
        destination : str
            destination node id
        size : float
            edge size/ penwidth
        weight : int, optional
            weight of the edge
        """
        super().add_edge(source, destination, weight, penwidth=str(size))

    def node_to_string(self, id: str) -> tuple[str, str]:
        """Return the node name/id and description for the given node id.

        Parameters
        ----------
        id : str
            id of the node

        Returns
        -------
        tuple[str, str]
            node name/id and description.
        """
        node = self.get_node(id)
        description = f"**Event:** {node.get_id()}"
        if frequency := node.get_data_from_key("frequency"):
            description = f"""{description}\n**Frequency:** {frequency}"""
        return node.get_id(), description
