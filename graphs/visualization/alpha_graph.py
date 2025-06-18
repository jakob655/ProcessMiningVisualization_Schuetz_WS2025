import math

from graphs.visualization.base_graph import BaseGraph


class AlphaGraph(BaseGraph):
    """A class to represent an AlphaGraph."""

    def __init__(self) -> None:
        """Initialize the AlphaGraph object."""
        super().__init__(rankdir="LR")
        self.adjacency = {}

    def add_event(
            self,
            title: str,
            spm: float,
            frequency: float,
            **event_data,
    ) -> None:
        """Add an event to the graph.

        Parameters
        ----------
        title : str
            name of the event
        spm : float
            spm value of the event
        frequency : float
            frequency of the event
        **event_data
            additional data for the event
        """
        event_data["spm"] = spm
        event_data["frequency"] = frequency
        rounded_freq = None
        if frequency:
            rounded_freq = math.ceil(frequency * 100) / 100
        label = f'<{title}<br/><font color="red">{rounded_freq:.2f}</font>>'
        super().add_node(
            id=title,
            label=label,
            data=event_data,
            shape="circle",
            style="filled",
            fillcolor="#FDFFF5",
        )

    def create_edge(self, source: str, destination: str, frequency: float = None, color: str = "black",
                    **edge_data) -> None:
        """Create an edge between two nodes.

        Parameters
        ----------
        source : str
            source node id
        destination : str
            destination node id
        frequency : float
            frequency of the edge
        color : str, optional
            color of the edge, by default "black"
        **edge_data
            additional data for the edge
        """
        self.adjacency.setdefault(source, []).append(destination)

        rounded_freq = None
        if frequency:
            rounded_freq = math.ceil(frequency * 100) / 100
        edge_data["frequency"] = frequency
        super().add_edge(source, destination, rounded_freq, color=color, data=edge_data)

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

    def node_to_string(self, id: str) -> tuple[str, str]:
        """Return the node name/id and description for the given node id.

        Parameters
        ----------
        id : str
            node id

        Returns
        -------
        tuple[str, str]
            node name/id and description. The description contains the node name, spm value and frequency.
        """
        node = self.get_node(id)
        description = ""

        if spm := node.get_data_from_key("spm"):
            description = f"{description}\n**SPM value:** {spm}"

        if frequency := node.get_data_from_key("frequency"):
            description = f"{description}\n**Frequency:** {frequency}"

        return node.get_id(), description
