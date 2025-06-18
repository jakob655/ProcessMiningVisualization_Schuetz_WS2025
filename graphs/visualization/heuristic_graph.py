import math

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
            spm: float,
            frequency: float,
            absolute_frequency: int,
            size: tuple[int, int],
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
        absolute_frequency : int
            absolute frequency of the event
        size : tuple[int, int]
            size of the node, width and height
        **event_data
            additional data for the event
        """
        event_data["spm"] = spm
        event_data["frequency"] = frequency
        event_data["absolute_frequency"] = absolute_frequency
        rounded_freq = None
        if frequency:
            rounded_freq = math.ceil(frequency * 100) / 100
        label = f'<{title}<br/><font color="red">{rounded_freq:.2f}</font>>'
        width, height = size
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
            frequency: float = None,
            weight: int = None,
            color: str = "black",
            **edge_data
    ) -> None:
        """Create an edge between two nodes.

        Parameters
        ----------
        source : str
            soure node id
        destination : str
            destination node id
        size : float
            size/penwidth of the edge
        frequency : float
            frequency of the edge
        weight : int, optional
            weight of the edge, by default None
        color : str, optional
            color of the edge, by default "black"
        **edge_data
            additional data for the edge
        """
        # add dependency threshold for expander on-click
        rounded_freq = None
        if frequency:
            rounded_freq = math.ceil(frequency * 100) / 100
        edge_data["frequency"] = frequency
        edge_data["weight"] = weight
        super().add_edge(source, destination, rounded_freq, penwidth=str(size), color=color, data=edge_data)

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
        description = ""

        if spm := node.get_data_from_key("spm"):
            description = f"{description}\n**SPM value:** {spm}"

        if frequency := node.get_data_from_key("frequency"):
            description = f"{description}\n**Frequency:** {frequency}"

        if absolute_frequency := node.get_data_from_key("absolute_frequency"):
            description = f"""{description}\n**Absolute Frequency:** {absolute_frequency}"""

        return node.get_id(), description
