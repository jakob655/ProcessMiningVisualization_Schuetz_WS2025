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
            normalized_frequency: float,
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
        normalized_frequency : float
            normalized frequency of the event
        absolute_frequency : int
            absolute frequency of the event
        size : tuple[int, int]
            size of the node, width and height
        **event_data
            additional data for the event
        """
        event_data["SPM value"] = spm
        event_data["Frequency *(absolute)*"] = absolute_frequency
        event_data["Frequency *(normalized)*"] = normalized_frequency
        label = f'<{title}<br/><font color="red">{absolute_frequency}</font>>'
        width, height = size
        super().add_node(
            id=title,
            label=label,
            data=event_data,
            width=str(width),
            height=str(height),
            shape="box",
            style="rounded, filled",
            fillcolor="#FDFFF5",
        )

    def create_edge(
            self,
            source: str,
            destination: str,
            size: float,
            normalized_frequency: float = None,
            absolute_frequency: int = None,
            dependency_score: float = None,
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
        normalized_frequency : float, optional
            normalized frequency of the edge, by default None
        absolute_frequency : int, optional
            absolute frequency of the edge, by default None
        dependency_score : float, optional
            dependency score of the edge
        color : str, optional
            color of the edge, by default "black"
        **edge_data
            additional data for the edge
        """
        edge_data["Frequency *(absolute)*"] = absolute_frequency
        edge_data["Frequency *(normalized)*"] = normalized_frequency
        if dependency_score is not None:
            edge_data["Dependency score"] = round(dependency_score, 3)
        super().add_edge(source, destination, absolute_frequency, penwidth=str(size), color=color, data=edge_data)
