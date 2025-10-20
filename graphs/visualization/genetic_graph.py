from graphs.visualization.base_graph import BaseGraph


class GeneticGraph(BaseGraph):
    """A class to represent an GeneticGraph."""

    def __init__(self) -> None:
        """Initialize the GeneticGraph object."""
        super().__init__(rankdir="LR")
        self.adjacency = {}

    def add_event(
            self,
            title: str,
            spm: float,
            normalized_frequency: float,
            absolute_frequency: int,
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
        **event_data
            additional data for the event
        """
        event_data["SPM value"] = spm
        event_data["Frequency *(absolute)*"] = absolute_frequency
        event_data["Frequency *(normalized)*"] = normalized_frequency
        label = f'<{title}<br/><font color="red">{absolute_frequency}</font>>'
        super().add_node(
            id=title,
            label=label,
            data=event_data,
            shape="circle",
            style="filled",
            fillcolor="#FDFFF5",
        )

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

    def add_place(self, place_id: str) -> None:
        """Add a place node to the graph.

        Parameters
        ----------
        place_id : str
            ID for the place node
        """
        super().add_node(
            id=place_id,
            label=" ",
            shape="circle",
            style="filled",
            fillcolor="#E1E1E1",
        )
    
    def add_silent_transition(self, transition_id: str) -> None:
        """Add a silent (tau) transition to the graph."""
        super().add_node(
            id=transition_id,
            label="tau",
            shape="box",
            style="filled",
            fillcolor="#EDEDED",
        )
