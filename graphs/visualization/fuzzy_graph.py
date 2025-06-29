import math

from graphs.visualization.base_graph import BaseGraph


class FuzzyGraph(BaseGraph):
    """A class to represent a FuzzyGraph."""

    def __init__(
        self,
    ) -> None:
        """Initialize the FuzzyGraph object."""
        super().__init__(rankdir="TB")

    def add_event(
            self,
            title: str,
            spm: float,
            normalized_frequency: float,
            absolute_frequency: int,
            significance: int,
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
        significance : int
            significance of the event
        size : tuple[int, int]
            size of the node, width and height
        **event_data
            additional data for the event
        """
        event_data["SPM value"] = spm
        event_data["Frequency *(normalized)*"] = normalized_frequency
        event_data["Frequency *(absolute)*"] = absolute_frequency
        event_data["Significance"] = significance
        rounded_freq = None
        if normalized_frequency:
            rounded_freq = math.ceil(normalized_frequency * 100) / 100
        label = f'<{title}<br/><font color="red">{rounded_freq:.2f}</font>>'
        width, height = size
        super().add_node(
            id=title,
            label=label,
            data=event_data,
            width=str(width),
            height=str(height),
            shape="box",
            style="filled",
            fillcolor="#FDFFF5",
        )

    def create_edge(
            self,
            source: str,
            destination: str,
            normalized_frequency: float = None,
            absolute_frequency: int = None,
            significance: float = None,
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
            normalized frequency of the edge
        absolute_frequency : float, optional
            absolute frequency of the edge
        significance : float, optional
            significance of the edge
        color : str, optional
            color of the edge, by default "black"
        **edge_data
            additional data for the edge
        """
        rounded_freq = None
        if normalized_frequency:
            rounded_freq = math.ceil(normalized_frequency * 100) / 100
        edge_data["Frequency *(normalized)*"] = normalized_frequency
        edge_data["Frequency *(absolute)*"] = absolute_frequency
        edge_data["Significance"] = significance
        super().add_edge(source, destination, rounded_freq, penwidth=str(rounded_freq), color=color, data=edge_data)

    def add_cluster(
        self,
        cluster_name: str,
        significance: int | float,
        size: tuple[int, int],
        merged_nodes: list[str],
        **cluster_data: dict[str, str | int | float],
    ) -> None:
        """Add a cluster to the graph.

        Parameters
        ----------
        cluster_name : str
            name of the cluster
        significance : int | float
            average significance of the cluster
        size : tuple[int, int]
            size of the node, width and height
        merged_nodes : list[str]
            list of nodes merged in the cluster
        """
        cluster_data["significance"] = significance
        cluster_data["nodes"] = merged_nodes
        width, height = size
        label = f"{cluster_name}\n{len(merged_nodes)} Elements\n~{significance}"
        super().add_node(
            id=cluster_name,
            label=label,
            data=cluster_data,
            shape="octagon",
            style="filled",
            fillcolor="#6495ED",
            width=str(width),
            height=str(height),
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
            node name/id and description. The description contains the node name and significance.
            If the node is a cluster, it also contains the list of nodes merged in the cluster.
        """
        node = self.get_node(id)
        node_name = node.get_id()
        description = ""
        if "cluster" in node_name.lower():
            description = f"**Cluster:** {node_name}"

        if spm := node.get_data_from_key("SPM value"):
            spm = f"{float(spm):.2f}"
            description = f"{description}\n**SPM value:** {spm}"

        if normalized_frequency := node.get_data_from_key("Frequency *(normalized)*"):
            normalized_frequency = f"{float(normalized_frequency):.2f}"
            description = f"{description}\n**Frequency *(normalized)*:** {normalized_frequency}"

        if absolute_frequency := node.get_data_from_key("Frequency *(absolute)*"):
            description = f"{description}\n**Frequency *(absolute)*:** {absolute_frequency}"

        if significance := node.get_data_from_key("Significance"):
            significance = f"{float(significance):.2f}"
            description = f"""{description}\n**Significance:** {significance}"""

        if nodes := node.get_data_from_key("nodes"):
            description = f"""{description}\n**Clustered Nodes:** {", ".join(nodes)}"""

        return node.get_id(), description
