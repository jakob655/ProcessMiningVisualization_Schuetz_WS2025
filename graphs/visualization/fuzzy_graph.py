from graphs.visualization.base_graph import BaseGraph
import math


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
            frequency: float,
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
        frequency : float
            frequency of the event
        significance : int
            significance of the event
        size : tuple[int, int]
            size of the node, width and height
        **event_data
            additional data for the event
        """
        event_data["spm"] = spm
        event_data["frequency"] = frequency
        event_data["significance"] = significance
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
            style="filled",
            fillcolor="#FDFFF5",
        )

    def create_edge(
            self,
            source: str,
            destination: str,
            frequency: float = None,
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
        # add binary significance for expander on-click
        rounded_freq = None
        if frequency:
            rounded_freq = math.ceil(frequency * 100) / 100
        edge_data["frequency"] = frequency
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

        if spm := node.get_data_from_key("spm"):
            description = f"{description}\n**SPM value:** {spm}"

        if frequency := node.get_data_from_key("frequency"):
            description = f"{description}\n**Frequency:** {frequency}"

        if significance := node.get_data_from_key("significance"):
            description = f"""{description}\n**Significance:** {significance}"""

        if nodes := node.get_data_from_key("nodes"):
            description = f"""{description}\n**Clustered Nodes:** {", ".join(nodes)}"""

        return node.get_id(), description