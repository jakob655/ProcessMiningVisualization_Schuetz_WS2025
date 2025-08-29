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
        rounded_freq = None
        if normalized_frequency:
            rounded_freq = round(normalized_frequency, 2)
        event_data["SPM value"] = spm
        event_data["Frequency/Unary Significance *(normalized)*"] = normalized_frequency
        event_data["Frequency *(absolute)*"] = absolute_frequency
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
            color: str = "black",
            correlation: float = None,
            significance: float = None,
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
        correlation: float, optional
            correlation of the edge
        significance: float, optional
            significance of the edge
        color : str, optional
            color of the edge, by default "black"
        **edge_data
            additional data for the edge
        """
        if normalized_frequency:
            normalized_frequency = round(normalized_frequency, 2)
        edge_data["Frequency *(normalized)*"] = normalized_frequency
        edge_data["Frequency *(absolute)*"] = absolute_frequency
        if significance:
            edge_data["Significance"] = round(significance, 2)
        if correlation:
            edge_data["Correlation"] = round(correlation, 2)
        super().add_edge(source, destination, normalized_frequency, penwidth=str(normalized_frequency),
                         color=color,
                         data=edge_data)

    def add_cluster(
            self,
            cluster_name: str,
            significance: int | float,
            size: tuple[int, int],
            merged_nodes: list[dict[str, str | int | float]],
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
        merged_nodes : list[dict[str, str | int | float]]
            list of dictionaries of nodes merged in the cluster
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
        node_id, description = super().node_to_string(id)
        node = self.get_node(id)

        if nodes := node.get_data_from_key("nodes"):
            sig_str = f"{node.get_data_from_key("significance"):.2f}"
            description = f"\n**Significance:** {sig_str}"
            description += "\n\n**Clustered Nodes:**"
            for n in nodes:
                description += f"\n **{n['id']}**:\n"
                description += f"- SPM value: {n.get('spm', '–'):.2f}\n"
                description += f"- Frequency/Unary Significance *(normalized)*: {n['norm_freq']:.2f}\n"
                description += f"- Frequency *(absolute)*: {n['abs_freq']}\n"

        return node_id, description