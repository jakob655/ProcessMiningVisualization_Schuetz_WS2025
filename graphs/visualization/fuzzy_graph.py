import numpy as np

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
        event_data["SPM value"] = spm
        event_data["Frequency *(absolute)*"] = absolute_frequency
        event_data["Frequency *(normalized)*/Unary Significance"] = normalized_frequency
        label = f'<{title}<br/><font color="red">{absolute_frequency}</font>>'
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
            size: float = 1,
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
        color : str, optional
            color of the edge, by default "black"
        correlation: float, optional
            correlation of the edge
        significance: float, optional
            significance of the edge
        **edge_data
            additional data for the edge
        """
        avg = ""
        if color == "red":
            avg = "Average "
        edge_data[avg + "Frequency *(absolute)*"] = absolute_frequency
        edge_data[avg + "Frequency *(normalized)*"] = normalized_frequency
        if significance:
            edge_data[avg + "Binary Significance"] = round(significance, 2)
        if correlation:
            edge_data[avg + "Correlation"] = round(correlation, 2)
        super().add_edge(source, destination, absolute_frequency, penwidth=str(size),
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
        abs_freqs = [node.get("abs_freq", 0) for node in merged_nodes]
        avg_abs_freq = round((np.mean(abs_freqs))) if abs_freqs else 0
        cluster_data["Average Frequency *(normalized)*/Average Unary Significance"] = significance
        cluster_data["Average Frequency *(absolute)*"] = avg_abs_freq
        cluster_data["Nodes"] = merged_nodes
        width, height = size
        label = f"""<{cluster_name}<br/>{len(merged_nodes)} Elements<br/><font color="red">~{avg_abs_freq}</font>>"""
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

        if nodes := node.get_data_from_key("Nodes"):
            sig_str = f"{node.get_data_from_key("Average Frequency *(normalized)*/Average Unary Significance"):.2f}"
            description = f"\n**Average Frequency *(absolute)*:** {node.get_data_from_key("Average Frequency *(absolute)*")}"
            description += f"\n**Average Frequency *(normalized)*/Average Unary Significance:** {sig_str}"
            description += "\n\n**Clustered Nodes:**"
            for n in nodes:
                description += f"\n **{n['id']}**:\n"
                description += f"- SPM value: {n.get('spm', '–'):.2f}\n"
                description += f"- Frequency *(absolute)*: {n['abs_freq']}\n"
                description += f"- Frequency *(normalized)*/Unary Significance: {n['norm_freq']:.2f}\n"

        return node_id, description
