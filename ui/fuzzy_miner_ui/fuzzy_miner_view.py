import streamlit as st

from components.number_input_slider import number_input_slider
from ui.base_algorithm_ui.base_algorithm_view import BaseAlgorithmView


class FuzzyMinerView(BaseAlgorithmView):
    """View for the Genetic Miner algorithm."""

    def render_log_filter_extensions(self, sidebar_values: dict[str, any]) -> None:
        st.write("### **Node Filtering**")

        number_input_slider(
            label="Unary Significance",
            min_value=sidebar_values["unary_significance"][0],
            max_value=sidebar_values["unary_significance"][1],
            key="unary_significance",
            help="Filters nodes by how frequently they appear in the log. Nodes with a significance below this threshold are considered less important and are subject to removal or clustering.",
        )

        number_input_slider(
            label="Correlation",
            min_value=sidebar_values["correlation"][0],
            max_value=sidebar_values["correlation"][1],
            key="correlation",
            help="Correlation measures how closely related two events following one another are.",
        )

        st.write("### **Edge Filtering**")

        number_input_slider(
            label="Edge Frequency",
            min_value=sidebar_values["edge_frequency_threshold"][0],
            max_value=sidebar_values["edge_frequency_threshold"][1],
            key="edge_frequency_threshold",
            help="Filter edges based on their frequency.",
        )

        number_input_slider(
            label="Binary Significance",
            min_value=sidebar_values["binary_significance"][0],
            max_value=sidebar_values["binary_significance"][1],
            key="binary_significance",
            help="Filters edges based on how significant their transitions are. Only edges with a binary significance above this threshold will be considered in the edge filtering and utility computation",
        )

        number_input_slider(
            label="Edge Cutoff",
            min_value=sidebar_values["edge_cutoff"][0],
            max_value=sidebar_values["edge_cutoff"][1],
            key="edge_cutoff",
            help="The edge cutoff parameter determines the aggressiveness of the algorithm, i.e. the higher its value, the more likely the algorithm remove edges.",
        )

        number_input_slider(
            label="Utility Ratio",
            min_value=sidebar_values["utility_ratio"][0],
            max_value=sidebar_values["utility_ratio"][1],
            key="utility_ratio",
            help="A configuratable utility ratio determines the weight and a larger value for utility ratio will perserve more significant edges, while a smaller value will favor highly correlated edges",
        )
