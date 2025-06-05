from ui.base_algorithm_ui.base_algorithm_view import BaseAlgorithmView
import streamlit as st
from components.number_input_slider import number_input_slider


class FuzzyMinerView(BaseAlgorithmView):
    """View for the Fuzzy Miner algorithm."""

    def render_node_filter_extensions(self, sidebar_values: dict[str, any]) -> None:
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

    def render_edge_filter_extensions(self, sidebar_values: dict[str, any]) -> None:
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
