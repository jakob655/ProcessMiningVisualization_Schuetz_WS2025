import streamlit as st

from components.number_input_slider import number_input_slider
from ui.base_algorithm_ui.base_algorithm_view import BaseAlgorithmView


class HeuristicMinerView(BaseAlgorithmView):
    """View for the Heuristic Miner algorithm."""

    def render_log_filter_extensions(self, sidebar_values: dict[str, any]) -> None:
        st.write("### **Edge Filtering**")

        number_input_slider(
            label="Edge Frequency",
            min_value=sidebar_values["edge_frequency_threshold"][0],
            max_value=sidebar_values["edge_frequency_threshold"][1],
            key="edge_frequency_threshold",
            help="Filter edges based on their frequency.",
        )

        number_input_slider(
            label="Dependency Threshold",
            min_value=sidebar_values["threshold"][0],
            max_value=sidebar_values["threshold"][1],
            key="threshold",
            help="Minimum dependency for displaying edges. Edges with a lower dependency will be removed.",
        )