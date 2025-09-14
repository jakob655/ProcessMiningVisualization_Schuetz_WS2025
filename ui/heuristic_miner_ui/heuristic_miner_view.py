import streamlit as st

from components.number_input_slider import number_input_slider
from ui.base_algorithm_ui.base_algorithm_view import BaseAlgorithmView


class HeuristicMinerView(BaseAlgorithmView):
    """View for the Heuristic Miner algorithm."""

    def render_log_filter_extensions(self, sidebar_values: dict[str, any]) -> None:
        st.write("### **Edge Filtering**")

        number_input_slider(
            label="Edge Frequency (normalized)",
            min_value=sidebar_values["edge_frequency_threshold_normalized"][0],
            max_value=sidebar_values["edge_frequency_threshold_normalized"][1],
            key="edge_freq_threshold_normalized",
            help="Filter edges based on normalized frequency (0–1).",
        )

        number_input_slider(
            label="Edge Frequency (absolute)",
            min_value=sidebar_values["edge_frequency_threshold_absolute"][0],
            max_value=sidebar_values["edge_frequency_threshold_absolute"][1],
            key="edge_freq_threshold_absolute",
            help="Filter edges based on absolute frequency (event counts).",
        )

        number_input_slider(
            label="Dependency Threshold",
            min_value=sidebar_values["threshold"][0],
            max_value=sidebar_values["threshold"][1],
            key="threshold",
            help="Minimum dependency for displaying edges. Edges with a lower dependency will be removed.",
        )
