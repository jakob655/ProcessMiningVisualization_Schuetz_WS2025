from ui.base_algorithm_ui.base_algorithm_view import BaseAlgorithmView
import streamlit as st
from components.number_input_slider import number_input_slider


class HeuristicMinerView(BaseAlgorithmView):
    """View for the Heuristic Miner algorithm."""

    def render_edge_filter_extensions(self, sidebar_values: dict[str, any]) -> None:
        number_input_slider(
            label="Dependency Threshold",
            min_value=sidebar_values["threshold"][0],
            max_value=sidebar_values["threshold"][1],
            key="threshold",
            help="Minimum dependency for displaying edges. Edges with a lower dependency will be removed.",
        )

        number_input_slider(
            label="Minimum Frequency",
            min_value=sidebar_values["frequency"][0],
            max_value=sidebar_values["frequency"][1],
            key="frequency",
            help="Minimum frequency for displaying edges and nodes. Edges with a lower frequency (weight) will be removed.",
        )
