from ui.base_algorithm_ui.base_algorithm_view import BaseAlgorithmView
import streamlit as st
from components.number_input_slider import number_input_slider


class HeuristicMinerView(BaseAlgorithmView):
    """View for the Heuristic Miner algorithm."""

    def render_sidebar(
        self, sidebar_values: dict[str, tuple[int | float, int | float]]
    ) -> None:
        """Renders the sidebar for the Heuristic Miner algorithm.

        Parameters
        ----------
        sidebar_values : dict[str, tuple[int  |  float, int  |  float]]
            A dictionary containing the minimum and maximum values for the sidebar sliders.
            The keys of the dictionary are equal to the keys of the sliders.
        """

        number_input_slider(
            label="Minimum Frequency",
            min_value=sidebar_values["frequency"][0],
            max_value=sidebar_values["frequency"][1],
            key="frequency",
            help="Minimum frequency for displaying edges and nodes. Edges with a lower frequency (weight) will be removed. Node with a lower frequency will be removed.",
        )

        number_input_slider(
            label="Dependency Threshold",
            min_value=sidebar_values["threshold"][0],
            max_value=sidebar_values["threshold"][1],
            key="threshold",
            help="Minimum dependency for displaying edges. Edges with a lower dependency will be removed.",
        )

        number_input_slider(
            label="SPM Threshold",
            min_value=sidebar_values["spm_threshold"][0],
            max_value=sidebar_values["spm_threshold"][1],
            key="spm_threshold",
            help="Filter nodes based on the SPM metric threshold.",
        )