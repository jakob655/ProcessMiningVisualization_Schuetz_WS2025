from ui.base_algorithm_ui.base_algorithm_view import BaseAlgorithmView
import streamlit as st
from components.number_input_slider import number_input_slider


class AlphaMinerView(BaseAlgorithmView):
    """View for the Alpha Miner algorithm."""

    def render_sidebar(self, sidebar_values: dict[str, tuple[int | float, int | float]]) -> None:
        """Renders the sidebar for the Alpha Miner algorithm.

        Parameters
        ----------
        sidebar_values : dict[str, tuple[int  |  float, int  |  float]]
            A dictionary containing the minimum and maximum values for the sidebar sliders.
            The keys of the dictionary are equal to the keys of the sliders.
        """

        number_input_slider(
            label="SPM Threshold",
            min_value=sidebar_values["spm_threshold"][0],
            max_value=sidebar_values["spm_threshold"][1],
            key="spm_threshold",
            help="Filter nodes based on the SPM metric threshold.",
        )