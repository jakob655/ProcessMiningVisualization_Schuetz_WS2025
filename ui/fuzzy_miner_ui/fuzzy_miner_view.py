from ui.base_algorithm_ui.base_algorithm_view import BaseAlgorithmView
import streamlit as st
from components.number_input_slider import number_input_slider


class FuzzyMinerView(BaseAlgorithmView):
    """View for the Fuzzy Miner algorithm."""

    def render_sidebar(
        self, sidebar_values: dict[str, tuple[int | float, int | float]]
    ) -> None:
        """Renders the sidebar for the Fuzzy Miner algorithm.

        Parameters
        ----------
        sidebar_values : dict[str, tuple[int  |  float, int  |  float]]
            A dictionary containing the minimum and maximum values for the sidebar sliders.
            The keys of the dictionary are equal to the keys of the sliders.
        """
        st.write("Significance Cutoff")

        number_input_slider(
            label="Significance",
            min_value=sidebar_values["significance"][0],
            max_value=sidebar_values["significance"][1],
            key="significance",
            help="Significance measures the frequency of events that are observed more frequently and are therefore considered more significant.",
        )

        number_input_slider(
            label="Correlation",
            min_value=sidebar_values["correlation"][0],
            max_value=sidebar_values["correlation"][1],
            key="correlation",
            help="Correlation measures how closely related two events following one another are.",
        )

        st.write("Edge filtering")

        number_input_slider(
            label="Edge Cutoff",
            min_value=sidebar_values["edge_cutoff"][0],
            max_value=sidebar_values["edge_cutoff"][1],
            key="edge_cutoff",
            help="The edge cutoff parameter determines the aggressiviness of the algorithm, i.e. the higher its value, the more likely the algorithm remove edges.",
        )

        number_input_slider(
            label="Utility Ratio",
            min_value=sidebar_values["utility_ratio"][0],
            max_value=sidebar_values["utility_ratio"][1],
            key="utility_ratio",
            help="A configuratable utility ratio determines the weight and a larger value for utility ratio will perserve more significant edges, while a smaller value will favor highly correlated edges",
        )
