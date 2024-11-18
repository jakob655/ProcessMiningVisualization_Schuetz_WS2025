from ui.base_algorithm_ui.base_algorithm_view import BaseAlgorithmView
import streamlit as st


class AlgorithmViewTemplate(BaseAlgorithmView):
    """Template for creating views for the algorithm."""

    def render_sidebar(
        self, sidebar_values: dict[str, tuple[int | float, int | float]]
    ) -> None:
        """Renders the sidebar for the mining algorithm.

        Parameters
        ----------
        sidebar_values : dict[str, tuple[int  |  float, int  |  float]]
            A dictionary containing the minimum and maximum values for the sidebar sliders.
            The keys of the dictionary are equal to the keys of the sliders.
        """
        # Add the sliders for the algorithm here using the sidebar_values dictionary
        # Example:
        # st.slider(
        #     label="Minimum Frequency",
        #     min_value=sidebar_values["frequency"][0],
        #     max_value=sidebar_values["frequency"][1],
        #     key="frequency",
        #     help="Minimum frequency for displaying edges and nodes. Edges with a lower frequency (weight) will be removed. Node with a lower frequency will be removed.",
        # )

        raise NotImplementedError("Method render_sidebar must be implemented")
