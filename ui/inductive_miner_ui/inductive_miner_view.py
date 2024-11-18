from ui.base_algorithm_ui.base_algorithm_view import BaseAlgorithmView
import streamlit as st
from components.number_input_slider import number_input_slider


class InductiveMinerView(BaseAlgorithmView):
    """View for the Inductive Miner algorithm."""

    def render_sidebar(
        self, sidebar_values: dict[str, tuple[int | float, int | float]]
    ) -> None:
        """Renders the sidebar for the Inductive Miner algorithm.

        Parameters
        ----------
        sidebar_values : dict[str, tuple[int  |  float, int  |  float]]
            A dictionary containing the minimum and maximum values for the sidebar sliders.
            The keys of the dictionary are equal to the keys of the sliders.
        """
        number_input_slider(
            label="Traces Threshold",
            min_value=sidebar_values["traces_threshold"][0],
            max_value=sidebar_values["traces_threshold"][1],
            key="traces_threshold",
            help="""The traces threshold parameter determines the minimum frequency of a trace to be included in the graph. 
            All traces with a frequency that is lower than treshold * max_trace_frequency will be removed. The higher the value, the less traces will be included in the graph.""",
        )

        number_input_slider(
            label="Activity Threshold",
            min_value=sidebar_values["activity_threshold"][0],
            max_value=sidebar_values["activity_threshold"][1],
            key="activity_threshold",
            help="""The activity threshold parameter determines the minimum frequency of an activity to be included in the graph. 
            All activities with a frequency that is lower than treshold * max_event_frequency will be removed.
            The higher the value, the less activities will be included in the graph.""",
        )
