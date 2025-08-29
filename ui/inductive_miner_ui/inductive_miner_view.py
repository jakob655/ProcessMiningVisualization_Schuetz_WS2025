from components.number_input_slider import number_input_slider
from ui.base_algorithm_ui.base_algorithm_view import BaseAlgorithmView


class InductiveMinerView(BaseAlgorithmView):
    """View for the Inductive Miner algorithm."""

    def render_log_filter_extensions(self, sidebar_values: dict[str, any]) -> None:

        number_input_slider(
            label="Traces Threshold",
            min_value=sidebar_values["traces_threshold"][0],
            max_value=sidebar_values["traces_threshold"][1],
            key="traces_threshold",
            help="""The traces threshold parameter determines the minimum frequency of a trace to be included in the graph. 
                    All traces with a frequency that is lower than treshold * max_trace_frequency will be removed. The higher the value, the less traces will be included in the graph.""",
        )