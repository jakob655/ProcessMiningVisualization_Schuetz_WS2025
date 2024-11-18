from ui.base_algorithm_ui.base_algorithm_controller import BaseAlgorithmController
from ui.inductive_miner_ui.inductive_miner_view import InductiveMinerView
from mining_algorithms.inductive_mining import InductiveMining
import streamlit as st


class InductiveMinerController(BaseAlgorithmController):
    """Controller for the Inductive Miner algorithm."""

    def __init__(
        self, views=None, mining_model_class=None, dataframe_transformations=None
    ):
        """Initializes the controller for the Inductive Miner algorithm.

        Parameters
        ----------
        views : List[BaseAlgorithmView] | BaseAlgorithmView, optional
            The views for the Inductive Miner algorithm. If None is passed, the default view is used, by default None
        mining_model_class : MiningInterface Class, optional
            The mining model class for the Inductive Miner algorithm. If None is passed, the default model class is used, by default None
        dataframe_transformations : DataframeTransformations, optional
            The class for the dataframe transformations. If None is passed, a new instance is created, by default None
        """
        if views is None:
            views = [InductiveMinerView()]

        if mining_model_class is None:
            mining_model_class = InductiveMining
        super().__init__(views, mining_model_class, dataframe_transformations)

    def get_page_title(self) -> str:
        """Returns the page title.

        Returns
        -------
        str
            The page title.
        """
        return "Inductive Mining"

    def process_algorithm_parameters(self):
        """Processes the algorithm parameters from the session state. The parameters are set to the instance variables.
        If the parameters are not set in the session state, the default values are used.
        """
        # set session state from instance variables if not set
        if "traces_threshold" not in st.session_state:
            st.session_state.traces_threshold = self.mining_model.get_traces_threshold()

        if "activity_threshold" not in st.session_state:
            st.session_state.activity_threshold = (
                self.mining_model.get_activity_threshold()
            )

        # set instance variables from session state
        self.traces_threshold = st.session_state.traces_threshold
        self.activity_threshold = st.session_state.activity_threshold

    def perform_mining(self) -> None:
        """Performs the mining of the Inductive Miner algorithm."""
        self.mining_model.generate_graph(self.activity_threshold, self.traces_threshold)

    def have_parameters_changed(self) -> bool:
        """Checks if the algorithm parameters have changed.

        Returns
        -------
        bool
            True if the algorithm parameters have changed, False otherwise.
        """
        return (
            self.mining_model.get_activity_threshold() != self.activity_threshold
            or self.mining_model.get_traces_threshold() != self.traces_threshold
        )

    def get_sidebar_values(self) -> dict[str, tuple[int | float, int | float]]:
        """Returns the sidebar values for the Inductive Miner algorithm.

        Returns
        -------
        dict[str, tuple[int | float, int | float]]
            A dictionary containing the minimum and maximum values for the sidebar sliders.
            The keys of the dictionary are equal to the keys of the sliders.
        """
        sidebar_values = {
            "traces_threshold": (0.0, 1.0),
            "activity_threshold": (0.0, 1.0),
        }

        return sidebar_values
