from ui.base_algorithm_ui.base_algorithm_controller import BaseAlgorithmController
from ui.heuristic_miner_ui.heuristic_miner_view import HeuristicMinerView
import streamlit as st
from mining_algorithms.heuristic_mining import HeuristicMining


class HeuristicMinerController(BaseAlgorithmController):
    """Controller for the Heuristic Miner algorithm."""

    def __init__(
        self, views=None, mining_model_class=None, dataframe_transformations=None
    ):
        """Initializes the controller for the Heuristic Miner algorithm.

        Parameters
        ----------
        views : List[BaseAlgorithmView] | BaseAlgorithmView, optional
            The views for the Heuristic Miner algorithm. If None is passed, the default view is used, by default None
        mining_model_class : MiningInterface Class, optional
            The mining model class for the Heuristic Miner algorithm. If None is passed, the default model class is used, by default None
        dataframe_transformations : DataframeTransformations, optional
            The class for the dataframe transformations. If None is passed, a new instance is created, by default None
        """
        if views is None:
            views = [HeuristicMinerView()]

        if mining_model_class is None:
            mining_model_class = HeuristicMining

        super().__init__(views, mining_model_class, dataframe_transformations)

    def get_page_title(self) -> str:
        """Returns the page title.

        Returns
        -------
        str
            The page title.
        """
        return "Heuristic Mining"

    def process_algorithm_parameters(self):
        """Processes the algorithm parameters from the session state. The parameters are set to the instance variables.
        If the parameters are not set in the session state, the default values are used.
        """
        # set session state from instance variables if not set
        if "threshold" not in st.session_state:
            st.session_state.threshold = self.mining_model.get_threshold()

        if "frequency" not in st.session_state:
            st.session_state.frequency = self.mining_model.get_min_frequency()

        if "spm_threshold" not in st.session_state:
            st.session_state.spm_threshold = self.mining_model.get_spm_threshold()

        # set instance variables from session state
        self.threshold = st.session_state.threshold
        self.frequency = st.session_state.frequency
        self.spm_threshold = st.session_state.spm_threshold

    def perform_mining(self) -> None:
        """Performs the mining of the Heuristic Miner algorithm."""
        self.mining_model.create_dependency_graph_with_graphviz(
            self.threshold, self.frequency, self.spm_threshold
        )

    def have_parameters_changed(self) -> bool:
        """Checks if the algorithm parameters have changed.

        Returns
        -------
        bool
            True if the algorithm parameters have changed, False otherwise.
        """
        return (
            self.mining_model.get_threshold() != self.threshold
            or self.mining_model.get_min_frequency() != self.frequency
            or self.mining_model.get_spm_threshold() != self.spm_threshold
        )

    def get_sidebar_values(self) -> dict[str, tuple[int | float, int | float]]:
        """Returns the sidebar values for the Heuristic Miner algorithm.

        Returns
        -------
        dict[str, tuple[int | float, int | float]]
            A dictionary containing the minimum and maximum values for the sidebar sliders.
            The keys of the dictionary are equal to the keys of the sliders.
        """
        sidebar_values = {
            "frequency": (1, self.mining_model.get_max_frequency()),
            "threshold": (0.0, 1.0),
            "spm_threshold": (0.0, 1.0),
        }

        return sidebar_values
