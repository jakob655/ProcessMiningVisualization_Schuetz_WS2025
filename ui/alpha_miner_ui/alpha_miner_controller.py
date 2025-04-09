from ui.base_algorithm_ui.base_algorithm_controller import BaseAlgorithmController
from ui.alpha_miner_ui.alpha_miner_view import AlphaMinerView
import streamlit as st
from mining_algorithms.alpha_mining import AlphaMining


class AlphaMinerController(BaseAlgorithmController):
    """Controller for the Alpha Miner algorithm."""

    def __init__(self, views=None, mining_model_class=None, dataframe_transformations=None):
        """Initializes the Alpha Miner controller.

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
            views = [AlphaMinerView()]

        if mining_model_class is None:
            mining_model_class = AlphaMining

        super().__init__(views, mining_model_class, dataframe_transformations)

    def get_page_title(self) -> str:
        """Returns the page title.

        Returns
        -------
        str
            The page title.
        """
        return "Alpha Mining"

    def process_algorithm_parameters(self):
        """Processes the algorithm parameters from the session state. The parameters are set to the instance variables.
        If the parameters are not set in the session state, the default values are used.
        """
        if "spm_threshold" not in st.session_state:
            st.session_state.spm_threshold = self.mining_model.get_spm_threshold()

        # set instance variable from session state
        self.spm_threshold = st.session_state.spm_threshold

    def perform_mining(self) -> None:
        """Performs the mining of the Alpha Miner algorithm."""
        self.mining_model.draw_graph(self.spm_threshold)

    def have_parameters_changed(self) -> bool:
        """Checks if the algorithm parameters have changed.

        Returns
        -------
        bool
            True if the algorithm parameters have changed, False otherwise.
        """
        return self.mining_model.get_spm_threshold() != self.spm_threshold

    def get_sidebar_values(self) -> dict[str, tuple[int | float, int | float]]:
        """Returns the sidebar values for the Alpha Miner algorithm.

        Returns
        -------
        dict[str, tuple[int | float, int | float]]
            A dictionary containing the minimum and maximum values for the sidebar sliders.
            The keys of the dictionary are equal to the keys of the sliders.
        """
        sidebar_values = {
           "spm_threshold": (0.0, 1.0),
        }

        return sidebar_values
