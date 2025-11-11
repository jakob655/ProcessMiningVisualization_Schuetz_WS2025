import streamlit as st
from mining_algorithms.alpha_mining import AlphaMining
from ui.alpha_miner_ui.alpha_miner_view import AlphaMinerView
from ui.base_algorithm_ui.base_algorithm_controller import BaseAlgorithmController


class AlphaMinerController(BaseAlgorithmController):
    """Controller for the Alpha Miner algorithm."""

    def __init__(self, views=None, mining_model_class=None, dataframe_transformations=None):
        """Initializes the Alpha Miner controller.

        Parameters
        ----------
        views : List[BaseAlgorithmView] | BaseAlgorithmView, optional
            The views for the Alpha Miner algorithm. If None is passed, the default view is used, by default None
        mining_model_class : MiningInterface Class, optional
            The mining model class for the Alpha Miner algorithm. If None is passed, the default model class is used, by default None
        dataframe_transformations : DataframeTransformations, optional
            The class for the dataframe transformations. If None is passed, a new instance is created, by default None
        """
        if views is None:
            views = [AlphaMinerView()]

        if mining_model_class is None:
            mining_model_class = AlphaMining

        self.use_petri_net = False
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
        """Processes the algorithm parameters from the session state.
        Calls the base implementation for shared filters and initializes additional Alpha Miner-specific parameters.
        """
        super().process_algorithm_parameters()
        if "alpha_use_petri_net" not in st.session_state:
            st.session_state.alpha_use_petri_net = self.use_petri_net
        self.use_petri_net = st.session_state.alpha_use_petri_net

    def have_parameters_changed(self) -> bool:
        """Checks if the algorithm parameters have changed.

        Returns
        -------
        bool
            True if any of the algorithm parameters have changed, False otherwise.
        """
        return (
                super().have_parameters_changed()
                or getattr(self.mining_model, 'use_petri_net', False) != self.use_petri_net
        )
    

    def get_sidebar_values(self) -> dict[str, tuple[float, float]]:
        """Returns the sidebar values for the Alpha Miner algorithm.

        Returns
        -------
        dict[str, tuple[float, float]]
            A dictionary containing the minimum and maximum values for the sidebar sliders.
            The keys of the dictionary are equal to the keys of the sliders.
        """
        return super().get_sidebar_values()

    def perform_mining(self) -> None:
        """Performs the mining of the Alpha Miner algorithm using the current filter parameters."""
        super().perform_mining(use_petri_net=self.use_petri_net)