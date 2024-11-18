from ui.base_algorithm_ui.base_algorithm_controller import BaseAlgorithmController
from ui.alpha_miner_ui.alpha_miner_view import AlphaMinerView
import streamlit as st
from api.pickle_save import pickle_load
from mining_algorithms.alpha_mining import AlphaMining


class AlphaGraphController(BaseAlgorithmController):
    """Controller for the Alpha Miner algorithm."""

    def __init__(
        self, working_directory, views=None, mining_model_class=None, dataframe_transformations=None
    ):
        """Initializes the controller for the Alpha Miner algorithm.

        Parameters
        ----------
        working_directory : str
            The directory where files (e.g., graph renderings) will be stored.
        views : List[BaseAlgorithmView] | BaseAlgorithmView, optional
            The views for the Alpha Miner algorithm. If None is passed, the default view is used, by default None
        mining_model_class : MiningInterface Class, optional
            The mining model class for the Alpha Miner algorithm. If None is passed, the default model class is used, by default None
        dataframe_transformations : DataframeTransformations, optional
            The class for the dataframe transformations. If None is passed, a new instance is created, by default None
        """
        self.model = None
        self.working_directory = working_directory

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
        return "Alpha Miner"

    def start_mining(self, cases):
        """Starts the mining process for the Alpha Miner algorithm.

        Parameters
        ----------
        cases : iterable
            A collection of cases to be processed.
        """
        self.model = self.mining_model_class(set(cases))
        self.draw_graph()

    def load_model(self, file_path):
        """Loads a saved model from the specified file path.

        Parameters
        ----------
        file_path : str
            The path to the saved model file.

        Returns
        -------
        str
            The path to the loaded model file.
        """
        self.model = pickle_load(file_path)
        self.draw_graph()
        return file_path

    def draw_graph(self):
        """Draws the dependency graph for the Alpha Miner algorithm."""
        if not self.model:
            st.error("No model available to draw the graph.")
            return None

        graph = self.model.draw_graph()
        graph.render(self.working_directory, format="dot")
        st.success("Graph rendered successfully.")
        return graph

    def get_model(self):
        """Returns the current model instance.

        Returns
        -------
        AlphaMining
            The current Alpha Mining model instance.
        """
        return self.model
