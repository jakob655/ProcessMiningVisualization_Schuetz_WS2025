import streamlit as st

from mining_algorithms.genetic_mining import GeneticMining
from ui.base_algorithm_ui.base_algorithm_controller import BaseAlgorithmController
from ui.genetic_miner_ui.genetic_miner_view import GeneticMinerView


class GeneticMinerController(BaseAlgorithmController):
    """Controller for the Genetic Miner algorithm."""

    def __init__(self, views=None, mining_model_class=None, dataframe_transformations=None):
        """Initializes the Genetic Miner controller.

        Parameters
        ----------
        views : List[BaseAlgorithmView] | BaseAlgorithmView, optional
            The views for the Genetic Miner algorithm. If None is passed, the default view is used, by default None
        mining_model_class : MiningInterface Class, optional
            The mining model class for the Genetic Miner algorithm. If None is passed, the default model class is used, by default None
        dataframe_transformations : DataframeTransformations, optional
            The class for the dataframe transformations. If None is passed, a new instance is created, by default None
        """
        self.population_size = None
        self.max_generations = None
        self.crossover_rate = None
        self.mutation_rate = None
        self.elitism_rate = None
        self.tournament_size = None
        self.power_value = None

        if views is None:
            views = [GeneticMinerView()]

        if mining_model_class is None:
            mining_model_class = GeneticMining

        super().__init__(views, mining_model_class, dataframe_transformations)

    def get_page_title(self) -> str:
        """Returns the page title.

        Returns
        -------
        str
            The page title.
        """
        return "Genetic Mining"

    def process_algorithm_parameters(self):
        """Processes the algorithm parameters from the session state.
        Calls the base implementation for shared filters and initializes additional Genetic Miner-specific parameters.
        """
        super().process_algorithm_parameters()

        # set session state from instance variables if not set
        if "population_size" not in st.session_state:
            st.session_state.population_size = self.mining_model.get_population_size()

        if "max_generations" not in st.session_state:
            st.session_state.max_generations = self.mining_model.get_max_generations()

        if "crossover_rate" not in st.session_state:
            st.session_state.crossover_rate = self.mining_model.get_crossover_rate()

        if "mutation_rate" not in st.session_state:
            st.session_state.mutation_rate = self.mining_model.get_mutation_rate()

        if "elitism_rate" not in st.session_state:
            st.session_state.elitism_rate = self.mining_model.get_elitism_rate()

        if "tournament_size" not in st.session_state:
            st.session_state.tournament_size = self.mining_model.get_tournament_size()

        if "power_value" not in st.session_state:
            st.session_state.power_value = self.mining_model.get_power_value()

        # set instance variables from session state
        self.population_size = st.session_state.population_size
        self.max_generations = st.session_state.max_generations
        self.crossover_rate = st.session_state.crossover_rate
        self.mutation_rate = st.session_state.mutation_rate
        self.elitism_rate = st.session_state.elitism_rate
        self.tournament_size = st.session_state.tournament_size
        self.power_value = st.session_state.power_value

    def have_parameters_changed(self) -> bool:
        """Checks if the algorithm parameters have changed.

        Returns
        -------
        bool
            True if any of the algorithm parameters have changed, False otherwise.
        """

        return (
                st.session_state.get("rerun_genetic_miner", False)
                or st.session_state.get("cancel_run", False)
                or super().have_parameters_changed()
        )

    def get_sidebar_values(self) -> dict[str, tuple[int | float, int | float]]:
        """Returns the sidebar values for the Genetic Miner algorithm.

        Returns
        -------
        dict[str, tuple[int | float, int | float]]
            A dictionary containing the minimum and maximum values for the sidebar sliders.
            The keys of the dictionary are equal to the keys of the sliders.
        """
        sidebar_values = super().get_sidebar_values()
        sidebar_values.update({
            "population_size": (10, 500),
            "max_generations": (10, 1000),
            "crossover_rate": (0.0, 1.0),
            "mutation_rate": (0.0, 0.99),
            "elitism_rate": (0.0, 0.99),
            "tournament_size": (2, min(20, self.population_size)),
            "power_value": (1, 9),
        })
        return sidebar_values

    def perform_mining(self) -> None:
        """Performs the mining of the Genetic Miner algorithm using the current filter parameters."""
        super().perform_mining(population_size=self.population_size, max_generations=self.max_generations,
                               crossover_rate=self.crossover_rate, mutation_rate=self.mutation_rate,
                               elitism_rate=self.elitism_rate, tournament_size=self.tournament_size,
                               power_value=self.power_value)

        # Reset the rerun flag after run
        st.session_state.rerun_genetic_miner = False
        st.session_state.cancel_run = False
