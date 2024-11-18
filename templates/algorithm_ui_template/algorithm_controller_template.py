from ui.base_algorithm_ui.base_algorithm_controller import BaseAlgorithmController
import streamlit as st


class AlgorithmControllerTemplate(BaseAlgorithmController):
    """Template controller for the algorithm."""

    def __init__(
        self, views=None, mining_model_class=None, dataframe_transformations=None
    ):
        """Initializes the controller.

        Parameters
        ----------
        views : List[BaseAlgorithmView] | BaseAlgorithmView, optional
            The views for the  algorithm. If None is passed, the default view is used, by default None
        mining_model_class : MiningInterface Class, optional
            The mining model class for the algorithm. If None is passed, the default model class is used, by default None
        dataframe_transformations : DataframeTransformations, optional
            The class for the dataframe transformations. If None is passed, a new instance is created, by default None
        """
        if views is None:
            views = []  # Add the views for the algorithm here

        if mining_model_class is None:
            mining_model_class = None  # Add the mining model class for the algorithm here (e.g. HeuristicMining). Only the class name is needed not the instance.
        super().__init__(views, mining_model_class, dataframe_transformations)

    def get_page_title(self) -> str:
        """Returns the page title.

        Returns
        -------
        str
            The page title.
        """
        raise NotImplementedError("The method 'get_page_title' must be implemented.")

    def process_algorithm_parameters(self):
        """Processes the algorithm parameters from the session state. The parameters are set to the instance variables.
        If the parameters are not set in the session state, the default values are used.
        """
        raise NotImplementedError(
            "The method 'process_algorithm_parameters' must be implemented."
        )

    def perform_mining(self) -> None:
        """Performs the mining of the algorithm."""
        raise NotImplementedError("The method 'perform_mining' must be implemented.")

    def have_parameters_changed(self) -> bool:
        """Checks if the algorithm parameters have changed.

        Returns
        -------
        bool
            True if the algorithm parameters have changed, False otherwise.
        """
        raise NotImplementedError(
            "The method 'have_parameters_changed' must be implemented."
        )

    def get_sidebar_values(self) -> dict[str, tuple[int | float, int | float]]:
        """Returns the sidebar values for the mining algorithm.

        Returns
        -------
        dict[str, tuple[int | float, int | float]]
            A dictionary containing the minimum and maximum values for the sidebar sliders.
            The keys of the dictionary are equal to the keys of the sliders.
        """
        sidebar_values = {}
        # e.g. sidebar_values["threshold"] = (0.0, 1.0)
        raise NotImplementedError(
            "The method 'get_sidebar_values' must be implemented."
        )
        return sidebar_values

    # Optional: if another transformation is needed, override this method otherwise it is not needed.
    def transform_df_to_log(self, df, **selected_columns) -> tuple:
        """Transforms the DataFrame to a log. The method uses the dataframe_transformations class to transform the DataFrame to a log.
        The selected columns are passed as keyword arguments to the 'DataframeTransformation' class. The method returns the log data that is used to create the mining model.
        Can be overridden by the subclass if additional transformations are needed.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame to be transformed.

        Returns
        -------
        tuple
            The log data that is used to create the mining model. The order of the log data must be the same as the order of the parameters of the create_mining_instance method of the mining model class.
        """
        self.dataframe_transformations.set_dataframe(df)
        # call a methode from the dataframe_transformations class to transform the DataFrame to a log.
        raise NotImplementedError(
            "The method 'transform_df_to_log' must be implemented."
        )

    # Optional: add logic to switch between views. only needed if multiple views are present
    def select_view(self) -> tuple:
        """Selects the view to be displayed. The first view in the list is selected as the default view.
        The method can be overridden in the child class to implement a different view selection logic, if needed.

        Returns
        -------
        tuple[BaseView, int]
            A tuple containing the selected view and the index of the view in the list.
        """
        raise NotImplementedError(
            "Method select_view must be implemented in the child class or be removed if not needed"
        )
