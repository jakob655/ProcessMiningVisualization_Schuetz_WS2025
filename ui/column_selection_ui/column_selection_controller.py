import streamlit as st
from ui.base_ui.base_controller import BaseController
from components.buttons import to_home
from config import algorithm_mappings
from analysis.predictions_model import PredictionModel
from transformations.dataframe_styler import DataFrameStyler
import pandas as pd
from logger import get_logger


class ColumnSelectionController(BaseController):
    """Controller for the column selection page. It processes the needed columns, predicts the columns if necessary and styles the dataframe."""

    max_rows_shown = 200

    def __init__(self, views=None, prediction_model=None, dataframe_styler=None):
        """Initializes the controller for the column selection page.

        Parameters
        ----------
        views : List[BaseView] | BaseView, optional
            The views for the column selection page. If None is passed, the default view is used, by default None
        prediction_model : PredictionModel, optional
            The prediction model for predicting columns. If None is passed, a new instance is created, by default None
        dataframe_styler : DataFrameStyler, optional
            The dataframe styler for styling the dataframe. If None is passed, a new instance is created, by default None
        """

        if views is None:
            from ui.column_selection_ui.standard_column_selection_view import (
                StandardColumnSelectionView,
            )

            views = [StandardColumnSelectionView()]

        if prediction_model is None:
            prediction_model = PredictionModel()

        if dataframe_styler is None:
            dataframe_styler = DataFrameStyler(self.max_rows_shown)

        self.predictions_model = prediction_model
        self.dataframe_styler = dataframe_styler
        super().__init__(views)

        self.logger = get_logger("ColumnSelectionController")

    def get_page_title(self) -> str:
        """Returns the page title.

        Returns
        -------
        str
            The page title.
        """
        return "Column Selection"

    def select_view(self):
        """Selects the view to display. The view is selected based on the selected algorithm.

        Returns
        -------
        tuple[BaseView, int]
            The selected view and the index of the view in the views list.
        """
        if self.selected_algorithm == "":
            # choose other view depending on chosen algorithm
            pass
        else:
            return self.views[0], 0

    def process_session_state(self):
        """Processes the session state. Checks if a file has been uploaded and if an algorithm has been selected.
        If not, an error message is displayed and the user is navigated back to the home page.
        The dataframe is set and the selected algorithm is stored in the session state.
        """
        super().process_session_state()
        if "df" not in st.session_state:
            self.logger.error("No file uploaded")
            self.logger.info("redirect to home page")
            self.error_message = "Please upload a file first"
            to_home("Home")
            st.rerun()

        if "algorithm_selection" not in st.session_state:
            st.session_state.algorithm_selection = next(iter(algorithm_mappings.keys()))

        self.selected_algorithm = st.session_state.algorithm_selection

        self.df = st.session_state.df
        self.dataframe_styler.set_dataframe(self.df)

    def predict_columns(
        self, needed_columns: list[str], columns: list[str]
    ) -> list[str | None]:
        """Predicts the columns needed for the algorithm. If the columns are not selected by the user, they are predicted. The prediction is based on the column names.
        If a column cannot be predicted, it will be set to None.

        Parameters
        ----------
        needed_columns : list[str]
            list of needed columns e.g. ["time_column", "case_column", "activity_column"]
        columns : list[str]
            list of columns from the dataframe

        Returns
        -------
        list[str | None]
            list of predicted columns. If one of the needed columns cannot be predicted, it will be None.
        """
        return self.predictions_model.predict_columns(columns, needed_columns)

    def style_df(self, column_styles: dict[str, str]) -> pd.DataFrame:
        """Styles the dataframe based on the column styles.

        Parameters
        ----------
        column_styles : dict[str, str]
            The styles for the columns. The keys are the column names and the values are the styles for the selected columns.

        Returns
        -------
        pd.DataFrame
            The styled dataframe.
        """
        self.dataframe_styler.set_column_styles(column_styles)
        return self.dataframe_styler.stlye_df(self.selected_columns)

    def process_needed_columns(self):
        """Processes the needed columns for the algorithm. If the columns are not selected by the user, they are predicted. The prediction is based on the column names.
        If a column cannot be predicted, it will be set to None. The selected columns are stored in the session state, to be displayed in the view.
        """
        self.selected_columns = dict()

        # if all columns are None, try to predict them
        if not any(
            [st.session_state.get(column, None) for column in self.needed_columns]
        ):
            predicted_columns = self.predict_columns(
                self.needed_columns, self.df.columns
            )
            # set the predicted columns in the session state
            for predicted_column, needed_column in zip(
                predicted_columns, self.needed_columns
            ):
                st.session_state[needed_column] = predicted_column

        for column in self.needed_columns:
            self.selected_columns[column] = st.session_state.get(column, None)

    def get_needed_columns(self, view):
        """Gets the needed columns for the algorithm.

        Parameters
        ----------
        view : BaseView
            The view for the column selection page.
        """
        self.needed_columns = view.get_needed_columns()

    def run(self, selected_view, index):
        """Runs the controller. It displays the back button, the column selection options, the dataframe and the algorithm selection options.

        Parameters
        ----------
        selected_view : BaseColumnSelectionView
            The selected view for the column selection page.
        index : int
            The index of the selected view in the views list.
        """
        selected_view.display_back_button()
        self.get_needed_columns(selected_view)
        self.process_needed_columns()
        selected_view.display_column_selections(self.df.columns)

        styled_df = self.style_df(
            selected_view.get_column_styles(),
        )

        selected_view.display_df(styled_df)
        selected_view.display_algorithm_selection([*algorithm_mappings.keys()])
        selected_view.display_mine_button()

    def on_mine_click(self):
        """Processes the selected columns and the selected algorithm. The selected columns are stored in the session state and the algorithm is set in the session state.
        If the columns are not selected, an error message is displayed and the user is navigated back to the column selection page.
        If a column is selected multiple times, an error message is displayed and the user is navigated back to the column selection page.
        """
        if not all(self.selected_columns.values()):
            self.logger.error("Not all columns are selected")
            st.session_state.error = "Please select a column for all columns"
            st.session_state.page = "ColumnSelection"
            return

        if len(set(self.selected_columns.values())) != len(
            self.selected_columns.values()
        ):
            self.logger.error("Columns are selected multiple times")
            st.session_state.error = "Please select a different column for each column"
            st.session_state.page = "ColumnSelection"
            return

        st.session_state.algorithm = algorithm_mappings[self.selected_algorithm]
        st.session_state.selected_columns = self.selected_columns
