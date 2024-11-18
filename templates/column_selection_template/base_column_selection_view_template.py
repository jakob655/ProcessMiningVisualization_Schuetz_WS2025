import streamlit as st
from ui.column_selection_ui.base_column_selection_view import BaseColumnSelectionView


class BaseColumnSelectionViewTemplate(BaseColumnSelectionView):
    """Standard view for the column selection."""

    def __init__(self):
        """Initializes the standard column selection view. It sets the needed columns and column styles."""
        super().__init__()
        self.needed_columns.extend([])
        # e.g.["time_column", "case_column", "activity_column"]
        self.column_styles.update(dict())
        # e.g.{
        #    "time_column": "background-color: #FF705B",
        #    "case_column": "background-color: #629AFF",
        #    "activity_column": "background-color: #57B868",
        # }

        raise NotImplementedError(
            "Method __init__ must be implemented in the child class"
        )

    def render_column_selections(self, columns: list[str]):
        """Renders the column selection options.

        Parameters
        ----------
        columns : list[str]
            The names of the needed columns
        """
        raise NotImplementedError(
            "Method render_column_selections must be implemented in the child class"
        )
