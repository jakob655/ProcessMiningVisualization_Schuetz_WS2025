import streamlit as st
from ui.column_selection_ui.standard_column_selection_view import (
    StandardColumnSelectionView,
)


class ExtendedColumnSelectionViewTemplate(StandardColumnSelectionView):
    """Standard view for the column selection."""

    def __init__(self):
        """Initializes the standard column selection view. It sets the needed columns and column styles."""
        super().__init__()

        # Add the needed columns and column styles here use extend or append
        # self.needed_columns.append("additional_column")
        # self.column_styles["additional_column"] = "background-color: #FF705B"
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
        super().render_column_selections(columns)
        # Add the rendering of the additional column here
        # e.g.
        # additional_col = st.columns(1)
        # with additional_col:
        #     st.selectbox("Select the :blue[case column]",
        #        columns,
        #        key=self.needed_columns[3],
        #        index=None
        #     )

        raise NotImplementedError(
            "Method render_column_selections must be implemented in the child class"
        )
