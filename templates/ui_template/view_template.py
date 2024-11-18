import streamlit as st
from ui.base_ui.base_view import BaseView


class ViewTemplate(BaseView):
    """Template for creating views."""

    def create_layout(self):
        """Creates the layout for the views."""
        super().create_layout()
        raise NotImplementedError(
            "Method create_layout must be implemented in the child class or be removed if not needed"
        )

    # Add more methods here as needed for displaying the view.
