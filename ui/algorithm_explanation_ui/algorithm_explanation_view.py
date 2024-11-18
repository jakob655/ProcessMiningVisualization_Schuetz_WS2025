import streamlit as st
from ui.base_ui.base_view import BaseView
from components.buttons import navigation_button


class AlgorithmExplanationView(BaseView):
    """View for the algorithm explanation page."""

    def create_layout(self):
        """Creates the layout for the algorithm explanation page."""
        super().create_layout()
        _, self.content_column, _ = st.columns([1, 6, 1])

    def display_algorithm_file(self, file_content: str):
        """Displays the content of the algorithm markdown file.

        Parameters
        ----------
        file_content : str
            The content of the algorithm markdown file.
        """
        with self.content_column:
            st.markdown(file_content)

    def display_back_button(self):
        """Displays the back button to navigate back to the algorithm page."""
        with self.content_column:
            navigation_button(
                "Back to Algorithm Page",
                route="Algorithm",
                type="secondary",
                use_container_width=True,
            )
