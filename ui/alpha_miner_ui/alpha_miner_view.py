from ui.base_algorithm_ui.base_algorithm_view import BaseAlgorithmView
import streamlit as st


class AlphaMinerView(BaseAlgorithmView):
    """View for the Alpha Miner algorithm."""

    def render_sidebar(self, sidebar_values: dict[str, tuple[int | float, int | float]]) -> None:
        """Renders the sidebar for the Alpha Miner algorithm."""
        # No sidebar components are needed for Alpha Miner in this implementation.
        pass
