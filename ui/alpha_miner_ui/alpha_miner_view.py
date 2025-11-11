import streamlit as st
from ui.base_algorithm_ui.base_algorithm_view import BaseAlgorithmView


class AlphaMinerView(BaseAlgorithmView):
    """View for the Alpha Miner algorithm."""
    def render_log_filter_extensions(self, sidebar_values: dict[str, any]) -> None:
        st.toggle(
            "Petri-Net Visualization",
            key="alpha_use_petri_net",
            value=st.session_state.get("alpha_use_petri_net", False),
            help="Switch between classic and petri-net visualization.",
        )
