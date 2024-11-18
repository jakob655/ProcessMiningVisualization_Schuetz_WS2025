import streamlit as st
from ui.home_ui.home_controller import HomeController
from ui.column_selection_ui.column_selection_controller import ColumnSelectionController
from ui.export_ui.export_controller import ExportController
from ui.algorithm_explanation_ui.algorithm_explanation_controller import (
    AlgorithmExplanationController,
)

from config import algorithm_routes


st.set_page_config(
    page_title="Process Mining Tool",
    page_icon=":bar_chart:",
    layout="wide",
)

if "page" not in st.session_state:
    st.session_state.page = "Home"

if st.session_state.page == "Home":
    HomeController().start()
elif st.session_state.page == "Algorithm":
    algorithm_routes[st.session_state.algorithm]().start()
elif st.session_state.page == "ColumnSelection":
    ColumnSelectionController().start()
elif st.session_state.page == "Export":
    ExportController().start()
elif st.session_state.page == "Documentation":
    AlgorithmExplanationController().start()
