import streamlit as st
from ui.base_ui.base_controller import BaseController
from components.buttons import navigate_to, to_home
from io_operations.import_operations import ImportOperations
from config import docs_path_mappings
from logger import get_logger


class AlgorithmExplanationController(BaseController):
    """Controller for the algorithm explanation page."""

    def __init__(self, views=None, import_model=None):
        """Initializes the controller for the algorithm explanation page.

        Parameters
        ----------
        views : List[BaseView] | BaseView, optional
            The views for the algorithm explanation page. If None is passed, the default view is used, by default None
        import_model : ImportOperations, optional
            The import operations model for reading files. If None is passed, a new instance is created, by default None
        """

        if views is None:
            from ui.algorithm_explanation_ui.algorithm_explanation_view import (
                AlgorithmExplanationView,
            )

            views = [AlgorithmExplanationView()]

        if import_model is None:
            import_model = ImportOperations()

        self.import_model = import_model
        super().__init__(views)

        self.logger = get_logger("AlgorithmExplanationController")

    def get_page_title(self) -> str:
        """Returns the page title. Here a title is not needed as the title is written in the markdown file.

        Returns
        -------
        str
            The page title. In this case, an empty string is returned.
        """
        return ""

    def process_session_state(self):
        """Processes the session state. Checks if an algorithm has been selected and if the algorithm has documentation.
        If not, an error message is displayed and the user is navigated back to the home or algorithm page.
        """
        if "algorithm" not in st.session_state:
            self.logger.error("Algorithm was not selected")
            self.logger.info("redirect to home page")
            st.session_state.error = "Algorithm not selected"
            to_home()

        if st.session_state.algorithm not in docs_path_mappings:
            self.logger.error("Algorithm does not have documentation")
            self.logger.info("redirect to algorithm page")
            st.session_state.error = "Algorithm does not have documentation"
            navigate_to("Algorithm")

        self.file_path = docs_path_mappings[st.session_state.algorithm]

    def read_algorithm_file(self) -> str:
        """Reads the content of the algorithm markdown file.

        Returns
        -------
        str
            The content of the algorithm markdown file.
        """
        file_content = self.import_model.read_file(self.file_path)
        return file_content

    def run(self, selected_view, index):
        """Runs the controller. It reads the algorithm file and displays it in the view.

        Parameters
        ----------
        selected_view : BaseView
            The selected view.
        index : int
            The index of the selected view in the views list.
        """
        self.selected_view = selected_view
        selected_view.display_back_button()
        try:
            file_content = self.read_algorithm_file()
            selected_view.display_algorithm_file(file_content)
        except FileNotFoundError as e:
            self.logger.exception(e)
            self.logger.error("Algorithm does not have a documentation file")
            self.logger.info("redirect to algorithm page")
            st.session_state.error = "Algorithm does not have a documentation file"
            navigate_to("Algorithm")
