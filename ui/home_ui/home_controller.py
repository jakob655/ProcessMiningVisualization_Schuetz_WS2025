import streamlit as st
from ui.base_ui.base_controller import BaseController
from analysis.detection_model import DetectionModel
from io_operations.import_operations import ImportOperations
from exceptions.io_exceptions import (
    UnsupportedFileTypeException,
    NotImplementedFileTypeException,
)
from logger import get_logger


class HomeController(BaseController):
    """Controller for the Home page."""

    def __init__(
        self,
        views=None,
        detection_model: DetectionModel = None,
        import_model: ImportOperations = None,
        supported_file_types: list[str] = None,
    ):
        """Initializes the controller for the Home page.

        Parameters
        ----------
        views :  List[BaseView] | BaseView, optional
            The views for the Home page. If None is passed, the HomeView is used, by default None
        detection_model : DetectionModel, optional
            The detection model for detecting file types and delimiters. If None is passed, a new instance is created, by default None
        import_model : ImportOperations, optional
            The import operations model for reading files. If None is passed, a new instance is created, by default None
        supported_file_types : list[str], optional
            The supported file types. If None is passed, the file suffixes from the config file are used, by default None
        """
        self.detection_model = DetectionModel()
        self.import_model = ImportOperations()
        if views is None:
            from ui.home_ui.home_view import HomeView

            views = [HomeView()]
        super().__init__(views)
        self.logger = get_logger("HomeController")

        if supported_file_types is None:
            from config import import_file_suffixes

            supported_file_types = import_file_suffixes

        self.supported_file_types = supported_file_types

    def get_page_title(self) -> str:
        """Returns the page title."""
        return ""

    def process_session_state(self):
        """Processes the session state. Checks if a file has been uploaded and stores it in a instance variable."""
        super().process_session_state()
        self.uploaded_file = st.session_state.get("uploaded_file", None)

    def process_file(self, selected_view):
        """Processes the uploaded file and displays the import view.
        The file type is detected and the view is displayed accordingly.

        Parameters
        ----------
        selected_view : BaseView
            The view to display the import view.
        """
        if self.uploaded_file is None:
            return

        try:
            file_type = self.detection_model.detect_file_type(self.uploaded_file)

            if file_type == "csv":
                line = self.import_model.read_line(self.uploaded_file)
                detected_delimiter = self.detection_model.detect_delimiter(line)
                self.uploaded_file.seek(0)
                selected_view.display_df_import(detected_delimiter)
            elif file_type == "pickle":
                model = self.import_model.read_model(self.uploaded_file)
                selected_view.display_model_import(model)
            else:
                raise NotImplementedFileTypeException(file_type)
        except UnsupportedFileTypeException as e:
            self.logger.exception(e)
            st.session_state.error = e.message
            st.rerun()
        except NotImplementedFileTypeException as e:
            self.logger.exception(e)
            st.session_state.error = e.message
            st.rerun()

    def set_model_and_algorithm(self, model, algorithm: str):
        """Sets the model and algorithm in the session state before navigating to the Algorithm page.

        Parameters
        ----------
        model : MiningInterface
            The mining model to be imported.
        algorithm : str
            The chosen algorithm.
        """
        st.session_state.model = model
        st.session_state.algorithm = algorithm

    def set_df(self, delimiter: str):
        """creates a dataframe from the uploaded file with the given delimiter.
        Stores the dataframe in the session state and changes the routing to the ColumnSelection page.

        Parameters
        ----------
        delimiter : str
            The delimiter to be used for the CSV file.
        """
        if delimiter == "":
            st.session_state.error = "Please enter a delimiter"
            # change routing to home
            st.session_state.page = "Home"
            return
        st.session_state.df = self.import_model.read_csv(self.uploaded_file, delimiter)

    def run(self, selected_view, index):
        """Runs the controller for the Home page. This method is called to display the Home page and to react to user input.

        Parameters
        ----------
        selected_view : BaseView
            The view to display the import view.
        index : int
            The index of the selected view.
        """
        self.selected_view = selected_view
        selected_view.display_intro()
        selected_view.display_file_upload(self.supported_file_types)
        if self.uploaded_file is not None:
            self.process_file(selected_view)
