from ui.base_ui.base_controller import BaseController
from ui.export_ui.export_view import ExportView
import streamlit as st
from components.buttons import to_home, navigate_to
from io_operations.export_operations import ExportOperations
from io_operations.import_operations import ImportOperations
from analysis.detection_model import DetectionModel
from exceptions.io_exceptions import (
    UnsupportedFileTypeException,
    NotImplementedFileTypeException,
)
from logger import get_logger


class ExportController(BaseController):
    """Controller for the Export page."""

    def __init__(
        self,
        views=None,
        export_model=None,
        import_model=None,
        detection_model=None,
        export_formats=None,
    ):
        """Initializes the controller for the Export page.

        Parameters
        ----------
        views : List[BaseView] | BaseView, optional
            The views for the Export page. If None is passed, the default view is used, by default None
        export_model : ExportOperations, optional
            The export operations model for exporting graphs and models. If None is passed, a new instance is created, by default None
        import_model : ImportOperations, optional
            The import operations model for reading files. If None is passed, a new instance is created, by default None
        detection_model : DetectionModel, optional
            The detection model for detecting mime types. If None is passed, a new instance is created, by default None
        export_formats : List[str], optional
            The export formats to display. If None is passed, the default formats from the config are used, by default None
        """
        if views is None:
            from ui.export_ui.export_view import ExportView

            views = [ExportView()]

        if export_formats is None:
            from config import graph_export_formats

            export_formats = graph_export_formats

        if export_model is None:
            export_model = ExportOperations()

        if import_model is None:
            import_model = ImportOperations()

        if detection_model is None:
            detection_model = DetectionModel()

        self.formats = export_formats

        self.export_model = export_model
        self.import_model = import_model
        self.detection_model = detection_model
        super().__init__(views)

        self.logger = get_logger("ExportController")

    def get_page_title(self) -> str:
        """Returns the page title.

        Returns
        -------
        str
            The page title.
        """
        return "Export"

    def process_session_state(self):
        """Processes the session state. Checks if a model has been selected and if a graph has been generated.
        If not, an error message is displayed and the user is navigated back to the home page or alforithm page.
        User input for the DPI and export format is processed.
        """
        super().process_session_state()
        if "model" not in st.session_state:
            self.logger.error("Model was not selected")
            self.logger.info("redirect to home page")
            st.session_state.error = "Model not selected"
            to_home("Home")
            st.rerun()

        self.mining_model = st.session_state.model

        if self.mining_model.graph is None:
            self.logger.error("Graph not generated")
            self.logger.info("redirect to algorithm page")
            st.session_state.error = "Graph not generated"
            navigate_to("Algorithm")

        if "dpi" not in st.session_state:
            st.session_state.dpi = 96

        if "export_format" not in st.session_state:
            st.session_state.export_format = "SVG"

        self.graph = self.mining_model.get_graph()
        self.dpi = st.session_state.dpi
        self.export_format = st.session_state.export_format

    def export_graph(self, format: str) -> str:
        """Exports the graph in the specified format to a temporary file on the disk and returns the file path.

        Parameters
        ----------
        format : str
            The format to export the graph to.

        Returns
        -------
        str
            The file path of the exported graph.

        Raises
        ------
        UnsupportedFileTypeException
            If the export format is not supported.
        """
        self.export_model.export_graph(self.graph, "temp/graph", format, dpi=self.dpi)
        return "temp/graph" + "." + format.lower()

    def read_png(self, file_path: str) -> str:
        """Reads the PNG file from the disk and returns the image.

        Parameters
        ----------
        file_path : str
            The file path of the PNG file.

        Returns
        -------
        str
            The image of the PNG file encoded as base64 string.

        Raises
        ------
        FileNotFoundError
            If the file is not found.
        """
        image = self.import_model.read_img(file_path)
        return image

    def pickle_model(self) -> bytes:
        """Pickle the model and return it as bytes.

        Returns
        -------
        bytes
            The model pickled as bytes.
        """
        return self.export_model.export_model_to_bytes(self.mining_model)

    def read_file(self, file_path: str) -> tuple[bytes, str]:
        """Reads the file from the disk and returns the content and MIME type.

        Parameters
        ----------
        file_path : str
            The file path of the file to read.

        Returns
        -------
        tuple[bytes, str]
            The content of the file as bytes and the MIME type of the file.

        Raises
        ------
        FileNotFoundError
            If the file is not found.
        UnsupportedFileTypeException
            If the export format is not supported

        """
        mime = self.detection_model.detect_mime_type(self.export_format)
        file_content = self.import_model.read_file_binary(file_path)

        return file_content, mime

    def run(self, selected_view, index):
        """Runs the controller for the Export page. This method is called to display the Export page and to react to user input.

        Parameters
        ----------
        selected_view : BaseView
            The view to display the import view.
        index : int
            The index of the selected view.
        """
        selected_view.display_back_button()
        selected_view.display_export_format(self.formats)
        if self.export_format == "PNG":
            selected_view.display_dpi_input(50, self.dpi, 1)

        selected_view.display_model_export_button("model.pickle", self.pickle_model())
        selected_view.display_disabled_export_button()

        try:
            selected_view.display_loading_text(
                f"Exporting graph to {self.export_format} ..."
            )
            file_path = self.export_graph(format=self.export_format)
            file, mime = self.read_file(file_path)
            selected_view.display_export_button(
                "graph." + self.export_format.lower(), file, mime
            )
            if self.export_format != "PNG":
                selected_view.display_loading_text(f"Exporting graph to PNG ...")
                png_file_path = self.export_graph(format="PNG")
            else:
                png_file_path = file_path
            png_file = self.read_png(png_file_path)

            selected_view.display_png(png_file)
        except UnsupportedFileTypeException as ex:
            self.logger.exception(ex)
            st.session_state.error = ex.message
            del st.session_state.export_format
            st.rerun()
        except FileNotFoundError as e:
            self.logger.exception(e)
            st.session_state.warning = "Error exporting graph. Please wait until the graph is generated, before changing formats or dpi."
            st.rerun()
        except NotImplementedFileTypeException as ex:
            self.logger.exception(ex)
            self.logger.info("Switching to SVG format")
            st.session_state.error = ex.message
            del st.session_state.export_format
            st.rerun()
