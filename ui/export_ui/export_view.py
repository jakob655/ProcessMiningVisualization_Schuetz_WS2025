from ui.base_ui.base_view import BaseView
import streamlit as st
from components.buttons import navigation_button
from components.PNGViewer import PNGViewer


class ExportView(BaseView):
    """View for the Export page."""

    png_height = 600

    def create_layout(self):
        """Creates the layout for the Export page.
        The layout consists of a container for the PNG image, a container for the back button,
        a container for the export format, a container for the DPI input, a container for the export button,
        and a container for the model export button.
        """
        super().create_layout()
        png_container_wrapper = st.container(
            border=True, height=self.png_height + 40
        )  # # add 40 to height to account for padding

        with png_container_wrapper:
            self.png_container = st.empty()

        self.back_button_container = st.container()

        with st.sidebar:
            self.export_format_container = st.container()
            self.dpi_container = st.container()
            export_button_col_wrapper, self.model_export_button_col = st.columns([1, 1])
            with export_button_col_wrapper:
                self.export_button_col = st.empty()

    def display_png(self, png):
        """Displays the PNG image."""
        with self.png_container:
            PNGViewer(png, height=self.png_height)

    def display_back_button(self):
        """Displays the back button."""
        with self.back_button_container:
            navigation_button("Back", "Algorithm", type="secondary")

    def display_export_format(self, formats: list[str]):
        """Displays the export format selection dropdown.

        Parameters
        ----------
        formats : List[str]
            The list of export formats.
        """
        with self.export_format_container:
            self.export_format = st.selectbox(
                "Export as:", formats, key="export_format"
            )

    def display_dpi_input(self, min_value: int, value: int, step: int):
        """Displays the DPI input.

        Parameters
        ----------
        min_value : int
            The minimum value for the DPI input.
        value : int
            The value for the DPI input.
        step : int
            The step for the DPI input.
        """
        with self.dpi_container:
            self.dpi = st.number_input(
                "DPI", min_value=min_value, value=value, step=step, key="dpi"
            )

    def display_export_button(self, file_name: str, file: bytes, mime: str):
        """Displays the export button.

        Parameters
        ----------
        file_name : str
            The name of the file to be exported.
        file : bytes
            The file to be exported as bytes.
        mime : str
            The MIME type of the file.
        """
        with self.export_button_col:
            st.download_button(
                label="Export",
                file_name=file_name,
                data=file,
                mime=mime,
                type="primary",
            )

    def display_disabled_export_button(self):
        """Displays a disabled export button."""
        with self.export_button_col:
            st.button("Export", disabled=True)

    def display_model_export_button(self, file_name: str, file: bytes):
        """Displays the model export button.

        Parameters
        ----------
        file_name : str
            The name of the file to be exported.
        file : bytes
            The file/model to be exported as bytes.
        """
        with self.model_export_button_col:
            st.download_button(
                label="Export Model",
                file_name=file_name,
                data=file,
                mime="application/octet-stream",
                type="primary",
            )

    def display_loading_text(self, text):
        """Displays a loading text.

        Parameters
        ----------
        text : str
            The text to display while the callback is running.
        """
        with self.png_container:
            st.write(text)
