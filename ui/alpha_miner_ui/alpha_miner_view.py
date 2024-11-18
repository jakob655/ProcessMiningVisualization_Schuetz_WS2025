from ui.base_algorithm_ui.base_algorithm_view import BaseAlgorithmView
import streamlit as st
from components.number_input_slider import number_input_slider

class AlphaMinerView(BaseAlgorithmView):
    """View for the Alpha Miner algorithm."""

    def __init__(self, working_directory="temp/graph_viz"):
        """
        Initializes the AlphaGraphView.

        Parameters
        ----------
        working_directory : str, optional
            Directory where the rendered graph files will be stored, by default "temp/graph_viz".
        """
        self.working_directory = working_directory
        self.alpha_controller = None

    def set_controller(self, controller):
        """Sets the controller for the Alpha Miner algorithm.

        Parameters
        ----------
        controller : AlphaGraphController
            The controller for the Alpha Miner.
        """
        self.alpha_controller = controller

    def render_sidebar(self, sidebar_values: dict[str, tuple[int | float, int | float]]) -> None:
        """Renders the sidebar for the Alpha Miner algorithm.

        Parameters
        ----------
        sidebar_values : dict[str, tuple[int | float, int | float]]
            A dictionary containing the minimum and maximum values for the sidebar sliders.
        """
        number_input_slider(
            label="Minimum Frequency",
            min_value=sidebar_values["frequency"][0],
            max_value=sidebar_values["frequency"][1],
            key="frequency",
            help="Minimum frequency for displaying edges and nodes.",
        )

        st.text_input(
            label="Save Directory",
            key="save_folder",
            value="saves/",
            help="Directory where the saved models will be stored."
        )

    def render_main_content(self):
        """Renders the main content area of the Alpha Miner algorithm."""
        st.title("Alpha Miner")

        # Graph visualization
        graph_area = st.empty()

        # Buttons
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("Start Mining"):
                filename = st.text_input("Filename", value="example")
                cases = st.text_area("Cases", value="case1,case2,...").split(",")
                self.start_mining(filename, cases)

        with col2:
            if st.button("Load Model"):
                self.load_model()

        with col3:
            st.button("Clear", on_click=self.clear_graph)

        # Generate file buttons
        if self.alpha_controller:
            st.download_button("Generate DOT", on_click=self.generate_dot)
            st.download_button("Generate SVG", on_click=self.generate_svg)
            st.download_button("Generate PNG", on_click=self.generate_png)

    def start_mining(self, filename, cases):
        """Starts the mining process with the given filename and cases.

        Parameters
        ----------
        filename : str
            The name of the file to save the model.
        cases : list
            A list of cases to process.
        """
        unique_cases = set(map(tuple, cases))
        self.alpha_controller.start_mining(unique_cases)
        st.success("Mining completed and graph generated.")

    def load_model(self):
        """Loads an existing model."""
        file_path = st.text_input("Enter the path to the saved model", value="saves/example_model.pickle")
        if not file_path:
            st.error("File path cannot be empty.")
            return
        try:
            self.alpha_controller.load_model(file_path)
            st.success("Model loaded successfully.")
        except Exception as e:
            st.error(f"Failed to load model: {str(e)}")

    def generate_dot(self):
        """Generates a DOT file for the graph."""
        self.alpha_controller.draw_graph()
        st.success("DOT file generated.")

    def generate_svg(self):
        """Generates an SVG file for the graph."""
        self.alpha_controller.draw_graph()
        st.success("SVG file generated.")

    def generate_png(self):
        """Generates a PNG file for the graph."""
        self.alpha_controller.draw_graph()
        st.success("PNG file generated.")

    def clear_graph(self):
        """Clears the current graph visualization."""
        self.alpha_controller.clear()
        st.info("Graph cleared.")
