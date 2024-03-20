from PyQt5.QtWidgets import QWidget, QHBoxLayout, QFileDialog

from api.custom_error import FileNotFoundException
from custom_ui.algorithm_view_interface import AlgorithmViewInterface
from custom_ui.alpha_graph_ui.alpha_graph_controller import AlphaGraphController
from custom_ui.custom_widgets import SaveProjectButton
from custom_ui.d3_html_widget import HTMLWidget


class AlphaGraphView(QWidget, AlgorithmViewInterface):

    def __init__(self, parent, saveFolder="saves/", workingDirectory='temp/graph_viz'):
        super().__init__()
        self.zoom_factor = None
        self.parent = parent
        self.initialized = False
        self.saveFolder = saveFolder
        self.workingDirectory = workingDirectory

        self.AlphaGraphController = AlphaGraphController(self)
        self.graph_widget = HTMLWidget(parent)
        self.graphviz_graph = None
        self.saveProject_button = SaveProjectButton(self.parent, self.saveFolder, self.get_model)

        # Create the main layout
        main_layout = QHBoxLayout(self)
        main_layout.addWidget(self.graph_widget, stretch=3)

    def startMining(self, filename, cases):
        self.saveProject_button.load_filename(filename)
        self.AlphaGraphController.start_mining(cases)

        self.graph_widget.start_server()
        self.initialized = True
        self.__mine_and_draw()

    def loadModel(self):
        try:
            file_path, _ = QFileDialog.getOpenFileName(None, "Select file", self.saveFolder, "Pickle files (*.pickle)")
            # If the user cancels the file dialog, return
            if not file_path:
                return -1
            filename = self.AlphaGraphController.load_model(file_path)
            if filename == -1:
                return -1
        except TypeError as e:
            message = "AlphaGraphView load_model(): Error: Something went wrong while loading an existing model."
            print(str(e))
            self.parent.show_pop_up_message(message, 6000)
            return -1

        self.saveProject_button.load_filename(filename)
        self.graph_widget.start_server()
        self.initialized = True
        self.__mine_and_draw()

    def __mine_and_draw(self):
        self.graphviz_graph = self.AlphaGraphController.mine_and_draw()

        # Load the image
        filename = self.workingDirectory + '.dot'
        self.graph_widget.set_source(filename)
        try:
            self.graph_widget.reload()
        except FileNotFoundException as e:
            print(e.message)

    def get_model(self):
        return self.AlphaGraphController.get_model()

    def __graph_exists(self):
        if not self.graphviz_graph:
            return False
        return True

    def generate_dot(self):
        if not self.__graph_exists():
            return
        self.graphviz_graph.render(self.workingDirectory, format='dot')
        print("alpha_graph_view: DOT generated")
        return

    def generate_svg(self):
        if not self.__graph_exists():
            return
        self.graphviz_graph.render(self.workingDirectory, format='svg')
        print("alpha_graph_view: SVG generated")
        return

    def generate_png(self):
        if not self.__graph_exists():
            return
        self.graphviz_graph.render(self.workingDirectory, format='png')
        print("alpha_graph_view: DOT generated")
        return

    def clear(self):
        self.graph_widget.clear()
        self.zoom_factor = 1
