from PyQt5.QtWidgets import QWidget, QHBoxLayout

from custom_ui.algorithm_view_interface import AlgorithmViewInterface
from custom_ui.alpha_graph_ui.alpha_graph_controller import AlphaGraphController
from custom_ui.d3_html_widget import HTMLWidget


class AlphaGraphView(QWidget, AlgorithmViewInterface):


    def __init__(self, parent, saveFolder="saves/", workingDirectory='temp/graph_viz'):
        super().__init__()
        self.parent = parent
        self.saveFolder = saveFolder
        self.workingDirectory = workingDirectory

        self.AlphaGraphController = AlphaGraphController(self)

        self.graph_widget = HTMLWidget(parent)

        # Create the main layout
        main_layout = QHBoxLayout(self)
        main_layout.addWidget(self.graph_widget, stretch=3)

    def startMining(self, filename, cases):
        pass

    def loadModel(self):
        pass

    def generate_png(self):
        pass

    def generate_svg(self):
        pass

    def generate_dot(self):
        pass

    def clear(self):
        pass
