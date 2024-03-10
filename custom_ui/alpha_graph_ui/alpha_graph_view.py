from PyQt5.QtWidgets import QWidget

from custom_ui.algorithm_view_interface import AlgorithmViewInterface

class AlphaGraphView(QWidget, AlgorithmViewInterface):
    def __init__(self, parent, saveFolder = "saves/", workingDirectory = 'temp/graph_viz'):
        super().__init__()
        self.saveFolder = saveFolder
        self.workingDirectory = workingDirectory