# This project was assisted by ChatGPT

import sys
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QLabel, QStyleFactory, QApplication, QMainWindow, QStackedWidget, QMessageBox, QFileDialog
from api.custom_error import FileNotFoundException, UndefinedErrorException
from custom_ui.alpha_graph_ui.alpha_graph_view import AlphaGraphView
from custom_ui.column_selection_view import ColumnSelectionView
from custom_ui.heuristic_graph_ui.heuristic_graph_view import HeuristicGraphView
from custom_ui.fuzzy_graph_ui.fuzzy_graph_view import FuzzyGraphView
from custom_ui.start_view import StartView
from custom_ui.dot_editor_view import DotEditorView
from custom_ui.d3_html_view import D3HTMLView
from custom_ui.export_view import ExportView
from custom_ui.custom_widgets import BottomOperationInterfaceWrapper


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # Set default window size
        self.setMinimumSize(1200, 600)

        # global variables with default values
        self.img_generated = False
        self.current_Algorithm = 0

        # Add a view widget for dot edit viewer (.dot Editor)
        self.dotEditorView = DotEditorView(self)

        # Add the experimental interactive HTMLView
        # IF IT IS DECIDED THIS FUNCTIONALITY IS UNNECESSARY: simply ctrl + f [htmlView] and delete all code involving it.
        self.htmlView = D3HTMLView(self)

        # Export view
        self.exportView = ExportView(self)

        # Add a view widget for assigning the necessary column-labels of the csv
        self.columnSelectionView = ColumnSelectionView(self)

        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!ADD YOUR ALGORITHM HERE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # ADD NEW ALGORITHM NAME IN THIS algorithms ARRAY
        # Create your algorithm view page like the heuristicGraphView below.
        # AND THEN append() YOUR ALGORITHMVIEW TO THE algorithmViews ARRAY
        # MAKE SURE THE INDEXING of both arrays match.
        self.algorithms = ["Heuristic Mining",
                           "Fuzzy Miner",
                           "Alpha Miner"]
        self.algorithmViews = []

        # The BottomOperationInterfaceWrapper adds a bottom layout with 2 buttons for mining/loading models.
        self.heuristicGraphView = BottomOperationInterfaceWrapper(
            self, HeuristicGraphView(self, 'saves/0/'), self.algorithms)
        self.algorithmViews.append(self.heuristicGraphView)

        # This is a placeholder for future algorithm views. DELETE OR REPLACE IT
        self.fuzzyGraphView = BottomOperationInterfaceWrapper(
            self, FuzzyGraphView(self, 'saves/1/'), self.algorithms)
        self.algorithmViews.append(self.fuzzyGraphView)

        self.alphaGraphView = BottomOperationInterfaceWrapper(
            self, AlphaGraphView(self, 'saves/2/'), self.algorithms)
        self.algorithmViews.append(self.alphaGraphView)


        # Add a view widget for the default view
        self.startView = BottomOperationInterfaceWrapper(
            self, StartView(self), self.algorithms)

        # Create a main widget that is stacked and can change depending on the needs
        self.mainWidget = QStackedWidget(self)
        self.mainWidget.addWidget(self.startView)
        self.mainWidget.addWidget(self.columnSelectionView)
        self.mainWidget.addWidget(self.dotEditorView)
        self.mainWidget.addWidget(self.htmlView)
        self.mainWidget.addWidget(self.exportView)

        # Add all the algorithm views
        for view in self.algorithmViews:
            self.mainWidget.addWidget(view)

        # Set welcome page as default
        self.mainWidget.setCurrentWidget(self.startView)
        self.setCentralWidget(self.mainWidget)

        # Add a file menu to allow users to upload csv files and so on.
        file_menu = self.menuBar().addMenu("File")
        edit_dot = file_menu.addAction("Edit dot file")
        edit_dot.triggered.connect(self.switch_to_dot_editor)
        d3Graph = file_menu.addAction(
            "Experimental d3-graphviz interactive graph view")  # htmlView
        d3Graph.triggered.connect(self.switch_to_html_view)  # htmlView
        export = file_menu.addAction("Export")
        export.triggered.connect(self.switch_to_export_view)
        self.__update_menu_state()
        # create a status Bar to display quick notifications
        self.statusBar()

        # Set the window title and show the window
        self.setWindowTitle("Graph Viewer")
        self.show()

    # if there is no image, menu_buttons should not be clickable.
    def __update_menu_state(self):
        # Enable or disable the actions based on the state of the variable
        for action in self.menuBar().actions():
            action.setEnabled(self.img_generated)

    # gets called by start_view.py 'create new process' button
    def switch_to_column_selection_view(self, delimiter):

        # Open a file dialog to allow users to select a CSV file
        filename, _ = QFileDialog.getOpenFileName(
            self, "Open CSV File", "tests/", "CSV files (*.csv)")

        # If the user cancels the file dialog, return
        if not filename:
            return

        # Change to Column Selection View
        self.__reset_canvas()
        try:
            self.columnSelectionView.load_csv(filename, delimiter)
        except UndefinedErrorException as e:
            print(e)
            self.show_pop_up_message(str(e))
            return
        self.columnSelectionView.load_algorithms(self.algorithms)
        self.mainWidget.setCurrentWidget(self.columnSelectionView)

    def switch_to_export_view(self):
        if not self.img_generated:
            popup = QMessageBox(self)
            popup.setText("Nothing to export. Please mine a model.")

            close_button = popup.addButton("Close", QMessageBox.AcceptRole)
            close_button.clicked.connect(popup.close)

            # Show the pop-up message
            popup.exec_()

            return

        self.algorithmViews[self.current_Algorithm].generate_png()
        self.exportView.load_algorithm(
            self.algorithmViews[self.current_Algorithm])
        self.mainWidget.setCurrentWidget(self.exportView)

    def switch_to_dot_editor(self):
        loaded = self.dotEditorView.load_file()
        if loaded:
            self.img_generated = True
            self.mainWidget.setCurrentWidget(self.dotEditorView)

    def switch_to_html_view(self):
        try:
            self.htmlView.start_server()
        except FileNotFoundException as e:
            print(e.message)
            self.show_pop_up_message(e.message)
            return
        self.htmlView.load_algorithm(self.algorithmViews[self.current_Algorithm])
        self.mainWidget.setCurrentWidget(self.htmlView)

    # used in export_view.py After export the view should return to the algorithm
    def switch_to_view(self, view):
        self.mainWidget.setCurrentWidget(view)

    # Column Selection View 'Cancel Selection' uses this
    def switch_to_start_view(self):
        self.mainWidget.setCurrentWidget(self.startView)
        self.__reset_canvas()

    # used in ColumnSelectionView
    def mine_new_process(self, filepath, cases, algorithm=0):

        try:
            index = self.algorithmViews[algorithm]
        except IndexError:
            print("main.py: ERROR Algorithm with index " +
                  str(algorithm)+" not defined!")
            return

        self.__reset_canvas()
        self.current_Algorithm = algorithm
        self.algorithmViews[algorithm].startMining(filepath, cases)
        self.img_generated = True
        self.__update_menu_state()
        self.mainWidget.setCurrentWidget(self.algorithmViews[algorithm])

    # used by BottomOperationInterfaceLayoutWidget
    def mine_existing_process(self, algorithm=0):
        try:
            index = self.algorithmViews[algorithm]
        except IndexError:
            print("main.py: ERROR Algorithm with index " +
                  str(algorithm)+" not defined!")
            return

        status = self.algorithmViews[algorithm].loadModel()
        if status == -1:
            return
        self.img_generated = True
        self.__update_menu_state()
        self.current_Algorithm = algorithm
        self.mainWidget.setCurrentWidget(self.algorithmViews[algorithm])

    # shows a quick status update/warning
    def show_pop_up_message(self, message, duration = 3000):
        # create a QLabel widget and set its text
        label = QLabel(message, self)

        # set the label's properties (background color, text color, etc.)
        label.setAutoFillBackground(True)
        label.setStyleSheet(
            'background-color: #ffff99; color: #333333; padding: 5px; border-radius: 3px;')

        # add the label to the status bar
        self.statusBar().addWidget(label)

        # create a timer to hide the label after the specified duration
        timer = QTimer(self)
        # use a lambda function to delete the label
        timer.timeout.connect(lambda: self.__msg_timeout(label))
        timer.start(duration)

    def __msg_timeout(self, label):
        self.statusBar().removeWidget(label)

    def __reset_canvas(self):
        self.dotEditorView.clear()
        # self.startView.clear()
        self.columnSelectionView.clear()
        self.htmlView.clear()
        for view in self.algorithmViews:
            view.clear()

    # overwrite closeEvent function
    def closeEvent(self, event):
        # It is important to shut down the html server.
        self.htmlView.clear()
        self.heuristicGraphView.clear()
        #TODO self.fuzzyGraphView.clear()
        super().closeEvent(event)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    # table header coloring won't work on windows style.
    app.setStyle(QStyleFactory.create('Fusion'))
    window = MainWindow()
    sys.exit(app.exec_())
