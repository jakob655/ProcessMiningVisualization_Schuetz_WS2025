from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QWidget, QHBoxLayout, QPushButton, QTableWidget, QMessageBox, QTableWidgetItem, QWidget, \
    QVBoxLayout
from PyQt5.QtGui import QColor
from api.csv_preprocessor import read
from api.custom_error import BadColumnException, UndefinedErrorException
from custom_ui.custom_widgets import CustomQComboBox
import csv
import pandas as pd


class ColumnSelectionView(QWidget):
    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self.delimiter = None

        # global const variables
        self.eventColor = "#1E90FF"
        self.caseColor = "#00BFFF"
        self.timeColor = "#6495ED"
        self.textColor = "#333333"
        self.defaultColor = "#808080"

        # assign default labels
        self.timeLabel = "timestamp"
        self.eventLabel = "event"
        self.caseLabel = "case"
        self.timeIndex = 0
        self.eventIndex = 1
        self.caseIndex = 2
        self.selected_column = 0
        self.selected_algorithm = 0
        self.filePath = None

        # set up table widget
        self.table = QTableWidget(self)
        self.table.setColumnCount(0)
        self.table.setRowCount(0)
        self.table.horizontalHeader().sectionClicked.connect(self.__column_header_clicked)

        # set up column selector combo box
        self.column_selector = CustomQComboBox()
        self.column_selector.currentIndexChanged.connect(self.__column_selected)

        # set up algorithm selector combo box
        self.algorithm_selector = CustomQComboBox()
        self.algorithm_selector.currentIndexChanged.connect(self.__algorithm_selected)

        # set up assign column buttons
        self.timeColumn_button = QPushButton('Assign to \nTimestamp', self)
        self.timeColumn_button.setFixedSize(100, 70)
        self.timeColumn_button.setStyleSheet(f"background-color: {self.timeColor}; color: {self.textColor};")
        self.timeColumn_button.clicked.connect(self.__assign_timeColumn)

        self.eventColumn_button = QPushButton('Assign to \nEvent', self)
        self.eventColumn_button.setFixedSize(100, 70)
        self.eventColumn_button.setStyleSheet(f"background-color: {self.eventColor}; color: {self.textColor};")
        self.eventColumn_button.clicked.connect(self.__assign_eventColumn)

        self.caseColumn_button = QPushButton('Assign to \nCase', self)
        self.caseColumn_button.setFixedSize(100, 70)
        self.caseColumn_button.setStyleSheet(f"background-color: {self.caseColor}; color: {self.textColor};")
        self.caseColumn_button.clicked.connect(self.__assign_caseColumn)
        # set up start import button
        self.start_import_button = QPushButton('Start Import', self)
        self.start_import_button.setFixedSize(100, 60)
        self.start_import_button.setStyleSheet(f"background-color: lime; color: {self.textColor};")
        self.start_import_button.clicked.connect(self.__start_import)

        # set up top layout
        top_layout = QHBoxLayout()
        top_layout.addWidget(self.column_selector)
        top_layout.addWidget(self.timeColumn_button)
        top_layout.addWidget(self.eventColumn_button)
        top_layout.addWidget(self.caseColumn_button)
        top_layout.setAlignment(Qt.AlignLeft)
        top_layout.setSpacing(10)

        # a return button for cancellation
        self.return_button = QPushButton('Back')
        self.return_button.setStyleSheet(f"background-color: red; color: {self.textColor}")
        self.return_button.setFixedSize(80, 40)
        self.return_button.clicked.connect(self.__return_to_start)

        # set up selector and import button layout
        buttom_layout = QHBoxLayout()
        buttom_layout.addWidget(self.return_button)
        buttom_layout.addWidget(self.algorithm_selector)
        buttom_layout.addWidget(self.start_import_button)
        top_layout.setSpacing(10)

        # set up main_layout
        main_layout = QVBoxLayout(self)
        main_layout.addLayout(top_layout)
        main_layout.addWidget(self.table)
        main_layout.addLayout(buttom_layout)

        # set up spacing and margins
        main_layout.setSpacing(0)
        main_layout.setContentsMargins(10, 10, 10, 10)

    # CALL BEFORE USAGE
    def load_csv(self, filepath, delimiter):
        self.filePath = filepath
        self.delimiter = delimiter
        df = None

        # Try loading with the provided delimiter
        try:
            df = pd.read_csv(filepath, delimiter=self.delimiter, encoding='utf-8-sig')
            # Populate the QTableWidget with data from the pandas DataFrame
            headers = df.columns.tolist()
            self.table.setColumnCount(len(headers))
            self.table.setHorizontalHeaderLabels(headers)
            self.column_selector.addItems(headers)

            for row_index, row_data in df.iterrows():
                self.table.insertRow(row_index)
                for col_index, col_data in enumerate(row_data):
                    self.table.setItem(row_index, col_index, QTableWidgetItem(str(col_data)))

            self.timeLabel = self.table.horizontalHeaderItem(0).text()
            self.eventLabel = self.table.horizontalHeaderItem(1).text()
            self.caseLabel = self.table.horizontalHeaderItem(2).text()
            self.__color_headers()

        except Exception as e:
            df = None
            print(f"An error occurred when loading the csv with the provided delimiter: {e}")

        # If loading with the provided delimiter fails, try with automatic delimiter detection
        if df is None:
            try:
                with open(filepath, 'r', encoding='utf-8-sig') as file:
                    try:
                        dialect = csv.Sniffer().sniff(file.read(1024))
                    except Exception as e:
                        raise UndefinedErrorException("ColumnSelectionView: " + str(e))

                    file.seek(0)

                    df = pd.read_csv(file, delimiter=dialect.delimiter)

                    self.table.setColumnCount(0)
                    self.table.setRowCount(0)

                    headers = df.columns.tolist()
                    self.table.setColumnCount(len(headers))
                    self.table.setHorizontalHeaderLabels(headers)
                    self.column_selector.addItems(headers)

                    for row_index, row_data in df.iterrows():
                        self.table.insertRow(row_index)
                        for col_index, col_data in enumerate(row_data):
                            self.table.setItem(row_index, col_index, QTableWidgetItem(str(col_data)))

                    self.timeLabel = self.table.horizontalHeaderItem(0).text()
                    self.eventLabel = self.table.horizontalHeaderItem(1).text()
                    self.caseLabel = self.table.horizontalHeaderItem(2).text()
                    self.__color_headers()
            except Exception as e:
                df = None
                print(f"An error occurred during automatic delimiter detection: {e}")

        if df is None:
            raise UndefinedErrorException("ColumnSelectionView: Could not determine delimiter")

    # CALL BEFORE USAGE
    def load_algorithms(self, array):
        for element in array:
            self.algorithm_selector.addItem(element)

    def __algorithm_selected(self, index):
        self.algorithm_selector.setCurrentIndex(index)
        self.selected_algorithm = index

    def __column_header_clicked(self, index):
        self.column_selector.setCurrentIndex(index)
        self.selected_column = index

    def __column_selected(self, index):
        self.selected_column = index

    def __assign_timeColumn(self):
        self.timeLabel = self.table.horizontalHeaderItem(self.selected_column).text()
        self.timeIndex = self.selected_column
        self.__color_headers()
        print(self.timeLabel + " assigned as time column")

    def __assign_caseColumn(self):
        self.caseLabel = self.table.horizontalHeaderItem(self.selected_column).text()
        self.caseIndex = self.selected_column
        self.__color_headers()
        print(self.caseLabel + " assigned as case column")

    def __assign_eventColumn(self):
        self.eventLabel = self.table.horizontalHeaderItem(self.selected_column).text()
        self.eventIndex = self.selected_column
        self.__color_headers()
        print(self.eventLabel + " assigned as event column")

    def __color_headers(self):
        for i in range(self.table.columnCount()):
            if self.timeIndex == i:
                self.table.horizontalHeaderItem(i).setBackground(QColor(self.timeColor))
            elif self.eventIndex == i:
                self.table.horizontalHeaderItem(i).setBackground(QColor(self.eventColor))
            elif self.caseIndex == i:
                self.table.horizontalHeaderItem(i).setBackground(QColor(self.caseColor))
            else:
                self.table.horizontalHeaderItem(i).setBackground(QColor(self.defaultColor))

    def __start_import(self):
        msgBox = QMessageBox()
        msgBox.setText(
            "Time label is " + self.timeLabel + "\n" + "Case label is " + self.caseLabel + "\n" + "Event label is " + self.eventLabel)
        msgBox.setInformativeText("Are these columns correct?")
        msgBox.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
        msgBox.setDefaultButton(QMessageBox.Cancel)
        ret = msgBox.exec_()

        if ret == QMessageBox.Cancel:
            return

        try:
            cases = read(self.filePath, self.timeLabel, self.caseLabel, self.eventLabel)
        except BadColumnException as e:
            print(e.message)
            return

        if not cases:
            print("ColumnSelectionView: ERROR Something went wrong when reading cases")
            return

        try:
            self.parent.mine_new_process(self.filePath, cases, self.selected_algorithm)
        except Exception as e:
            self.__return_to_start()
            self.__show_error_message(f"An error occurred during process mining: {e}")
            print(f"An error occurred during mine_new_process: {e}")

    def __return_to_start(self):
        self.parent.switch_to_start_view()

    def __show_error_message(self, message):
        error_msg_box = QMessageBox()
        error_msg_box.setIcon(QMessageBox.Critical)
        error_msg_box.setText("Error")
        error_msg_box.setInformativeText(message + "\nTry again with another dataset")
        error_msg_box.setStandardButtons(QMessageBox.Ok)
        error_msg_box.exec_()

    def clear(self):
        self.timeLabel = "timestamp"
        self.caseLabel = "case"
        self.eventLabel = "event"
        self.selected_column = 0
        self.column_selector.clear()
        self.algorithm_selector.clear()
        self.table.clear()
        self.filePath = None
