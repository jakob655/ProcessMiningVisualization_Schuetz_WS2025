import csv
from streamlit.runtime.uploaded_file_manager import UploadedFile
from exceptions.io_exceptions import UnsupportedFileTypeException
from logger import get_logger


class DetectionModel:

    def __init__(
        self,
        file_type_mappings: dict[str, list[str]] = None,
        mime_type_mappings: dict[str, str] = None,
    ):
        """Initializes the DetectionModel.

        Parameters
        ----------
        file_type_mappings : dict[str, list[str]], optional
            A dictionary that maps file types to their suffixes, by default None

        mime_type_mappings : dict[str, str], optional
            A dictionary that maps file types to their mime types, by default None
        """
        if file_type_mappings is None:
            from config import import_file_types_mapping

            file_type_mappings = import_file_types_mapping

        if mime_type_mappings is None:
            from config import graph_export_mime_types

            mime_type_mappings = graph_export_mime_types

        self.file_type_mappings = file_type_mappings
        self.mime_type_mappings = mime_type_mappings

    def detect_file_type(self, file_path: str | UploadedFile) -> str:
        """Detect the type of a file based on the file extension.

        Parameters
        ----------
        file_path : str | UploadedFile
            The path to the file or the uploaded file

        Returns
        -------
        str
            The detected file type.

        Raises
        ------
        UnsupportedFileTypeException
            If the file type is not supported
        """
        if isinstance(file_path, UploadedFile):
            file_name = file_path.name
        else:
            file_name = file_path

        for file_type, suffixes in self.file_type_mappings.items():
            for suffix in suffixes:
                if file_name.endswith(suffix):
                    return file_type

        raise UnsupportedFileTypeException(file_name.split(".")[-1][1:])

    def detect_delimiter(self, row: str) -> str:
        """Detect the delimiter of a CSV row.

        Parameters
        ----------
        row : str
            A row of a CSV file

        Returns
        -------
        str
            The detected delimiter. If no delimiter is detected, an empty string is returned.
        """
        detected_delimiter = ""
        try:
            dialect = csv.Sniffer().sniff(row)
            detected_delimiter = dialect.delimiter
        except Exception as e:
            logger = get_logger("DetectionModel")
            logger.error(f"Delimiter detection failed")
            logger.info("set delimiter to empty string")

        return detected_delimiter

    def detect_mime_type(self, export_format: str) -> str:
        """Detect the mime type of a file.

        Parameters
        ----------
        export_format : str
            The format of the file

        Returns
        -------
        str
            The detected mime type.

        Raises
        ------
        UnsupportedFileTypeException
            If the export format is not supported
        """

        try:
            return self.mime_type_mappings[export_format.lower()]
        except KeyError:
            raise UnsupportedFileTypeException(export_format)
