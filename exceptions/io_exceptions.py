class InvalidColumnNameException(Exception):
    """Exception raised when the column name is not found in the dataframe."""

    def __init__(self, column_names):
        self.message = f"Column '{column_names}' not found in the dataset"
        super().__init__(self.message)


class UnsupportedFileTypeException(Exception):
    """Exception raised when the file type is not supported."""

    def __init__(self, file_type):
        self.message = f"Unsupported file type '{file_type}'"
        super().__init__(self.message)


class NotImplementedFileTypeException(Exception):
    """Exception raised when the file type is supported but not implemented."""

    def __init__(self, file_type):
        self.message = f"File type '{file_type}' not implemented"
        super().__init__(self.message)
