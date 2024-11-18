class InvalidTypeException(Exception):
    """Exception raised when the type of the input is not the expected one."""

    def __init__(self, expected_type, received_type):
        self.message = (
            f"Got type '{received_type}', but expected type '{expected_type}'"
        )
        super().__init__(self.message)


class TypeIsNoneException(Exception):
    """Exception raised when the type of the input is None."""

    def __init__(self, message):
        self.message = message
        super().__init__(self.message)
