import streamlit as st
from abc import ABC
from exceptions.type_exceptions import InvalidTypeException
from logger import get_logger


class BaseView(ABC):
    """Base class for the views. It provides the basic layout and methods for the views."""

    controller = None
    logger = get_logger("BaseView")

    def create_layout(self):
        """Creates the layout for the views."""
        return

    def display_error_message(self, error_message: str):
        """Displays an error message on the page.

        Parameters
        ----------
        error_message : str
            The error message to be displayed.
        """
        st.error(error_message)

    def display_success_message(self, success_message: str):
        """Displays a success message on the page.

        Parameters
        ----------
        success_message : str
            The success message to be displayed.
        """
        st.success(success_message)

    def display_info_message(self, info_message: str):
        """Displays an info message on the page.

        Parameters
        ----------
        info_message : str
            The info message to be displayed.
        """
        st.info(info_message)

    def display_warning_message(self, warning_message: str):
        """Displays a warning message on the page.

        Parameters
        ----------
        warning_message : str
            The warning message to be displayed.
        """
        st.warning(warning_message)

    def display_page_title(self, title: str):
        """Displays the title of the page.

        Parameters
        ----------
        title : str
            The title of the page.
        """
        st.title(title)

    def set_controller(self, controller):
        """Sets the controller for the view.

        Parameters
        ----------
        controller : BaseController
            The controller for the view.

        Raises
        ------
        InvalidTypeException
            If the controller is not a subclass of BaseController.
        """
        from ui.base_ui.base_controller import BaseController

        if not isinstance(controller, BaseController):
            self.logger.error(
                f"Invalid controller type: {type(controller)}, expected: {BaseController}"
            )
            raise InvalidTypeException(BaseController, type(controller))
        self.controller = controller
