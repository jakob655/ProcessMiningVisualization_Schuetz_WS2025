import streamlit as st
from ui.base_ui.base_controller import BaseController


class ControllerTemplate(BaseController):
    """Template for creating controllers. The class provides the basic methods for all the controllers. It must be inherited by the child class."""

    # add more parameters as needed to the constructor
    def __init__(self, views=None):
        """Initializes the controller for the views.

        Parameters
        ----------
        views : List[BaseView] | BaseView
            The views for the page. If a list is passed, the first view in the list is selected as the default view.
        """

        if views is None:
            views = []  # add default views here

        super().__init__(views)

    # Optional: add logic to switch between views. only needed if multiple views are present
    def select_view(self) -> tuple:
        """Selects the view to be displayed. The first view in the list is selected as the default view.
        The method can be overridden in the child class to implement a different view selection logic, if needed.

        Returns
        -------
        tuple[BaseView, int]
            A tuple containing the selected view and the index of the view in the list.
        """
        raise NotImplementedError(
            "Method select_view must be implemented in the child class or be removed if not needed"
        )

    def process_session_state(self):
        """Processes the session state. Can read data from the session state and assign it to the corresponding variables or set data in the session state."""
        super().process_session_state()

        raise NotImplementedError(
            "Method process_session_state must be implemented in the child class"
        )

    def get_page_title(self) -> str:
        """Returns the title of the page.

        Returns
        -------
        str
            The title of the page.

        Raises
        ------
        NotImplementedError
            If the method is not implemented by the subclass.
        """
        raise NotImplementedError(
            "Method get_page_title must be implemented in the child class"
        )

    def run(self, selected_view, index):
        """Runs the controller logic. This method must be implemented.

        Parameters
        ----------
        selected_view : BaseView
            The selected view for the controller.
        index : int
            The index of the selected view in the list of views.

        Raises
        ------
        NotImplementedError
            If the method is not implemented by the subclass.
        """
        raise NotImplementedError("Method run must be implemented in the child class")
