import streamlit as st
from abc import ABC, abstractmethod
from exceptions.type_exceptions import TypeIsNoneException, InvalidTypeException
from logger import get_logger


class BaseController(ABC):
    """Base class for the controllers. It provides the basic methods for all the controllers."""

    def __init__(self, views=None):
        """Initializes the controller for the views.

        Parameters
        ----------
        views : List[BaseView] | BaseView
            The views for the page. If a list is passed, the first view in the list is selected as the default view.


        Raises
        ------
        TypeIsNoneException
            If no views are provided to the controller.
        InvalidTypeException
            If a view is not a subclass of BaseView.
        """
        self.logger = get_logger("BaseController")

        if views is None:
            raise TypeIsNoneException("No views provided to the controller")

        from ui.base_ui.base_view import BaseView

        if isinstance(views, list) or isinstance(views, tuple):
            for view in views:
                if not isinstance(view, BaseView):
                    self.logger.error(
                        f"Invalid type: {type(view)}, expected: {BaseView}"
                    )
                    raise InvalidTypeException(BaseView, type(view))
            self.views = list(views)
        else:
            if not isinstance(views, BaseView):
                self.logger.error(f"Invalid type: {type(views)}, expected: {BaseView}")
                raise InvalidTypeException(BaseView, type(views))
            self.views = [views]

        self.error_message = None
        self.info_message = None
        self.success_message = None
        self.warning_message = None

    def select_view(self) -> tuple:
        """Selects the view to be displayed. The first view in the list is selected as the default view.
        The method can be overridden in the child class to implement a different view selection logic, if needed.

        Returns
        -------
        tuple[BaseView, int]
            A tuple containing the selected view and the index of the view in the list.
        """
        return self.views[0], 0

    def process_session_state(self):
        """Processes the session state for the messages. The method checks if there are any messages in the session state and assigns them to the corresponding variables."""
        if "error" in st.session_state:
            self.error_message = st.session_state.error
            del st.session_state.error

        if "info" in st.session_state:
            self.info_message = st.session_state.info
            del st.session_state.info

        if "success" in st.session_state:
            self.success_message = st.session_state.success
            del st.session_state.success

        if "warning" in st.session_state:
            self.warning_message = st.session_state.warning
            del st.session_state.warning

    def display_messages(self, view):
        """Displays the messages on the page, if any are present.

        Parameters
        ----------
        view : BaseView
            The view for the controller.
        """
        if self.error_message is not None:
            view.display_error_message(self.error_message)

        if self.info_message is not None:
            view.display_info_message(self.info_message)

        if self.success_message is not None:
            view.display_success_message(self.success_message)

        if self.warning_message is not None:
            view.display_warning_message(self.warning_message)

    @abstractmethod
    def get_page_title(self) -> str:
        """Returns the title of the page. The method must be implemented in the child class.

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

    def start(self):
        """Entry point for the controller. The method first processes the session state, selects the view, sets the controller for the view
        Afterwards it displays the messages, displays the page title, creates the layout, and runs the indidual controller logic.
        """
        self.process_session_state()
        selected_view, pos = self.select_view()
        selected_view.set_controller(self)
        self.display_messages(selected_view)
        selected_view.display_page_title(self.get_page_title())
        selected_view.create_layout()
        self.run(selected_view, pos)

    @abstractmethod
    def run(self, selected_view, index):
        """Runs the controller logic. This method must be implemented in the child class.

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
