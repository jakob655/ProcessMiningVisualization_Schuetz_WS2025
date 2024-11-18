import streamlit as st


def home_button(
    label: str = "Home", route: str = "Home", use_container_width=False
) -> None:
    """Create a button to navigate to the home page.

    Parameters
    ----------
    label : str, optional
        button label, by default "Home"
    route : str, optional
        route to navigate to, by default "Home"
    use_container_width : bool, optional
        set the button width to the container width, by default False
    """
    st.button(
        label,
        type="secondary",
        on_click=to_home,
        args=(route,),
        use_container_width=use_container_width,
    )


def to_home(route="Home"):
    """Navigate to the home page. If an error message is present, it is displayed.

    Parameters
    ----------
    route : str, optional
        _description_, by default "Home"
    """
    error_message = None
    if "error" in st.session_state:
        error_message = st.session_state.error
    st.session_state.clear()
    if error_message:
        st.session_state.error = error_message
    st.session_state.page = route


def navigation_button(
    label: str,
    route: str,
    type="primary",
    use_container_width=False,
    beforeNavigate=None,
    args=None,
    disabled=False,
    key=None,
) -> None:
    """Create a button to navigate to a different page.

    Parameters
    ----------
    label : str
        button label
    route : str
        route to navigate to
    type : str, optional
        button type, by default "primary"
    use_container_width : bool, optional
        set the button width to the container width, by default False
    beforeNavigate : function, optional
        function to execute before navigating, by default None
    args : tuple, optional
        arguments to pass to the beforeNavigate function, by default None
    disabled : bool, optional
        disable the button, by default False
    key : str, optional
        button key, by default None
    """
    if args is None:
        args = ()
    st.button(
        label,
        type=type,
        on_click=navigate_to,
        args=(route, beforeNavigate, *args),
        use_container_width=use_container_width,
        disabled=disabled,
        key=key,
    )


def navigate_to(route: str, beforeNavigate=None, *args):
    """Navigate to a different page.

    Parameters
    ----------
    route : str
        route to navigate to
    beforeNavigate : function, optional
        function to execute before navigating, by default None
    *args : tuple
        arguments to pass to the beforeNavigate function
    """
    st.session_state.page = route
    if beforeNavigate:
        beforeNavigate(*args)
