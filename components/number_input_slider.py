import streamlit as st


def number_input_slider(
    label: str,
    min_value: int | float = None,
    max_value: int | float = None,
    value: int | float = None,
    step: int | float = None,
    key: str = None,
    help: str = None,
    ratio: int | list[int] | None = None,
) -> None:
    """Renders a slider and a number input field.

    Parameters
    ----------
    label : str
        The label of the slider and number input field.
    min_value : int | float, optional
        Minimum value of the slider and number input field, by default None
    max_value : int | float, optional
        Maximum value of the slider and number input field, by default None
    value : int | float, optional
        Default value of the slider and number input field, by default None
    step : int | float, optional
        Step size of the slider, by default None
    key : str, optional
        The key of the slider and number input field, by default None
    help : str, optional
        Help text of the slider, by default None
    ratio : int | list[int] | None, optional
        The ratio of the slider and number input field, by default None
    """

    if ratio is None:
        ratio = [5, 2]

    if isinstance(ratio, list):
        ratio = ratio[:2]

    silder_column, number_input_column = st.columns(ratio)

    with silder_column:
        st.slider(
            label=label,
            min_value=min_value,
            max_value=max_value,
            key=key,
            help=help,
        )

    with number_input_column:
        st.number_input(
            label=" ",
            min_value=min_value,
            max_value=max_value,
            value=st.session_state[key] if key in st.session_state else value,
            key=f"{key}_text_input",
            on_change=set_session_state,
            args=(key, f"{key}_text_input"),
        )


def set_session_state(key: str, number_input_key: str) -> None:
    st.session_state[key] = st.session_state[number_input_key]
