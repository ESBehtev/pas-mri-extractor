"""Small Streamlit state helpers for the production extraction flow."""

import streamlit as st


DEFAULTS = {
    "report_text": "",
    "last_result": None,
    "last_error": None,
    "is_running": False,
}


def init_session_state() -> None:
    for key, value in DEFAULTS.items():
        st.session_state.setdefault(key, value)


def save_result(result: dict | None, error: str | None = None) -> None:
    st.session_state["last_result"] = result
    st.session_state["last_error"] = error
