import streamlit as st


SESSION_DEFAULTS = {
    "is_running": False,
    "model_loaded": False,
    "last_result": None,
    "last_dual_result": None,
    "last_sections": None,
    "last_model_name": None,
    "last_diagnostic_mode": False,
    "report_text": "",
    "extract_requested": False,
}


def init_session_state() -> None:
    for key, value in SESSION_DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = value


def set_running(is_running: bool) -> None:
    st.session_state["is_running"] = is_running


def request_extraction() -> None:
    st.session_state["extract_requested"] = True


def clear_extraction_request() -> None:
    st.session_state["extract_requested"] = False


def save_extraction_result(
    *,
    result: dict,
    dual_result: dict | None,
    sections: object,
    model_name: str,
    diagnostic_mode: bool,
) -> None:
    st.session_state["model_loaded"] = True
    st.session_state["last_result"] = result
    st.session_state["last_dual_result"] = dual_result
    st.session_state["last_sections"] = sections
    st.session_state["last_model_name"] = model_name
    st.session_state["last_diagnostic_mode"] = diagnostic_mode


def get_last_outputs() -> tuple[dict | None, dict | None, object | None, bool]:
    return (
        st.session_state.get("last_result"),
        st.session_state.get("last_dual_result"),
        st.session_state.get("last_sections"),
        st.session_state.get("last_diagnostic_mode", False),
    )
