import streamlit as st

try:
    from llm_risk_helpers import reset_llm_risk_state_values
except ModuleNotFoundError:
    from app.llm_risk_helpers import reset_llm_risk_state_values


SESSION_DEFAULTS = {
    "is_running": False,
    "model_loaded": False,
    "last_result": None,
    "last_dual_result": None,
    "last_sections": None,
    "last_llm_risk_result": None,
    "extraction_result": None,
    "rule_based_result": None,
    "llm_risk_result": None,
    "llm_risk_status": "skipped",
    "llm_risk_errors": [],
    "llm_risk_warnings": [],
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


def build_rule_based_result(result: dict | None) -> dict | None:
    if not result:
        return None

    return {
        "score": result.get("score") or {},
        "predicted_risks": result.get("predicted_risks") or {},
        "recommendation": result.get("recommendation") or {},
        "computed_rationale": result.get("computed_rationale"),
    }


def reset_llm_risk_state(session_state: object | None = None) -> None:
    state = session_state if session_state is not None else st.session_state
    reset_llm_risk_state_values(state)


def save_extraction_result(
    *,
    result: dict,
    dual_result: dict | None,
    sections: object,
    model_name: str,
    diagnostic_mode: bool,
    llm_risk_result: dict | None = None,
) -> None:
    st.session_state["model_loaded"] = True
    st.session_state["last_result"] = result
    st.session_state["last_dual_result"] = dual_result
    st.session_state["last_llm_risk_result"] = llm_risk_result
    st.session_state["extraction_result"] = result
    st.session_state["rule_based_result"] = build_rule_based_result(result)
    st.session_state["llm_risk_result"] = (
        llm_risk_result.get("llm_risk") if llm_risk_result else None
    )
    st.session_state["llm_risk_status"] = (
        llm_risk_result.get("status") if llm_risk_result else "skipped"
    )
    st.session_state["llm_risk_errors"] = (
        llm_risk_result.get("errors") if llm_risk_result else []
    )
    st.session_state["llm_risk_warnings"] = (
        llm_risk_result.get("warnings") if llm_risk_result else []
    )
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


def get_last_llm_risk_output() -> dict | None:
    return st.session_state.get("last_llm_risk_result")
